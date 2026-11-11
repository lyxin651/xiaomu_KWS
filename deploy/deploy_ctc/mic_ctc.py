#!/usr/bin/env python3
"""CTC-KWS 麦克风唤醒测试（单阶段，v2_verifier 的 backbone+CTC 头）。

模型：hybrid v2_verifier(ep7 best) 多任务训练、单头部署——ONNX 只含
backbone+CTC 头(输出逐帧 6-token log-probs,CMVN 烤入图内)，INT8。

打分：KeywordPathMatcher(与训练侧 ctc.KeywordPathMatcher 逐位一致的
纯 numpy 拷贝)——单调关键词路径似然比，score = 完成路径的
mean per-token log-LR(nats/token)。score > 0 表示关键词路径比
"无关键词"解释更可信。

TOKEN LIST(必须与训练侧一致):
  vocab   = ['blank', 'fill', 'x', 'iao', 'm', 'u']   # id 0-5
  keyword = ['x','iao','m','u','x','iao','m','u']     # 小木小木 -> [2,3,4,5,2,3,4,5]
阈值口径：v2_verifier dev 工作点 thr=-0.10(0.5FA/h);更灵敏可降到 -1。

用法：
    python mic_ctc.py --list-devices
    python mic_ctc.py --device 2
    python mic_ctc.py --device 2 --threshold -1.0   # 更灵敏
    python mic_ctc.py --wav some.wav                # 离线 wav 测试
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from streaming_fbank import StreamingFbank  # noqa: E402

FRAME_SHIFT_MS = 10
SAMPLE_RATE = 16000
CHUNK_MS = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000

# ---- TOKEN LIST（与训练侧 conf/mdtc_xiaomu_ctc_rnnt_verifier.yaml 一致）----
VOCAB = ['blank', 'fill', 'x', 'iao', 'm', 'u']
BLANK, FILL = 0, 1
KEYWORD_TOKENS = [2, 3, 4, 5, 2, 3, 4, 5]  # x iao m u x iao m u


class KeywordPathMatcher:
    """单调关键词路径打分器（paddlespeech.kws.models.ctc 同款拷贝）。

    每帧要么停留(cost = log P(blank)+P(fill))要么发射下一个关键词 token;
    score = (最优完成路径 - 同帧全停留基线) / 关键词长度。
    """

    def __init__(self, keyword_tokens=KEYWORD_TOKENS, blank=BLANK, fill=FILL):
        self.kw = list(keyword_tokens)
        self.length = len(self.kw)
        self.blank = blank
        self.fill = fill
        self.reset()

    def reset(self):
        self.best = np.full(self.length, -np.inf, dtype=np.float64)
        self.cumstay = 0.0
        self.completed = -np.inf
        self.end_frame = -1
        self.total_frames = 0

    def push(self, logp):
        stay = np.logaddexp(logp[self.blank], logp[self.fill])
        shifted = np.concatenate(
            ([self.cumstay], self.best[:-1])) + logp[self.kw]
        self.cumstay += stay
        self.best = np.maximum(self.best + stay, shifted)
        self.total_frames += 1
        done = self.best[-1] - self.cumstay
        if done > self.completed:
            self.completed = done
            self.end_frame = self.total_frames - 1

    @property
    def score(self):
        return self.completed / self.length


class CtcKwsSession:
    """流式 CTC KWS:fbank chunk -> ONNX log_probs -> matcher。

    触发后 matcher 复位（路径被消费），并进入 cooldown 静默期。
    """

    def __init__(self, model_path, threshold=-0.10, cooldown_ms=500.0):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(
            str(model_path), providers=['CPUExecutionProvider'])
        names = [i.name for i in self.sess.get_inputs()]
        if names != ['x', 'cache']:
            raise ValueError('unexpected ONNX inputs {}'.format(names))
        init_path = str(model_path).replace('.onnx', '_init_cache.npy')
        self.cache = np.load(init_path).astype(np.float32)
        out_shape = self.sess.get_outputs()[0].shape
        print('model: {} (output {}, vocab={})'.format(
            Path(model_path).name, out_shape, VOCAB))
        self.threshold = threshold
        self.cooldown_frames = int(round(cooldown_ms / FRAME_SHIFT_MS))
        self.matcher = KeywordPathMatcher()
        self.cooldown = 0
        self.total_frames = 0
        self.wakes = 0

    def push_frames(self, frames):
        """frames: [n, 80] raw fbank（CMVN 在图内）。返回触发事件列表。"""
        events = []
        if len(frames) == 0:
            return events
        x = np.asarray(frames, dtype=np.float32)[None]      # [1, n, 80]
        log_probs, self.cache = self.sess.run(
            None, {'x': x, 'cache': self.cache})
        for logp in log_probs[0]:
            self.total_frames += 1
            self.cooldown = max(0, self.cooldown - 1)
            self.matcher.push(logp)
            if self.matcher.score >= self.threshold and self.cooldown == 0:
                events.append({
                    'time_ms': (self.total_frames - 1) * FRAME_SHIFT_MS,
                    'score': float(self.matcher.score),
                })
                self.wakes += 1
                self.cooldown = self.cooldown_frames
                self.matcher.reset()   # 路径已消费,重新开始匹配
        return events


class StreamResampler:
    """任意源采样率 -> 16k 的分块流式重采样(带跨块状态,无 scipy 依赖)。

    整数倍抽取(如 48k->16k, 32k->16k):每组求均值——自带抗混叠;
    非整数比(如 44.1k->16k):线性插值(KWS 频带内足够)。
    """

    def __init__(self, src_rate, dst_rate=SAMPLE_RATE):
        if src_rate <= 0:
            raise ValueError('invalid source rate')
        self.src_rate, self.dst_rate = src_rate, dst_rate
        self.integer = (src_rate % dst_rate == 0)
        self.factor = src_rate // dst_rate if self.integer else None
        self.ratio = dst_rate / src_rate
        self.buf = np.zeros(0, dtype=np.float32)
        self.pos = 0.0  # 下一个输出样本在 buf 中的位置(线性插值用)

    def push(self, samples):
        self.buf = np.concatenate([self.buf, np.asarray(samples, np.float32)])
        if self.integer:
            n = len(self.buf) // self.factor
            if n == 0:
                return np.zeros(0, dtype=np.float32)
            out = self.buf[:n * self.factor].reshape(
                [n, self.factor]).mean(axis=1)
            self.buf = self.buf[n * self.factor:]
            return out
        outs = []
        while self.pos + 1.0 < len(self.buf):
            i = int(self.pos)
            frac = self.pos - i
            outs.append(self.buf[i] * (1.0 - frac) + self.buf[i + 1] * frac)
            self.pos += 1.0 / self.ratio
        keep = int(self.pos)  # 保留插值所需的最尾部
        self.buf = self.buf[keep:]
        self.pos -= keep
        return np.asarray(outs, dtype=np.float32)


def run_mic(session_factory, device=None):
    import sounddevice as sd
    kws = session_factory()
    fbank = StreamingFbank(sample_rate=SAMPLE_RATE,
                           frame_length_ms=25, frame_shift_ms=10, n_mels=80)
    # 设备常不支持 16k 直开(USB/内置麦原生 44.1k/48k):
    # 按设备默认采样率开流,回调内重采样到 16k
    dev = device if device is not None else sd.default.device[0]
    info = sd.query_devices(dev)
    src_rate = int(info['default_samplerate'])
    resampler = None
    if src_rate != SAMPLE_RATE:
        resampler = StreamResampler(src_rate)
        print('mic sample rate: {} -> resample to {}'.format(
            src_rate, SAMPLE_RATE))
    blocksize = int(src_rate * CHUNK_MS / 1000)
    print('listening... (Ctrl-C 退出)  threshold={} nats/token'.format(
        kws.threshold))

    def callback(indata, frames, time_info, status):
        if status:
            print('audio status:', status, file=sys.stderr)
        wav = indata[:, 0].astype(np.float32)
        if resampler is not None:
            wav = resampler.push(wav)
            if len(wav) == 0:
                return
        frames_out = fbank.push(wav)
        for event in kws.push_frames(frames_out):
            print('[{:.1f}s] WAKE  score={:+.3f}  (wake #{})'.format(
                event['time_ms'] / 1000.0, event['score'], kws.wakes))

    with sd.InputStream(samplerate=src_rate, blocksize=blocksize,
                        channels=1, dtype='float32', device=device,
                        callback=callback):
        while True:
            time.sleep(0.2)


def run_wav(session_factory, wav_path):
    import soundfile as sf
    kws = session_factory()
    fbank = StreamingFbank(sample_rate=SAMPLE_RATE,
                           frame_length_ms=25, frame_shift_ms=10, n_mels=80)
    wav, sr = sf.read(wav_path, dtype='float32')
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != SAMPLE_RATE:
        import resampy
        wav = resampy.resample(wav, sr, SAMPLE_RATE)
    n_frames = 0
    for pos in range(0, len(wav), CHUNK_SAMPLES):
        frames = fbank.push(wav[pos:pos + CHUNK_SAMPLES])
        for event in kws.push_frames(frames):
            n_frames = kws.total_frames
            print('[{:.1f}s] WAKE  score={:+.3f}'.format(
                event['time_ms'] / 1000.0, event['score']))
    print('done: {} frames, {} wakes'.format(kws.total_frames, kws.wakes))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    here = Path(__file__).resolve().parent
    p.add_argument('--model',
                   default=str(here / 'kws_ctc_v2verifier_e7_int8.onnx'))
    p.add_argument('--threshold', type=float, default=-0.10,
                   help='关键词路径分数阈值(nats/token, dev 工作点 -0.10)')
    p.add_argument('--cooldown-ms', type=float, default=500.0)
    p.add_argument('--device', type=int, default=None)
    p.add_argument('--list-devices', action='store_true')
    p.add_argument('--wav', default=None, help='离线测试 wav')
    args = p.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    def factory():
        return CtcKwsSession(args.model, threshold=args.threshold,
                             cooldown_ms=args.cooldown_ms)

    if args.wav:
        run_wav(factory, args.wav)
    else:
        run_mic(factory, device=args.device)


if __name__ == '__main__':
    main()

