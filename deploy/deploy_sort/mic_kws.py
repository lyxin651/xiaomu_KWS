#!/usr/bin/env python3
"""本地麦克风流式唤醒词检测（"小木小木"），纯 ONNX 推理，无 paddle 依赖。

部署包内容（本目录）：
    kws_cmvn_g1_e100.onnx            流式模型（CMVN 已烤入图首层，喂 raw fbank 即可）
    kws_cmvn_g1_e100_init_cache.npy  cold-start 初始 cache（不能用零初始化！）
    global_cmvn.npz                  CMVN 统计（仅参考，运行时不需要——已在图内）
    streaming_fbank.py               增量 fbank 前端（与训练侧 kaldi_fbank 数值一致）
    mic_kws.py                       本文件

用法：
    # 列出音频设备
    python mic_kws.py --list-devices

    # 默认设备，阈值 0.84（epoch_100 全量评估的推荐工作点）
    python mic_kws.py

    # 指定设备 + 参数
    python mic_kws.py --device 2 --threshold 0.5 --chunk-ms 30

依赖：numpy, onnxruntime, sounddevice (conda install -c conda-forge portaudio)
"""
import argparse
import queue
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from streaming_fbank import StreamingFbank  # noqa: E402

SAMPLE_RATE = 16000
N_MELS = 80
FRAME_SHIFT_MS = 10.0
FRAME_LENGTH_MS = 25.0


class IntegerDecimator:
    """状态化抗混叠整数倍降采样（如 48000→16000，ratio=3）。

    纯 numpy FIR 低通 + 抽取，跨 chunk 保留滤波器状态，无边界伪影。
    """

    def __init__(self, ratio, numtaps=95):
        self.ratio = int(ratio)
        n = np.arange(numtaps) - (numtaps - 1) / 2
        h = np.sinc(n / self.ratio) * np.hamming(numtaps) / self.ratio
        self.h = h / h.sum()          # DC 增益 = 1
        self.zi = np.zeros(numtaps - 1)

    def process(self, x):
        buf = np.concatenate([self.zi, x.astype(np.float64)])
        y = np.convolve(buf, self.h, mode='full')[len(self.h) - 1:]
        self.zi = buf[-(len(self.h) - 1):]
        return y[::self.ratio][:len(x) // self.ratio].astype(np.float32)


def open_input_stream(sd, device, chunk_samples):
    """打开 16kHz 等效输入流；设备不支持 16k 时自动换原生采样率 + 降采样。

    尝试顺序：16000 → 48000(÷3) → 44100(soxr) → 设备默认率。
    整数倍用内置 FIR 降采样器；44100/16000 非整数倍需 soxr（pip install soxr）。

    返回 (stream, get_pcm, actual_native_rate)；get_pcm() 阻塞取 16k PCM 块。
    """
    dev = device if device is not None else sd.default.device[0]
    native = int(sd.query_devices(dev)['default_samplerate'])
    candidates = [16000, 48000, 44100, native]      # 48000 整数倍优先

    for rate in dict.fromkeys(candidates):          # 去重保序
        ratio = rate / SAMPLE_RATE
        integer = abs(ratio - round(ratio)) <= 1e-6

        if rate == SAMPLE_RATE:
            resample, block = None, chunk_samples
        elif integer:
            dec = IntegerDecimator(int(round(ratio)))
            resample, block = dec.process, chunk_samples * int(round(ratio))
        else:
            try:
                import soxr
            except ImportError:
                continue
            rs = soxr.ResampleStream(rate, SAMPLE_RATE, 1, dtype='float32')
            resample, block = rs.resample_chunk, chunk_samples * round(ratio)

        q = queue.Queue()

        def make_cb(resample, q):
            def wrapped(indata, frames_n, time_info, status):
                if status:
                    print(f'[audio] {status}', file=sys.stderr)
                x = indata[:, 0].copy()
                y = resample(x) if resample is not None else x
                if len(y):
                    q.put(y)
            return wrapped

        try:
            stream = sd.InputStream(samplerate=rate, channels=1,
                                    dtype='float32', blocksize=block,
                                    device=device,
                                    callback=make_cb(resample, q))
        except sd.PortAudioError:
            continue
        stream.__enter__()
        if rate != SAMPLE_RATE:
            print(f'[info] 设备不支持 16kHz，以 {rate}Hz 采集 + '
                  f'{"FIR 整数倍" if integer else "soxr"}降采样')
        return stream, q.get, rate

    raise RuntimeError(
        f'无法以 16000/48000/44100/{native}Hz 打开输入设备 {dev}。'
        f'试默认设备（去掉 --device）、安装 soxr（pip install soxr），'
        f'或用 arecord -L 查看设备支持的采样率')


class OnnxKWS:
    """ONNX 流式唤醒检测：cache 逐层增量推理 + 触发规则。

    触发规则与训练侧评估完全一致：
        连续 min_trigger_frames 帧得分 >= threshold → 唤醒；
        唤醒后 cooldown_ms 内不再触发。
    """

    def __init__(self, model_path, threshold=0.84, cooldown_ms=500.0,
                 min_trigger_frames=3, score_smooth=3):
        self.sess = ort.InferenceSession(
            str(model_path), providers=['CPUExecutionProvider'])
        names = [i.name for i in self.sess.get_inputs()]
        if names != ['x', 'cache']:
            raise ValueError(f'unexpected ONNX inputs {names}')
        init_path = str(model_path).replace('.onnx', '_init_cache.npy')
        self.cache = np.load(init_path).astype(np.float32)
        self.threshold = threshold
        self.cooldown_frames = int(round(cooldown_ms / FRAME_SHIFT_MS))
        self.min_trigger_frames = min_trigger_frames
        self.recent = deque(maxlen=score_smooth)
        self.cooldown = 0
        self.total_frames = 0
        self.wakes = 0

    def push_frames(self, frames):
        """frames: list/array [n, 80] raw fbank（不做 CMVN，图内已含）。"""
        events = []
        if len(frames) == 0:
            return events
        x = np.asarray(frames, dtype=np.float32)[None]   # [1, n, 80]
        probs, self.cache = self.sess.run(None, {'x': x, 'cache': self.cache})
        for score in probs[0, :, 0]:
            self.total_frames += 1
            self.cooldown = max(0, self.cooldown - 1)
            self.recent.append(float(score))
            if self.cooldown > 0:
                continue
            if (len(self.recent) >= self.min_trigger_frames
                    and all(s >= self.threshold
                            for s in list(self.recent)[-self.min_trigger_frames:])):
                self.wakes += 1
                events.append({
                    'time_ms': (self.total_frames - 1) * FRAME_SHIFT_MS,
                    'score': float(np.mean(list(self.recent)[-self.min_trigger_frames:])),
                    'peak': float(max(self.recent)),
                })
                self.cooldown = self.cooldown_frames
        return events


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    here = Path(__file__).resolve().parent
    p.add_argument('--model', default=str(here / 'kws_cmvn_g1_e100.onnx'))
    p.add_argument('--threshold', type=float, default=0.84)
    p.add_argument('--cooldown-ms', type=float, default=500.0)
    p.add_argument('--min-trigger-frames', type=int, default=3)
    p.add_argument('--chunk-ms', type=float, default=30.0)
    p.add_argument('--device', type=int, default=None,
                   help='输入设备编号（--list-devices 查看）')
    p.add_argument('--list-devices', action='store_true')
    args = p.parse_args()

    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        return

    kws = OnnxKWS(args.model, threshold=args.threshold,
                  cooldown_ms=args.cooldown_ms,
                  min_trigger_frames=args.min_trigger_frames)
    fbank = StreamingFbank(sample_rate=SAMPLE_RATE,
                           frame_length_ms=FRAME_LENGTH_MS,
                           frame_shift_ms=FRAME_SHIFT_MS,
                           n_mels=N_MELS)

    chunk = int(round(SAMPLE_RATE * args.chunk_ms / 1000.0))
    t_start = time.time()

    print('=' * 60)
    print('小木小木 · 流式唤醒（ONNX + CMVN）')
    print('=' * 60)
    dev_name = sd.query_devices(args.device if args.device is not None
                                else sd.default.device[0])['name']
    print(f'模型    : {Path(args.model).name}')
    print(f'麦克风  : {dev_name}')
    print(f'阈值    : {args.threshold}  (连续 {args.min_trigger_frames} 帧, '
          f'冷却 {args.cooldown_ms:.0f}ms)')
    print(f'chunk   : {args.chunk_ms:.0f}ms / {chunk} samples @ {SAMPLE_RATE}Hz')
    print('说 "小木小木" 唤醒，Ctrl+C 退出')
    print('=' * 60)

    try:
        stream, get_pcm, actual_sr = open_input_stream(sd, args.device, chunk)
        try:
            while True:
                pcm = get_pcm()
                frames = fbank.push(pcm)
                if not frames:
                    continue
                for ev in kws.push_frames(frames):
                    wall = datetime.now().strftime('%H:%M:%S')
                    print(f'>>> [唤醒 #{kws.wakes}] {wall}  '
                          f't=+{ev["time_ms"]/1000:.2f}s  '
                          f'score={ev["score"]:.3f}  peak={ev["peak"]:.3f}')
        finally:
            stream.close()
    except KeyboardInterrupt:
        dur = time.time() - t_start
        print(f'\n退出。运行 {dur:.0f}s，处理 {kws.total_frames} 帧，'
              f'唤醒 {kws.wakes} 次')


if __name__ == '__main__':
    main()

