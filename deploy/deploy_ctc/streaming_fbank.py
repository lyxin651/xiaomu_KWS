"""Incremental fbank matching PaddleSpeech's default kaldi_fbank settings.

Why numpy: the fbank loop is hot and we want to keep the audio path
free of paddle device switches. Model forward still runs on GPU.
"""
import numpy as np


_MEL_K = 1127.0  # kaldi HTK-style mel constant


def _hz_to_mel(hz):
    return _MEL_K * np.log(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (np.exp(mel / _MEL_K) - 1.0)


def _next_pow2(n):
    return 1 << (int(n) - 1).bit_length()


def _build_mel_basis(sample_rate, n_fft, n_mels, f_min=20.0, f_max=None):
    """Match kaldi's _get_mel_banks: triangular min(up, down) without slaney norm."""
    if f_max is None:
        f_max = sample_rate / 2.0
    # kaldi_fbank appends an all-zero Nyquist bin after building the basis.
    n_freqs = n_fft // 2
    fft_bin_width = sample_rate / n_fft
    mel_low = _hz_to_mel(f_min)
    mel_high = _hz_to_mel(f_max)
    mel_step = (mel_high - mel_low) / (n_mels + 1)
    # mel coordinates of the three triangle vertices per bin
    mel_left = mel_low + np.arange(n_mels, dtype=np.float32) * mel_step
    mel_right = mel_low + (np.arange(n_mels, dtype=np.float32) + 2.0) * mel_step
    # mel coordinate of every fft bin
    fft_mels = _hz_to_mel(fft_bin_width * np.arange(n_freqs, dtype=np.float32))
    # broadcast: left/right (n_mels, 1), fft_mels (1, n_freqs)
    up = (fft_mels[None, :] - mel_left[:, None]) / mel_step
    down = (mel_right[:, None] - fft_mels[None, :]) / mel_step
    weights = np.clip(np.minimum(up, down), 0.0, 1.0).astype(np.float32)
    return weights


def _build_povey_window(frame_length, n_fft):
    """Match PaddleSpeech's symmetric Hann window raised to 0.85."""
    n = np.arange(frame_length, dtype=np.float32)
    base = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (frame_length - 1))
    win = base ** 0.85
    padded = np.zeros(n_fft, dtype=np.float32)
    padded[:frame_length] = win
    return padded


class StreamingFbank:
    """Incremental fbank. Each push() emits a list of new mel frames."""

    def __init__(self, sample_rate=16000, frame_length_ms=25.0,
                 frame_shift_ms=10.0, n_mels=80, n_fft=None,
                 preemphasis=0.97, energy_floor=1e-7):
        self.sr = sample_rate
        self.frame_length = int(round(sample_rate * frame_length_ms / 1000.0))
        self.frame_shift = int(round(sample_rate * frame_shift_ms / 1000.0))
        self.n_mels = n_mels
        self.n_fft = n_fft or _next_pow2(self.frame_length)
        self.preemphasis = preemphasis
        self.energy_floor = energy_floor
        self.mel_basis = _build_mel_basis(sample_rate, self.n_fft, n_mels)
        self.window = _build_povey_window(self.frame_length, self.n_fft)
        self.reset()

    def reset(self):
        self.buffer = np.zeros(self.n_fft, dtype=np.float32)
        self.buffer_filled = 0
        self.samples_since_frame = 0

    def push(self, pcm_chunk):
        """Returns a list of new 80-dim mel frames (empty if not ready).

        Order matches kaldi: store RAW samples; on frame emission:
          1) DC removal on raw samples
          2) preemphasis
          3) window + FFT
        """
        out = []
        for sample in pcm_chunk:
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = float(sample)
            self.buffer_filled = min(self.buffer_filled + 1, self.n_fft)
            self.samples_since_frame += 1
            if (self.samples_since_frame >= self.frame_shift
                    and self.buffer_filled >= self.frame_length):
                self.samples_since_frame = 0
                out.append(self._compute_frame())
        return out

    def _compute_frame(self):
        # 1) raw frame + DC removal on raw samples (kaldi order)
        frame = self.buffer[-self.frame_length:].copy()
        frame = frame - frame.mean()
        # 2) preemphasis along the frame (kaldi uses previous-sample: y[n] = x[n] - 0.97*x[n-1])
        preemph = np.empty_like(frame)
        preemph[0] = (1.0 - self.preemphasis) * frame[0]
        preemph[1:] = frame[1:] - self.preemphasis * frame[:-1]
        # 3) window + zero-pad to n_fft
        windowed = np.zeros(self.n_fft, dtype=np.float32)
        windowed[:self.frame_length] = preemph * self.window[:self.frame_length]
        spectrum = np.fft.rfft(windowed)
        power = spectrum.real ** 2 + spectrum.imag ** 2  # kaldi |X(k)|^2, no /n_fft
        mel = power[:-1] @ self.mel_basis.T
        mel = np.maximum(mel, self.energy_floor).astype(np.float32)
        return np.log(mel)

    def compute_offline(self, waveform):
        """Helper: compute fbank for a full waveform via repeated push()."""
        frames = []
        chunk = max(self.frame_shift, 1024)
        for start in range(0, len(waveform), chunk):
            frames.extend(self.push(waveform[start:start + chunk]))
        return np.stack(frames) if frames else np.zeros((0, self.n_mels), dtype=np.float32)
