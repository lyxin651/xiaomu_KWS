#!/usr/bin/env python3
"""限制 ONNX Runtime 线程数后运行 run_matrix.py。

多进程并发时 ORT 默认按物理核数开 intra_op 线程池，几路并发就会线程自旋、
互相抢核（实测可慢 20 倍）。用法：

    ORT_THREADS=4 .venv/bin/python scripts/run_limited.py --datasets kws_dataset \
        --models sort --methods WebRTC-NS Wiener-Filter TSDCA-BA --no-resume
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def limit_onnx_threads(n):
    import onnxruntime as ort
    orig = ort.InferenceSession

    def patched(path_or_bytes, sess_options=None, providers=None,
                provider_options=None, **kwargs):
        so = sess_options or ort.SessionOptions()
        so.intra_op_num_threads = n
        so.inter_op_num_threads = 1
        try:
            so.add_session_config_entry('session.intra_op.allow_spinning', '0')
        except Exception:
            pass
        return orig(path_or_bytes, so, providers, provider_options, **kwargs)

    ort.InferenceSession = patched


def main():
    n = int(os.environ.get('ORT_THREADS', '4'))
    limit_onnx_threads(n)
    import run_matrix
    run_matrix.main()


if __name__ == '__main__':
    main()
