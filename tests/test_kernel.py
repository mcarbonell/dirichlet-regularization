"""
Tests for native C Spectral DMA Kernel and Python numerical parity.
"""

import ctypes
import os

import numpy as np
import pytest
import torch

from dreg.quantization import Base3TritQuantizer
from dreg.spectral import dct2d, dct_matrix_1d, idct2d


def get_kernel_lib():
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kernel_dir = os.path.join(script_dir, "kernel")

    # Try finding existing compiled library
    candidates = [
        os.path.join(kernel_dir, "spectral_dma_kernel.dll"),
        os.path.join(kernel_dir, "libspectral_dma_kernel.so"),
        os.path.join(kernel_dir, "libspectral_dma_kernel.dylib"),
    ]
    lib_path = next((p for p in candidates if os.path.exists(p)), None)

    if lib_path is None:
        try:
            from kernel.build_kernel import build
            lib_path = build()
        except Exception:
            return None

    try:
        c_lib = ctypes.CDLL(lib_path)
        c_lib.c_spectral_init.argtypes = []
        c_lib.c_spectral_init.restype = ctypes.c_int

        c_lib.c_spectral_cleanup.argtypes = []
        c_lib.c_spectral_cleanup.restype = None

        c_lib.c_spectral_decode_matrix_direct.argtypes = [
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_float,
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        ]
        c_lib.c_spectral_decode_matrix_direct.restype = None
        return c_lib
    except Exception:
        return None


class TestCKernelParity:
    def test_c_pytorch_parity(self):
        c_lib = get_kernel_lib()
        if c_lib is None:
            pytest.skip("C kernel shared library not available on this platform")

        c_lib.c_spectral_init()
        try:
            M, N = 64, 64
            torch.manual_seed(123)
            orig_w = torch.randn(M, N, dtype=torch.float32)
            spec = dct2d(orig_w)
            quantizer = Base3TritQuantizer()
            packed = quantizer.quantize_matrix(spec)

            ref_reconstructed = quantizer.dequantize_matrix(packed)
            ref_w = idct2d(ref_reconstructed).numpy()

            d_m = dct_matrix_1d(M)
            d_n = dct_matrix_1d(N)
            d_out_t_np = d_m.t().contiguous().numpy()
            d_in_np = d_n.contiguous().numpy()

            b0_bytes = packed["band0"]["data"].tobytes()
            b1_bytes = packed["band1"]["data"].tobytes()
            b2_bytes = packed["band2"]["data"].tobytes()

            m0_idx = np.where(packed["band0"]["mask"].reshape(-1))[0].astype(np.int32)
            m1_idx = np.where(packed["band1"]["mask"].reshape(-1))[0].astype(np.int32)
            m2_idx = np.where(packed["band2"]["mask"].reshape(-1))[0].astype(np.int32)

            dct_buf = np.zeros(M * N, dtype=np.float32)
            temp_buf = np.zeros(M * N, dtype=np.float32)
            target_buf = np.zeros(M * N, dtype=np.float32)

            c_lib.c_spectral_decode_matrix_direct(
                M, N,
                len(m0_idx), len(m1_idx), len(m2_idx),
                float(packed["band0"]["min"]),
                float(packed["band0"]["scale"]),
                float(packed["band1"]["min"]),
                float(packed["band1"]["scale"]),
                float(packed["band2"]["scale"]),
                b0_bytes, b1_bytes, b2_bytes,
                m0_idx.ctypes.data, m1_idx.ctypes.data, m2_idx.ctypes.data,
                d_out_t_np.ctypes.data, d_in_np.ctypes.data,
                dct_buf.ctypes.data, temp_buf.ctypes.data, target_buf.ctypes.data
            )

            c_reconstructed = target_buf.reshape(M, N)
            max_diff = float(np.max(np.abs(ref_w - c_reconstructed)))

            # Absolute discrepancy must be within machine precision for float32
            assert max_diff < 1e-5, f"Discrepancy too high: {max_diff}"
        finally:
            c_lib.c_spectral_cleanup()
