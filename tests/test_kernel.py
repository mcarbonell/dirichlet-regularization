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

        c_lib.c_spectral_register_matrix.argtypes = [
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_float,
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        c_lib.c_spectral_register_matrix.restype = ctypes.c_int

        c_lib.c_spectral_decode_sublayer_sync.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        ]
        c_lib.c_spectral_decode_sublayer_sync.restype = None

        c_lib.c_spectral_dma_start_prefetch.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        ]
        c_lib.c_spectral_dma_start_prefetch.restype = None

        c_lib.c_spectral_dma_wait_prefetch.argtypes = []
        c_lib.c_spectral_dma_wait_prefetch.restype = None
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

    def test_async_dma_prefetch_and_reinit_lifecycle(self):
        """Tests sublayer registration, async DMA ping-pong prefetch, and safe re-init."""
        c_lib = get_kernel_lib()
        if c_lib is None:
            pytest.skip("C kernel shared library not available on this platform")

        M, N = 32, 32
        torch.manual_seed(7)
        quantizer = Base3TritQuantizer(r0=0.15, r1=0.40, r2=1.0)
        d_m = dct_matrix_1d(M)
        d_n = dct_matrix_1d(N)
        d_out_t_np = d_m.t().contiguous().numpy()
        d_in_np = d_n.contiguous().numpy()

        mats = []
        refs = []
        for _ in range(2):
            w = torch.randn(M, N, dtype=torch.float32)
            packed = quantizer.quantize_matrix(dct2d(w))
            refs.append(idct2d(quantizer.dequantize_matrix(packed)).numpy())
            mats.append({
                "packed": packed,
                "b0": packed["band0"]["data"].tobytes(),
                "b1": packed["band1"]["data"].tobytes(),
                "b2": packed["band2"]["data"].tobytes(),
                "m0": np.where(packed["band0"]["mask"].reshape(-1))[0].astype(np.int32),
                "m1": np.where(packed["band1"]["mask"].reshape(-1))[0].astype(np.int32),
                "m2": np.where(packed["band2"]["mask"].reshape(-1))[0].astype(np.int32),
            })

        assert c_lib.c_spectral_init() == 0
        try:
            # Invalid registration checks
            assert c_lib.c_spectral_register_matrix(
                -1, 0, M, N, 0, 0, 0, 0.0, 1.0, 0.0, 1.0, 1.0,
                b"", b"", b"", 0, 0, 0, 0, 0, 0
            ) == -1
            assert c_lib.c_spectral_register_matrix(
                0, 99, M, N, 0, 0, 0, 0.0, 1.0, 0.0, 1.0, 1.0,
                b"", b"", b"", 0, 0, 0, 0, 0, 0
            ) == -2
            assert c_lib.c_spectral_register_matrix(
                0, 0, 0, N, 0, 0, 0, 0.0, 1.0, 0.0, 1.0, 1.0,
                b"", b"", b"", 0, 0, 0, 0, 0, 0
            ) == -3

            # Register 2 matrices in sublayer 0 at consecutive offsets
            for idx, item in enumerate(mats):
                p = item["packed"]
                rc = c_lib.c_spectral_register_matrix(
                    0, idx, M, N,
                    len(item["m0"]), len(item["m1"]), len(item["m2"]),
                    float(p["band0"]["min"]), float(p["band0"]["scale"]),
                    float(p["band1"]["min"]), float(p["band1"]["scale"]),
                    float(p["band2"]["scale"]),
                    item["b0"], item["b1"], item["b2"],
                    item["m0"].ctypes.data, item["m1"].ctypes.data, item["m2"].ctypes.data,
                    d_out_t_np.ctypes.data, d_in_np.ctypes.data,
                    idx * M * N,
                )
                assert rc == 0

            dct_buf = np.zeros(M * N, dtype=np.float32)
            temp_buf = np.zeros(M * N, dtype=np.float32)
            target_buf = np.zeros(2 * M * N, dtype=np.float32)

            # Async DMA prefetch
            c_lib.c_spectral_dma_start_prefetch(0, target_buf.ctypes.data, dct_buf.ctypes.data, temp_buf.ctypes.data)
            c_lib.c_spectral_dma_wait_prefetch()

            out0 = target_buf[: M * N].reshape(M, N)
            out1 = target_buf[M * N :].reshape(M, N)
            assert np.max(np.abs(out0 - refs[0])) < 1e-5
            assert np.max(np.abs(out1 - refs[1])) < 1e-5

            # Re-init regression test (must clear previous 2-matrix sublayer metadata)
            assert c_lib.c_spectral_init() == 0
            p0 = mats[0]["packed"]
            assert c_lib.c_spectral_register_matrix(
                0, 0, M, N,
                len(mats[0]["m0"]), len(mats[0]["m1"]), len(mats[0]["m2"]),
                float(p0["band0"]["min"]), float(p0["band0"]["scale"]),
                float(p0["band1"]["min"]), float(p0["band1"]["scale"]),
                float(p0["band2"]["scale"]),
                mats[0]["b0"], mats[0]["b1"], mats[0]["b2"],
                mats[0]["m0"].ctypes.data, mats[0]["m1"].ctypes.data, mats[0]["m2"].ctypes.data,
                d_out_t_np.ctypes.data, d_in_np.ctypes.data,
                0,
            ) == 0
            target_single = np.zeros(M * N, dtype=np.float32)
            c_lib.c_spectral_decode_sublayer_sync(0, target_single.ctypes.data, dct_buf.ctypes.data, temp_buf.ctypes.data)
            assert np.max(np.abs(target_single.reshape(M, N) - refs[0])) < 1e-5
        finally:
            c_lib.c_spectral_cleanup()
            # Second cleanup must be a safe no-op
            c_lib.c_spectral_cleanup()
