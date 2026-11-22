"""
Master End-to-End Reproducibility Script
========================================
Runs and verifies core empirical and engineering claims of the Dirichlet
Regularization (`dreg`) repository in a single reproducible execution:
  1. Full automated unit test suite verification
  2. Native C DMA Kernel float32 numerical parity check
  3. Real training proof (Standard vs Dirichlet MLP): roughness, 2D-DCT compaction & TritQ error
  4. Comparative controls on the trained weights: Random Orthogonal Rotation & SVD truncation
  5. Executive verification summary report
"""

import os
import subprocess
import sys
import time

# Ensure repository root is importable regardless of working directory / invocation path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from dreg import (  # noqa: E402
    Base3TritQuantizer,
    DirichletLoss,
    dct2d,
    dirichlet_energy_2d,
    idct2d,
    random_orthogonal_transform_2d,
    svd_low_rank_approximation,
)


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def step_1_run_test_suite():
    print_header("[Step 1/4] Running Complete Unit Test Suite via Pytest...")
    t0 = time.perf_counter()
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"], cwd=ROOT_DIR, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    summary_line = result.stdout.strip().splitlines()[-1].strip("= ") if result.stdout.strip() else ""
    print(result.stdout.strip())
    if result.returncode != 0:
        print("[!] Unit tests failed:")
        print(result.stderr)
        return False, summary_line
    print(f"[+] Unit test suite PASSED in {elapsed:.2f}s.")
    return True, summary_line


def step_2_verify_c_kernel_parity():
    print_header("[Step 2/4] Verifying C DMA Kernel Numerical Parity...")
    try:
        from examples.benchmark_c_dma import dct_matrix_1d, load_c_kernel

        c_lib = load_c_kernel()
        c_lib.c_spectral_init()

        M, N = 128, 128
        torch.manual_seed(42)
        orig_w = torch.randn(M, N, dtype=torch.float32)
        spec = dct2d(orig_w)
        quantizer = Base3TritQuantizer(r0=0.15, r1=0.40, r2=1.00)
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
            dct_buf.ctypes.data, temp_buf.ctypes.data, target_buf.ctypes.data,
        )

        c_reconstructed = target_buf.reshape(M, N)
        max_diff = float(np.max(np.abs(ref_w - c_reconstructed)))
        c_lib.c_spectral_cleanup()

        print(f"[*] Maximum Absolute Numerical Discrepancy (C vs PyTorch): {max_diff:.8f}")
        passed = max_diff < 1e-5
        if passed:
            print("[+] PARITY VERIFIED: Float32 match verified (< 1e-5).")
        else:
            print("[!] Warning: Parity tolerance exceeded.")
        return passed, max_diff
    except Exception as e:
        print(f"[!] C kernel parity test skipped or failed with error: {e}")
        return False, None


def _train_demo_pair():
    """Trains two matched 2-layer MLPs (Standard vs Dirichlet) on structured data."""
    torch.manual_seed(42)
    in_dim, hid_dim, num_classes = 128, 128, 10
    centroids = torch.randn(num_classes, in_dim) * 2.0
    y_train = torch.randint(0, num_classes, (1024,))
    x_train = centroids[y_train] + torch.randn(1024, in_dim) * 0.8

    def make_model():
        torch.manual_seed(1337)
        return nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=False),
            nn.GELU(),
            nn.Linear(hid_dim, num_classes, bias=False),
        )

    models = {}
    for label, lam in [("Standard (AdamW)", 0.0), ("Dirichlet (dreg)", 20.0)]:
        net = make_model()
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        dloss = DirichletLoss(weight_decay=lam)
        for _ in range(12):
            for i in range(0, len(x_train), 64):
                bx, by = x_train[i : i + 64], y_train[i : i + 64]
                opt.zero_grad()
                loss = crit(net(bx), by)
                if lam > 0.0:
                    loss = loss + dloss(net)
                loss.backward()
                opt.step()
        models[label] = net
    return models["Standard (AdamW)"], models["Dirichlet (dreg)"]


def step_3_measure_spectral_mechanism():
    print_header("[Step 3/4] Measuring Emergent Spectral Compaction (Trained Standard vs Dirichlet MLP)...")
    std_net, topo_net = _train_demo_pair()
    w_std = std_net[0].weight.detach()
    w_topo = topo_net[0].weight.detach()

    ed_std = dirichlet_energy_2d(w_std, normalization="edges").item()
    ed_topo = dirichlet_energy_2d(w_topo, normalization="edges").item()

    quantizer = Base3TritQuantizer(r0=0.15, r1=0.40, r2=1.00)
    print(f"{'Trained Model':<20} | {'Edge E_D':<10} | {'E(rho<=0.4)':<12} | {'TritQ bpp':<10} | {'Rel. L2 Err'}")
    print("-" * 76)

    stats = {}
    for name, w in [("Standard (AdamW)", w_std), ("Dirichlet (dreg)", w_topo)]:
        spec = dct2d(w)
        rho = quantizer.compute_radial_grid(*w.shape, w.device)
        low_energy = ((spec[rho <= 0.40] ** 2).sum() / (spec ** 2).sum()).item()
        packed = quantizer.quantize_matrix(spec)
        rec = idct2d(quantizer.dequantize_matrix(packed))
        rel_err = (rec - w).norm().item() / w.norm().item()
        ed = dirichlet_energy_2d(w, normalization="edges").item()
        stats[name] = {"ed": ed, "low_energy": low_energy, "bpp": packed["bpp"], "rel_err": rel_err}
        print(f"{name:<20} | {ed:<10.6f} | {low_energy*100:<11.1f}% | {packed['bpp']:<10.3f} | {rel_err*100:.1f}%")

    compaction_ratio = stats["Dirichlet (dreg)"]["low_energy"] / max(stats["Standard (AdamW)"]["low_energy"], 1e-6)
    passed = (ed_topo < ed_std) and (stats["Dirichlet (dreg)"]["rel_err"] < stats["Standard (AdamW)"]["rel_err"])
    print(
        f"\n[+] Emergent Dirichlet Effect: E_D reduced by {(1 - ed_topo / ed_std) * 100:.1f}%, "
        f"low-freq energy {compaction_ratio:.2f}x higher, TritQ error reduced from "
        f"{stats['Standard (AdamW)']['rel_err']*100:.1f}% -> {stats['Dirichlet (dreg)']['rel_err']*100:.1f}%."
    )
    return passed, w_topo, compaction_ratio


def step_4_measure_baselines_comparison(w_topo: torch.Tensor):
    print_header("[Step 4/4] Comparing Controls on Trained Topographic Weights (Rotation vs SVD)...")
    quantizer = Base3TritQuantizer(r0=0.15, r1=0.40, r2=1.00)

    # Control 1: Random orthogonal rotation breaks spatial smoothness while preserving Frobenius norm
    w_rot, _, _ = random_orthogonal_transform_2d(w_topo, seed=42)
    ed_orig = dirichlet_energy_2d(w_topo, normalization="edges").item()
    ed_rot = dirichlet_energy_2d(w_rot, normalization="edges").item()

    p_orig = quantizer.quantize_matrix(dct2d(w_topo))
    err_orig = (idct2d(quantizer.dequantize_matrix(p_orig)) - w_topo).norm().item() / w_topo.norm().item()

    p_rot = quantizer.quantize_matrix(dct2d(w_rot))
    err_rot = (idct2d(quantizer.dequantize_matrix(p_rot)) - w_rot).norm().item() / w_rot.norm().item()

    # Control 2: Budget-matched FP32 SVD truncation at the same compression ratio
    ratio = 32.0 / max(p_orig["bpp"], 1e-6)
    w_svd, rank = svd_low_rank_approximation(w_topo, compression_ratio=ratio)
    svd_err = (w_svd - w_topo).norm().item() / w_topo.norm().item()

    print(f"[*] Topographic Weights (Native):    Edge E_D={ed_orig:.6f} | TritQ ({p_orig['bpp']:.2f} bpp) Rel. Err={err_orig*100:.1f}%")
    print(f"[*] Random Orthogonal Rotation:      Edge E_D={ed_rot:.6f} | TritQ ({p_rot['bpp']:.2f} bpp) Rel. Err={err_rot*100:.1f}%")
    print(f"[*] Budget-Matched FP32 SVD ({ratio:.1f}x): Rank k={rank}          | Rel. Err={svd_err*100:.1f}%")
    print("[+] Controls computed successfully (rotation destroys spatial smoothness and raises DCT error).")
    return ed_rot > ed_orig and err_rot > err_orig


def main():
    print_header("DIRICHLET REGULARIZATION: MASTER REPRODUCIBILITY SUITE")
    print("Executing all empirical verifications end-to-end...\n")

    t_start = time.perf_counter()
    ok_tests, test_summary = step_1_run_test_suite()
    ok_kernel, c_diff = step_2_verify_c_kernel_parity()
    ok_spectral, w_topo, compaction_ratio = step_3_measure_spectral_mechanism()
    ok_baselines = step_4_measure_baselines_comparison(w_topo)
    t_total = time.perf_counter() - t_start

    print_header("REPRODUCIBILITY VERIFICATION SUMMARY")
    print(f"  1. Unit Tests Suite:                   {'[PASSED]' if ok_tests else '[FAILED]'} ({test_summary})")
    print(f"  2. C DMA Kernel Precision Parity:      {'[PASSED] (diff=' + f'{c_diff:.2e})' if ok_kernel else '[FAILED]'}")
    print(f"  3. Emergent 2D-DCT Spectral Check:     {'[PASSED]' if ok_spectral else '[FAILED]'} ({compaction_ratio:.2f}x low-freq energy)")
    print(f"  4. Orthogonal Rotation & SVD Controls: {'[PASSED]' if ok_baselines else '[FAILED]'}")
    print("-" * 80)
    print(f"Total Verification Elapsed Time: {t_total:.2f} seconds.")
    print("=" * 80)

    if not (ok_tests and ok_kernel and ok_spectral and ok_baselines):
        sys.exit(1)


if __name__ == "__main__":
    main()
