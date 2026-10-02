"""
Build Script for Spectral DMA Micro-Kernel
==========================================
Automatically compiles `spectral_dma_kernel.c` into a shared library (.dll / .so / .dylib)
using gcc, clang, or MSVC cl.
"""

import os
import platform
import subprocess
import sys


def build():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    c_source = os.path.join(script_dir, "spectral_dma_kernel.c")

    system = platform.system().lower()
    if "windows" in system:
        lib_name = "spectral_dma_kernel.dll"
    elif "darwin" in system:
        lib_name = "libspectral_dma_kernel.dylib"
    else:
        lib_name = "libspectral_dma_kernel.so"

    output_path = os.path.join(script_dir, lib_name)
    print(f"[*] Building {lib_name} from {os.path.basename(c_source)}...")

    env = os.environ.copy()
    if "windows" in system:
        if os.path.exists("C:\\msys64\\mingw64\\bin"):
            env["PATH"] = "C:\\msys64\\mingw64\\bin;" + env.get("PATH", "")

    compilers = [
        "C:\\msys64\\mingw64\\bin\\gcc.exe" if os.path.exists("C:\\msys64\\mingw64\\bin\\gcc.exe") else "gcc",
        "gcc",
        "clang",
        "cl"
    ]
    chosen_compiler = None
    for cc in compilers:
        try:
            res = subprocess.run([cc, "--version"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0:
                chosen_compiler = cc
                break
        except FileNotFoundError:
            continue

    if chosen_compiler is None:
        print("[!] Error: No C compiler (gcc, clang, cl) found in PATH.")
        sys.exit(1)

    print(f"[*] Found compiler: {chosen_compiler}")
    machine = platform.machine().lower()
    is_x86 = any(arch in machine for arch in ["x86_64", "amd64", "x86", "i386", "i686"])

    if "cl" not in chosen_compiler.lower():
        cmd = [
            chosen_compiler,
            "-O3",
            "-shared",
        ]
        if is_x86:
            cmd.extend(["-mavx2", "-mfma"])
        if "windows" not in system:
            cmd.extend(["-fPIC", "-pthread", "-lm"])
        cmd.extend([
            "-o",
            output_path,
            c_source,
        ])
    else:  # cl (MSVC)
        cmd = [
            "cl",
            "/O2",
            "/LD",
            c_source,
            f"/Fe:{output_path}",
        ]

    print(f"[*] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    if result.returncode == 0:
        print(f"[+] Successfully compiled: {output_path}")
        return output_path
    else:
        print(f"[!] Compilation failed with code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    build()
