"""
Base-3 Quantum Trit Quantization & .tritq Binary Serialization
=============================================================
Implements hierarchical 4-band spectral quantization and base-3 packaging:
  - 5 balanced trits {-1, 0, +1} packed into 1 byte (3^5 = 243 <= 256) -> 1.60 bits/trit.
  - Effective bpp depends on matrix geometry (M×N) and quantizer radii (r0, r1, r2):
      · Default (r0=0.10, r1=0.25, r2=0.50): ~0.47 bpp (~68× on 256×1024)
      · Aggressive (r0=0.10, r1=0.30, r2=0.75): ~0.91 bpp (~35× on 256×1024)
      · Calibrated TinyStories-10M / MNIST (r0=0.15, r1=0.40, r2=1.00): ~1.64 bpp (~19.5×)
      · Mid-Band (r0=0.20, r1=0.55, r2=1.15): ~2.18 bpp (~14.7×)
    See paper Table 5 and §3.3 for the full empirical rate–distortion frontier.
"""

import json
import struct
from typing import Any, BinaryIO, Dict, Optional, Tuple

import numpy as np
import torch


class Base3TritQuantizer:
    """
    Quantizes 2D-DCT spectral coefficients into 4 hierarchical frequency bands.

    Default radii (r0=0.10, r1=0.25, r2=0.50) yield ~0.47 bpp (~68×).
    The calibrated TinyStories-10M & MNIST setting (r0=0.15, r1=0.40, r2=1.00)
    yields ~1.64 bpp (~19.5×). See paper Table 5 for the full rate–distortion sweep:
      (0.08, 0.22, 0.50) -> 0.43 bpp, (0.10, 0.30, 0.75) -> 0.91 bpp,
      (0.15, 0.40, 1.00) -> 1.64 bpp, (0.20, 0.55, 1.15) -> 2.18 bpp,
      (0.30, 0.70, 1.30) -> 2.79 bpp, (0.45, 0.90, 1.41) -> 3.77 bpp.

      - Band 0 (rho <= r0): DC / Basal harmonics -> 8-bit uint (8 bpp)
      - Band 1 (r0 < rho <= r1): Mid frequencies -> 4-bit nibbles (4 bpp)
      - Band 2 (r1 < rho <= r2): High-mid frequencies -> Base-3 Trits (1.6 bpp)
      - Band 3 (rho > r2): High frequencies -> Truncated to zero (0 bpp)
    """
    def __init__(
        self,
        r0: float = 0.10,
        r1: float = 0.25,
        r2: float = 0.50,
        band2_mode: str = "lloyd_max",
    ):
        if not (0.0 <= r0 <= r1 <= r2):
            raise ValueError(
                f"Quantizer radii must satisfy 0 <= r0 <= r1 <= r2, got ({r0}, {r1}, {r2})"
            )
        if band2_mode not in {"lloyd_max", "legacy"}:
            raise ValueError(
                f"Unknown band2_mode: {band2_mode}. Expected 'lloyd_max' or 'legacy'."
            )
        self.r0 = float(r0)
        self.r1 = float(r1)
        self.r2 = float(r2)
        self.band2_mode = band2_mode
        self._lut = self._build_trit_lut()

    @staticmethod
    def _build_trit_lut() -> np.ndarray:
        """
        Builds a 256 x 5 float32 lookup table for instant O(1) byte-to-trit unpacking:
            byte_val = t0 + t1*3 + t2*9 + t3*27 + t4*81, where t in {0, 1, 2} maps to {-1, 0, +1}
        """
        lut = np.zeros((256, 5), dtype=np.float32)
        for b in range(243):
            val = b
            for i in range(5):
                trit = val % 3
                val //= 3
                lut[b, i] = float(trit - 1)  # 0 -> -1.0, 1 -> 0.0, 2 -> +1.0
        return lut

    def compute_radial_grid(self, m: int, n: int, device: torch.device) -> torch.Tensor:
        """
        Computes normalized Euclidean radial frequency coordinates in [0, sqrt(2)).
        """
        u = torch.arange(m, device=device, dtype=torch.float32).unsqueeze(1) / float(m)
        v = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(0) / float(n)
        return torch.sqrt(u.pow(2) + v.pow(2))

    def quantize_matrix(self, spec: torch.Tensor) -> Dict[str, Any]:
        """
        Quantizes a 2D spectral tensor into compressed packed buffers.
        """
        if spec.dim() != 2:
            raise ValueError(f"Expected 2D spectral tensor, got shape {tuple(spec.shape)}")
        m, n = spec.shape
        rho = self.compute_radial_grid(m, n, spec.device)

        mask0 = rho <= self.r0
        mask1 = (rho > self.r0) & (rho <= self.r1)
        mask2 = (rho > self.r1) & (rho <= self.r2)

        # Band 0: 8-bit
        vals0 = spec[mask0]
        if vals0.numel() > 0:
            min0, max0 = vals0.min().item(), vals0.max().item()
            scale0 = (max0 - min0) / 255.0 if max0 > min0 else 1.0
            q0 = torch.clamp(((vals0 - min0) / (scale0 + 1e-12)).round(), 0, 255).to(torch.uint8)
        else:
            min0, scale0 = 0.0, 1.0
            q0 = torch.empty(0, dtype=torch.uint8, device=spec.device)

        # Band 1: 4-bit nibbles (2 values per byte)
        vals1 = spec[mask1]
        if vals1.numel() > 0:
            min1, max1 = vals1.min().item(), vals1.max().item()
            scale1 = (max1 - min1) / 15.0 if max1 > min1 else 1.0
            q1_raw = torch.clamp(((vals1 - min1) / (scale1 + 1e-12)).round(), 0, 15).to(torch.uint8)
            if len(q1_raw) % 2 != 0:
                q1_raw = torch.cat([q1_raw, torch.zeros(1, dtype=torch.uint8, device=spec.device)])
            q1_packed = (q1_raw[0::2] << 4) | (q1_raw[1::2] & 0x0F)
        else:
            min1, scale1 = 0.0, 1.0
            q1_packed = torch.empty(0, dtype=torch.uint8, device=spec.device)

        # Band 2: Base-3 trits (5 trits per byte)
        vals2 = spec[mask2]
        if vals2.numel() > 0:
            if self.band2_mode == "lloyd_max":
                # 1D Lloyd-Max optimal symmetric ternary quantization:
                # strictly minimizes Band-2 MSE and guarantees reconstructed energy <= 100%.
                abs_v = vals2.abs()
                tau = 0.7 * abs_v.mean().item()
                scale2 = 0.0
                for _ in range(3):
                    active = abs_v > tau
                    if active.any():
                        scale2 = abs_v[active].mean().item()
                        tau = 0.5 * scale2
                    else:
                        scale2 = 0.0
                        break
                threshold = tau
            else:
                scale2 = vals2.std().item() if vals2.numel() > 1 else (vals2.abs().mean().item() or 1.0)
                threshold = 0.5 * scale2

            # Map {-1, 0, +1} -> {0, 1, 2}
            trits = torch.ones_like(vals2, dtype=torch.int64)  # default 0 (code 1)
            if scale2 > 0.0:
                trits[vals2 > threshold] = 2   # +1
                trits[vals2 < -threshold] = 0  # -1

            # Pack into uint8 bytes
            rem = (5 - (len(trits) % 5)) % 5
            if rem > 0:
                trits = torch.cat([trits, torch.ones(rem, dtype=torch.int64, device=spec.device)])

            t0 = trits[0::5]
            t1 = trits[1::5] * 3
            t2 = trits[2::5] * 9
            t3 = trits[3::5] * 27
            t4 = trits[4::5] * 81
            q2_packed = (t0 + t1 + t2 + t3 + t4).to(torch.uint8)
        else:
            scale2 = 1.0
            q2_packed = torch.empty(0, dtype=torch.uint8, device=spec.device)

        total_bytes = q0.numel() + q1_packed.numel() + q2_packed.numel()
        bpp = (total_bytes * 8.0) / max(spec.numel(), 1)

        return {
            "shape": (m, n),
            "bpp": bpp,
            "band0": {"data": q0.cpu().numpy(), "min": min0, "scale": scale0, "mask": mask0.cpu().numpy()},
            "band1": {"data": q1_packed.cpu().numpy(), "min": min1, "scale": scale1, "mask": mask1.cpu().numpy(), "count": vals1.numel()},
            "band2": {"data": q2_packed.cpu().numpy(), "scale": scale2, "mask": mask2.cpu().numpy(), "count": vals2.numel()},
        }

    def dequantize_matrix(self, packed: Dict[str, Any], device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Unpacks and reconstructs the full 2D spectral tensor from compressed buffers.
        """
        m, n = packed["shape"]
        rec = torch.zeros((m, n), dtype=torch.float32, device=device)

        # Band 0: Unpack 8-bit
        b0 = packed["band0"]
        if len(b0["data"]) > 0:
            q0 = torch.from_numpy(b0["data"]).to(device=device, dtype=torch.float32)
            vals0 = q0 * b0["scale"] + b0["min"]
            mask0 = torch.from_numpy(b0["mask"]).to(device=device)
            rec[mask0] = vals0

        # Band 1: Unpack 4-bit nibbles
        b1 = packed["band1"]
        if b1["count"] > 0:
            q1_packed = torch.from_numpy(b1["data"]).to(device=device)
            high = (q1_packed >> 4) & 0x0F
            low = q1_packed & 0x0F
            q1_unpacked = torch.stack([high, low], dim=1).view(-1)[:b1["count"]].float()
            vals1 = q1_unpacked * b1["scale"] + b1["min"]
            mask1 = torch.from_numpy(b1["mask"]).to(device=device)
            rec[mask1] = vals1

        # Band 2: Unpack Base-3 trits via LUT
        b2 = packed["band2"]
        if b2["count"] > 0:
            q2_bytes = b2["data"]
            unpacked_trits = self._lut[q2_bytes].reshape(-1)[:b2["count"]]
            vals2 = torch.from_numpy(unpacked_trits).to(device=device) * b2["scale"]
            mask2 = torch.from_numpy(b2["mask"]).to(device=device)
            rec[mask2] = vals2

        return rec


class TritQFormat:
    """
    Serializes and deserializes entire Transformer checkpoints in the .tritq binary format.

    By default, embedding and vocabulary projection tensors (matching ``exclude_substrings``)
    are stored uncompressed in FP16 rather than spectrally quantized, preserving token geometry.
    """
    MAGIC = b"TRITQ_V1"
    DEFAULT_EXCLUDE_SUBSTRINGS: Tuple[str, ...] = ("emb", "embed", "lm_head", "wte", "wpe")
    MAX_HEADER_BYTES = 1_000_000
    MAX_TENSORS = 10_000
    MAX_MATRIX_ELEMENTS = 500_000_000

    @staticmethod
    def _read_exact(f: BinaryIO, nbytes: int) -> bytes:
        data = f.read(nbytes)
        if len(data) != nbytes:
            raise ValueError(f"Truncated .tritq file: expected {nbytes} bytes, got {len(data)}")
        return data

    @classmethod
    def _should_compress(cls, name: str, tensor: torch.Tensor, exclude_substrings: Tuple[str, ...]) -> bool:
        if tensor.dim() != 2:
            return False
        lname = name.lower()
        return not any(sub in lname for sub in exclude_substrings)

    @classmethod
    def save(
        cls,
        filepath: str,
        model_weights: Dict[str, torch.Tensor],
        config_dict: Dict[str, Any],
        quantizer: Optional[Base3TritQuantizer] = None,
        exclude_substrings: Optional[Tuple[str, ...]] = None,
    ):
        if quantizer is None:
            quantizer = Base3TritQuantizer()
        excl = cls.DEFAULT_EXCLUDE_SUBSTRINGS if exclude_substrings is None else tuple(s.lower() for s in exclude_substrings)

        with open(filepath, "wb") as f:
            f.write(cls.MAGIC)
            header_meta = {
                "config": config_dict,
                "r0": float(quantizer.r0),
                "r1": float(quantizer.r1),
                "r2": float(quantizer.r2),
                "version": 1,
            }
            cfg_bytes = json.dumps(header_meta).encode("utf-8")
            f.write(struct.pack("<I", len(cfg_bytes)))
            f.write(cfg_bytes)

            f.write(struct.pack("<I", len(model_weights)))
            for name, w in model_weights.items():
                name_bytes = name.encode("utf-8")
                f.write(struct.pack("<H", len(name_bytes)))
                f.write(name_bytes)
                if cls._should_compress(name, w, excl):
                    from .spectral import dct2d
                    spec = dct2d(w.float())
                    packed = quantizer.quantize_matrix(spec)
                    # Serialized flag: 1 = compressed spectral
                    f.write(struct.pack("<B", 1))
                    f.write(struct.pack("<II", packed["shape"][0], packed["shape"][1]))
                    # Band 0
                    b0_data = packed["band0"]["data"].tobytes()
                    f.write(struct.pack("<Iff", len(b0_data), packed["band0"]["min"], packed["band0"]["scale"]))
                    f.write(b0_data)
                    # Band 1
                    b1_data = packed["band1"]["data"].tobytes()
                    f.write(struct.pack("<IffI", len(b1_data), packed["band1"]["min"], packed["band1"]["scale"], packed["band1"]["count"]))
                    f.write(b1_data)
                    # Band 2
                    b2_data = packed["band2"]["data"].tobytes()
                    f.write(struct.pack("<IfI", len(b2_data), packed["band2"]["scale"], packed["band2"]["count"]))
                    f.write(b2_data)
                else:
                    # Uncompressed parameter (1D norms/biases or excluded 2D embeddings/lm_head)
                    raw_fp16 = w.detach().half().cpu().numpy().tobytes()
                    f.write(struct.pack("<B", 0))
                    f.write(struct.pack("<I", len(w.shape)))
                    for s in w.shape:
                        f.write(struct.pack("<I", s))
                    f.write(struct.pack("<I", len(raw_fp16)))
                    f.write(raw_fp16)

    @classmethod
    def load(
        cls,
        filepath: str,
        device: torch.device = torch.device("cpu"),
        quantizer: Optional[Base3TritQuantizer] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        """
        Loads a .tritq checkpoint and reconstructs all weight tensors.

        If *quantizer* is None, radii are taken from the file header
        (r0, r1, r2 saved at write time). If a custom quantizer is passed,
        its radii must match the serialized band counts.

        Returns:
            (config_dict, weights_dict) where config_dict is the model config
            and weights_dict maps tensor names to reconstructed torch.Tensors.
        """
        from .spectral import idct2d

        with open(filepath, "rb") as f:
            magic = cls._read_exact(f, 8)
            if magic != cls.MAGIC:
                raise ValueError(f"Invalid .tritq file: expected magic {cls.MAGIC!r}, got {magic!r}")

            (cfg_len,) = struct.unpack("<I", cls._read_exact(f, 4))
            if cfg_len > cls.MAX_HEADER_BYTES:
                raise ValueError(f"Invalid .tritq header: cfg_len {cfg_len} exceeds 1 MB limit")
            cfg_str = cls._read_exact(f, cfg_len).decode("utf-8")
            try:
                header_meta = json.loads(cfg_str)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid .tritq header: expected JSON metadata, got unparseable content. "
                    f"JSON error: {e}"
                ) from e

            if isinstance(header_meta, dict) and "config" in header_meta:
                config_dict = header_meta["config"]
                file_r0 = float(header_meta.get("r0", 0.10))
                file_r1 = float(header_meta.get("r1", 0.25))
                file_r2 = float(header_meta.get("r2", 0.50))
            else:
                config_dict = header_meta
                file_r0, file_r1, file_r2 = 0.10, 0.25, 0.50

            if quantizer is None:
                quantizer = Base3TritQuantizer(r0=file_r0, r1=file_r1, r2=file_r2)

            (num_tensors,) = struct.unpack("<I", cls._read_exact(f, 4))
            if num_tensors > cls.MAX_TENSORS:
                raise ValueError(f"Invalid .tritq header: num_tensors {num_tensors} exceeds 10k limit")
            weights: Dict[str, torch.Tensor] = {}

            for _ in range(num_tensors):
                (name_len,) = struct.unpack("<H", cls._read_exact(f, 2))
                name = cls._read_exact(f, name_len).decode("utf-8")
                (flag,) = struct.unpack("<B", cls._read_exact(f, 1))

                if flag == 1:
                    (M, N) = struct.unpack("<II", cls._read_exact(f, 8))
                    if M == 0 or N == 0 or M * N > cls.MAX_MATRIX_ELEMENTS:
                        raise ValueError(f"Invalid compressed tensor shape ({M}, {N}) for '{name}'")

                    (b0_len, b0_min, b0_scale) = struct.unpack("<Iff", cls._read_exact(f, 12))
                    if b0_len > M * N:
                        raise ValueError(f"Invalid Band 0 byte length {b0_len} for '{name}'")
                    b0_data = np.frombuffer(cls._read_exact(f, b0_len), dtype=np.uint8).copy()

                    (b1_len, b1_min, b1_scale, b1_count) = struct.unpack("<IffI", cls._read_exact(f, 16))
                    if b1_count > M * N or b1_len != (b1_count + 1) // 2:
                        raise ValueError(f"Invalid Band 1 header ({b1_len} bytes, {b1_count} count) for '{name}'")
                    b1_data = np.frombuffer(cls._read_exact(f, b1_len), dtype=np.uint8).copy()

                    (b2_len, b2_scale, b2_count) = struct.unpack("<IfI", cls._read_exact(f, 12))
                    if b2_count > M * N or b2_len != (b2_count + 4) // 5:
                        raise ValueError(f"Invalid Band 2 header ({b2_len} bytes, {b2_count} count) for '{name}'")
                    b2_data = np.frombuffer(cls._read_exact(f, b2_len), dtype=np.uint8).copy()

                    rho = quantizer.compute_radial_grid(M, N, torch.device("cpu"))
                    mask0 = (rho <= quantizer.r0).numpy()
                    mask1 = ((rho > quantizer.r0) & (rho <= quantizer.r1)).numpy()
                    mask2 = ((rho > quantizer.r1) & (rho <= quantizer.r2)).numpy()

                    if b0_len != int(mask0.sum()) or b1_count != int(mask1.sum()) or b2_count != int(mask2.sum()):
                        raise ValueError(
                            f"Quantizer radii ({quantizer.r0}, {quantizer.r1}, {quantizer.r2}) do not match "
                            f"serialized band counts for '{name}'"
                        )

                    packed = {
                        "shape": (M, N),
                        "band0": {"data": b0_data, "min": b0_min, "scale": b0_scale, "mask": mask0},
                        "band1": {"data": b1_data, "min": b1_min, "scale": b1_scale, "mask": mask1, "count": b1_count},
                        "band2": {"data": b2_data, "scale": b2_scale, "mask": mask2, "count": b2_count},
                    }

                    rec_spec = quantizer.dequantize_matrix(packed, device=device)
                    weights[name] = idct2d(rec_spec)

                elif flag == 0:
                    (ndim,) = struct.unpack("<I", cls._read_exact(f, 4))
                    if ndim > 8:
                        raise ValueError(f"Invalid tensor ndim {ndim} for '{name}'")
                    shape = []
                    numel = 1
                    for _ in range(ndim):
                        (s,) = struct.unpack("<I", cls._read_exact(f, 4))
                        shape.append(s)
                        numel *= s
                    if numel > cls.MAX_MATRIX_ELEMENTS:
                        raise ValueError(f"Invalid raw tensor size {numel} for '{name}'")
                    (raw_len,) = struct.unpack("<I", cls._read_exact(f, 4))
                    if raw_len != numel * 2:
                        raise ValueError(f"Invalid FP16 byte length {raw_len} for shape {shape} in '{name}'")
                    raw_bytes = cls._read_exact(f, raw_len)
                    arr = np.frombuffer(raw_bytes, dtype=np.float16).copy()
                    weights[name] = torch.from_numpy(arr).float().reshape(shape).to(device)
                else:
                    raise ValueError(f"Unknown serialization flag {flag} for '{name}'")

        return config_dict, weights

