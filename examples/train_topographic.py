"""
Example: Train Topographic Transformer with Dirichlet Harmonic Pinning
======================================================================
Demonstrates end-to-end training of a causal Transformer where all linear
weights are constrained to a continuous 2D cortical lattice.
"""

import math
import time

import torch
import torch.optim as optim

from dreg import TopographicConfig, TopographicTransformer


def get_synthetic_data(vocab_size=256, seq_len=128, num_samples=1000):
    """Generates synthetic structured Markovian sequences for fast testing."""
    gen = torch.Generator().manual_seed(42)
    data = torch.zeros((num_samples, seq_len), dtype=torch.long)
    data[:, 0] = torch.randint(0, vocab_size, (num_samples,), generator=gen)
    for t in range(1, seq_len):
        mask = torch.rand((num_samples,), generator=gen) > 0.30
        structured = (data[:, t - 1] * 7 + 13) % vocab_size
        rand_tok = torch.randint(0, vocab_size, (num_samples,), generator=gen)
        data[:, t] = torch.where(mask, structured, rand_tok)
    return data


def main():
    print("=" * 70)
    print(" TRAINING TOPOGRAPHIC TRANSFORMER WITH DIRICHLET HARMONIC PINNING")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    cfg = TopographicConfig(
        vocab_size=256,
        d_model=128,
        n_heads=4,
        n_layers=4,
        ffn_dim=256,
        max_seq_len=128,
        topo_lambda=15.0,
    )

    model = TopographicTransformer(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Total parameters: {num_params:,}")

    init_linears = model.get_linear_projections()
    init_energy = sum(m.dirichlet_energy().item() for m in init_linears) / len(init_linears)

    data = get_synthetic_data(cfg.vocab_size, cfg.max_seq_len, num_samples=500).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("\n--- Starting Training (5 Epochs) ---")
    model.train()
    start_time = time.time()
    for epoch in range(1, 6):
        total_loss = 0.0
        total_topo = 0.0
        batches = data.split(32)

        for batch in batches:
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad()
            logits, ce_loss = model(inputs, targets)
            topo_loss = model.topographic_loss()
            loss = ce_loss + topo_loss

            loss.backward()
            optimizer.step()

            total_loss += ce_loss.item()
            total_topo += topo_loss.item()

        avg_ce = total_loss / len(batches)
        avg_topo = total_topo / len(batches)
        ppl = math.exp(min(avg_ce, 20.0))
        print(f"Epoch {epoch:2d}/5 | CE Loss: {avg_ce:.4f} | Topo Penalty: {avg_topo:.6f} | PPL: {ppl:.2f}")

    elapsed = time.time() - start_time
    print(f"\n[+] Training finished in {elapsed:.2f}s")

    # Evaluate Dirichlet energy vs initialization
    print("\n--- Topological Health Audit ---")
    linears = model.get_linear_projections()
    energies = [m.dirichlet_energy().item() for m in linears]
    mean_energy = sum(energies) / len(energies)
    reduction_pct = (1.0 - mean_energy / max(init_energy, 1e-12)) * 100.0
    print(f"[*] Initial Mean Dirichlet Energy: {init_energy:.6f}")
    print(f"[*] Final Mean Dirichlet Energy:   {mean_energy:.6f} ({reduction_pct:+.1f}% smoothness gain)")
    if mean_energy < init_energy:
        print("[+] Spatial smoothness improved during training.")
    else:
        print("[!] Note: Dirichlet energy did not decrease — consider increasing topo_lambda or epochs.")


if __name__ == "__main__":
    main()
