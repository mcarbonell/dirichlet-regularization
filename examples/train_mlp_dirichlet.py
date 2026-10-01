"""
Beyond-Transformers Demonstration: Dirichlet Regularization on MLP Classifiers
==============================================================================
Demonstrates applying Dirichlet Regularization (`dreg.DirichletLoss`) to standard
Multi-Layer Perceptrons (MLPs).

We compare two identical 3-layer MLPs:
  1. Standard MLP: trained with Cross-Entropy + standard L2 weight decay.
  2. Topographic MLP: trained with Cross-Entropy + DirichletLoss.

We measure:
  - Test classification accuracy
  - Weight matrix Dirichlet harmonic energy
  - Spectral energy concentration in low-frequency DCT harmonics
  - Tolerance to aggressive spectral / ternary quantization

Usage:
  python examples/train_mlp_dirichlet.py --dataset synthetic --epochs 10
  python examples/train_mlp_dirichlet.py --dataset mnist --epochs 5
"""

import argparse
import copy
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dreg
from dreg import DirichletLoss, dct2d, idct2d, Base3TritQuantizer


class SimpleMLP(nn.Module):
    """Standard 3-layer MLP classifier."""
    def __init__(self, in_features: int, hidden_dim: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim, bias=False)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2, bias=False)
        self.act2 = nn.GELU()
        self.fc3 = nn.Linear(hidden_dim // 2, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.fc1(x))
        h = self.act2(self.fc2(h))
        return self.fc3(h)


def get_data(dataset_name: str, batch_size: int):
    """Load MNIST or create a reproducible synthetic classification dataset."""
    if dataset_name.lower() == "mnist":
        try:
            import torchvision
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda t: torch.flatten(t))
            ])
            train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
            test_ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            print("[+] Loaded MNIST dataset (784 features, 10 classes)")
            return train_loader, test_loader, 784, 10
        except Exception as e:
            print(f"[!] Warning: Could not download/load MNIST ({e}). Falling back to synthetic dataset.")

    # Synthetic multi-class clustered data
    torch.manual_seed(42)
    in_features = 256
    num_classes = 10
    n_train = 4000
    n_test = 1000

    # Create distinct class centroids
    centroids = torch.randn(num_classes, in_features) * 2.5

    # Train split
    y_train = torch.randint(0, num_classes, (n_train,))
    x_train = centroids[y_train] + torch.randn(n_train, in_features) * 0.8

    # Test split
    y_test = torch.randint(0, num_classes, (n_test,))
    x_test = centroids[y_test] + torch.randn(n_test, in_features) * 0.8

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"[+] Generated synthetic dataset ({n_train} train, {n_test} test, {in_features} features, {num_classes} classes)")
    return train_loader, test_loader, in_features, num_classes


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute classification accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def analyze_spectral_energy(model: nn.Module) -> dict:
    """
    Compute total Dirichlet energy and low-frequency spectral energy percentage.
    Low-frequency band is defined as the top-left quadrant of the 2D-DCT spectrum.
    """
    total_dirichlet = 0.0
    total_frobenius = 0.0
    low_freq_energy = 0.0
    num_mats = 0

    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, nn.Linear):
                w = mod.weight.detach()
                e_d = dreg.dirichlet_energy_2d(w).item()
                total_dirichlet += e_d

                # 2D-DCT transform
                s = dct2d(w)
                m, n = s.shape
                frob = torch.sum(s ** 2).item()
                total_frobenius += frob

                # Energy in low-frequency quadrant (r <= 0.40 radius)
                u = torch.linspace(0, 1, m, device=w.device)
                v = torch.linspace(0, 1, n, device=w.device)
                grid_u, grid_v = torch.meshgrid(u, v, indexing="ij")
                rho = torch.sqrt(grid_u ** 2 + grid_v ** 2)
                low_mask = rho <= 0.40
                low_frob = torch.sum((s * low_mask.float()) ** 2).item()
                low_freq_energy += low_frob
                num_mats += 1

    ratio = (low_freq_energy / max(total_frobenius, 1e-12)) * 100.0
    return {
        "dirichlet_energy": total_dirichlet / max(num_mats, 1),
        "low_freq_pct": ratio,
    }


def quantize_and_evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    Evaluate model under Base-3 Trit spectral quantization.
    Replaces each Linear layer's weight with its quantized & reconstructed version.
    """
    q_model = copy.deepcopy(model)
    quantizer = Base3TritQuantizer(r0=0.15, r1=0.40, r2=1.0)

    with torch.no_grad():
        for mod in q_model.modules():
            if isinstance(mod, nn.Linear):
                w = mod.weight.detach()
                packed = quantizer.quantize_matrix(w)
                rec_spec = quantizer.dequantize_matrix(packed, device=device)
                w_rec = idct2d(rec_spec)
                mod.weight.copy_(w_rec)

    return evaluate(q_model, loader, device)


def spectral_truncate_and_evaluate(model: nn.Module, loader: DataLoader, device: torch.device, keep_radius: float = 0.40) -> float:
    """
    Evaluate model retaining only low-frequency 2D-DCT coefficients (r <= keep_radius).
    All high-frequency components are zeroed out (low-pass filtering the weights).
    """
    trunc_model = copy.deepcopy(model)
    with torch.no_grad():
        for mod in trunc_model.modules():
            if isinstance(mod, nn.Linear):
                w = mod.weight.detach()
                s = dct2d(w)
                m, n = s.shape
                u = torch.linspace(0, 1, m, device=w.device)
                v = torch.linspace(0, 1, n, device=w.device)
                grid_u, grid_v = torch.meshgrid(u, v, indexing="ij")
                rho = torch.sqrt(grid_u ** 2 + grid_v ** 2)
                low_mask = (rho <= keep_radius).float()
                w_rec = idct2d(s * low_mask)
                mod.weight.copy_(w_rec)
    return evaluate(trunc_model, loader, device)


def train_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    use_dirichlet: bool,
    dirichlet_lambda: float,
):
    """Train an MLP with or without Dirichlet regularization."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    dirichlet_loss_fn = DirichletLoss(weight_decay=dirichlet_lambda) if use_dirichlet else None

    print(f"\n--- Training {name} ({'with Dirichlet reg (lambda=' + str(dirichlet_lambda) + ')' if use_dirichlet else 'Standard L2 only'}) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        running_task_loss = 0.0
        running_topo_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            task_loss = criterion(out, y)

            if use_dirichlet:
                topo_loss = dirichlet_loss_fn(model.modules())
                total_loss = task_loss + topo_loss
                running_topo_loss += topo_loss.item()
            else:
                total_loss = task_loss

            total_loss.backward()
            optimizer.step()
            running_task_loss += task_loss.item()

        n_batches = len(train_loader)
        avg_task = running_task_loss / n_batches
        avg_topo = running_topo_loss / n_batches if use_dirichlet else 0.0
        test_acc = evaluate(model, test_loader, device)

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            topo_str = f" | Topo Loss: {avg_topo:.4f}" if use_dirichlet else ""
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Task Loss: {avg_task:.4f}{topo_str} | Test Acc: {test_acc * 100:.2f}%")

    return model


def main():
    parser = argparse.ArgumentParser(description="Beyond-Transformers: Dirichlet Regularization on MLP Classifiers")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "mnist"], help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension of MLP")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda-dirichlet", type=float, default=0.02, help="Dirichlet regularization weight")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")

    # Data
    train_loader, test_loader, in_features, num_classes = get_data(args.dataset, args.batch_size)

    # 1. Standard Model
    torch.manual_seed(1337)
    std_model = SimpleMLP(in_features, args.hidden_dim, num_classes)
    std_model = train_model(
        "Standard MLP",
        std_model,
        train_loader,
        test_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        use_dirichlet=False,
        dirichlet_lambda=0.0,
    )

    # 2. Topographic Model (same init seed)
    torch.manual_seed(1337)
    topo_model = SimpleMLP(in_features, args.hidden_dim, num_classes)
    topo_model = train_model(
        "Topographic MLP",
        topo_model,
        train_loader,
        test_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        use_dirichlet=True,
        dirichlet_lambda=args.lambda_dirichlet,
    )

    # Analysis
    print("\n" + "=" * 78)
    print(" BEYOND-TRANSFORMERS EVALUATION: STANDARD VS TOPOGRAPHIC MLP")
    print("=" * 78)

    acc_std = evaluate(std_model, test_loader, device) * 100.0
    acc_topo = evaluate(topo_model, test_loader, device) * 100.0

    stats_std = analyze_spectral_energy(std_model)
    stats_topo = analyze_spectral_energy(topo_model)

    print("[*] Evaluating sub-1.0 bpp Base-3 Trit Quantization tolerance...")
    q_acc_std = quantize_and_evaluate(std_model, test_loader, device) * 100.0
    q_acc_topo = quantize_and_evaluate(topo_model, test_loader, device) * 100.0

    drop_std = acc_std - q_acc_std
    drop_topo = acc_topo - q_acc_topo
    print("[*] Evaluating low-pass spectral truncation (r <= 0.40, ~20% lowest DCT components)...")
    trunc_acc_std = spectral_truncate_and_evaluate(std_model, test_loader, device, keep_radius=0.40) * 100.0
    trunc_acc_topo = spectral_truncate_and_evaluate(topo_model, test_loader, device, keep_radius=0.40) * 100.0
    trunc_drop_std = acc_std - trunc_acc_std
    trunc_drop_topo = acc_topo - trunc_acc_topo

    print("\n" + "-" * 78)
    print(f"{'Metric':<30} | {'Standard MLP':<20} | {'Topographic MLP':<20}")
    print("-" * 78)
    print(f"{'FP32 Test Accuracy':<30} | {acc_std:6.2f}%{'':<13} | {acc_topo:6.2f}%{'':<13}")
    print(f"{'Dirichlet Energy (E_D)':<30} | {stats_std['dirichlet_energy']:6.4f}{'':<14} | {stats_topo['dirichlet_energy']:6.4f}{'':<14}")
    print(f"{'Low-Freq Energy (r <= 0.40)':<30} | {stats_std['low_freq_pct']:6.2f}%{'':<13} | {stats_topo['low_freq_pct']:6.2f}%{'':<13}")
    print(f"{'Low-Pass Acc (r <= 0.40)':<30} | {trunc_acc_std:6.2f}%{'':<13} | {trunc_acc_topo:6.2f}%{'':<13}")
    print(f"{'Low-Pass Drop (Delta)':<30} | -{trunc_drop_std:5.2f}%{'':<13} | -{trunc_drop_topo:5.2f}%{'':<13}")
    print(f"{'Quantized Accuracy (0.94 bpp)':<30} | {q_acc_std:6.2f}%{'':<13} | {q_acc_topo:6.2f}%{'':<13}")
    print(f"{'Quantization Drop (Delta)':<30} | -{drop_std:5.2f}%{'':<13} | -{drop_topo:5.2f}%{'':<13}")
    print("-" * 78)

    if trunc_drop_topo < trunc_drop_std:
        gain = trunc_drop_std - trunc_drop_topo
        print(f"[*] Result: Topographic MLP preserved +{gain:.2f}% higher accuracy under aggressive low-pass truncation.")
    print("=" * 78)


if __name__ == "__main__":
    main()
