import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)


class Expert(nn.Module):
    def __init__(self, d_model, d_ff, activation=nn.SiLU):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.activation = activation()

    def forward(self, x):
        return self.w2(self.activation(self.w1(x)))


class Router(nn.Module):
    def __init__(self, d_model, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x)
        full_probs = F.softmax(logits, dim=-1)
        top_k_values, top_k_indices = torch.topk(full_probs, self.top_k, dim=-1)
        top_k_gates = top_k_values / top_k_values.sum(dim=-1, keepdim=True)
        return top_k_gates, top_k_indices, full_probs


class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])
        self.router = Router(d_model, n_experts, top_k)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        gates, indices, full_probs = self.router(x)
        output = torch.zeros_like(x)

        for k in range(self.top_k):
            expert_indices = indices[:, :, k]
            expert_gates = gates[:, :, k]
            for expert_id in range(self.n_experts):
                mask = expert_indices == expert_id
                if mask.any():
                    expert_output = self.experts[expert_id](x[mask])
                    output[mask] += expert_output * expert_gates[mask].unsqueeze(-1)

        routing_info = {
            "gates": gates.detach(),
            "indices": indices.detach(),
            "full_probs": full_probs.detach(),
        }
        return output, routing_info


class DenseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


def compute_load_balancing_loss(gates, indices, full_probs, n_experts, alpha=0.01):
    batch_size, seq_len, top_k = gates.shape
    n_tokens = batch_size * seq_len

    expert_counts = torch.zeros(n_experts)
    for k in range(top_k):
        for expert_id in range(n_experts):
            expert_counts[expert_id] += (indices[:, :, k] == expert_id).float().sum()
    f = expert_counts / (n_tokens * top_k)
    p = full_probs.mean(dim=[0, 1])
    loss = alpha * n_experts * (f * p).sum()

    return loss, {
        "expert_fractions": f,
        "expert_avg_probs": p,
        "load_imbalance": f.max() / (f.min() + 1e-8),
        "loss": loss.item(),
    }


def benchmark_layer(layer, x, n_warmup=10, n_runs=50):
    layer.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            layer(x)
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            layer(x)
            times.append(time.perf_counter() - start)
    return np.mean(times) * 1000, np.std(times) * 1000


def simulate_offloading_latency(n_experts, top_k, expert_size_mb,
                                pcie_bandwidth_gbps=16, gpu_compute_ms=1):
    transfer_ms = top_k * expert_size_mb / (pcie_bandwidth_gbps * 1000) * 1000
    compute_ms = top_k * gpu_compute_ms
    return {
        "transfer_ms": transfer_ms,
        "compute_ms": compute_ms,
        "total_ms": transfer_ms + compute_ms,
        "gpu_memory_mb": top_k * expert_size_mb,
        "total_memory_mb": n_experts * expert_size_mb,
    }


def demo_routing(d_model=256, d_ff=1024, n_experts=8, top_k=2):
    moe = MoELayer(d_model, d_ff, n_experts, top_k)
    x = torch.randn(2, 16, d_model)
    with torch.no_grad():
        output, routing = moe(x)

    total_params = sum(p.numel() for p in moe.parameters())
    active_params = (
        sum(p.numel() for p in moe.experts[0].parameters()) * top_k
        + sum(p.numel() for p in moe.router.parameters())
    )
    print(f"MoE: {n_experts} experts, top-{top_k}")
    print(f"  Total params:  {total_params:,}")
    print(f"  Active params: {active_params:,} per token")
    print(f"  Sparsity:      {1 - active_params / total_params:.1%}")

    loss, stats = compute_load_balancing_loss(
        routing["gates"], routing["indices"], routing["full_probs"], n_experts
    )
    print(f"  Load imbalance: {stats['load_imbalance']:.2f}x")


def demo_specialization(d_model=32, d_ff=64, n_experts=4, n_clusters=4, epochs=200):
    centers = torch.randn(n_clusters, d_model) * 3
    data_x, data_y, labels = [], [], []
    for i, center in enumerate(centers):
        cluster_x = center + torch.randn(100, d_model) * 0.5
        W = torch.randn(d_model, d_model) * 0.3
        data_x.append(cluster_x)
        data_y.append(cluster_x @ W + torch.randn(100, d_model) * 0.1)
        labels.extend([i] * 100)

    X = torch.cat(data_x).unsqueeze(0)
    Y = torch.cat(data_y).unsqueeze(0)
    labels = np.array(labels)

    moe = MoELayer(d_model, d_ff, n_experts, top_k=1)
    optimizer = torch.optim.Adam(moe.parameters(), lr=0.001)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output, routing = moe(X)
        task_loss = F.mse_loss(output, Y)
        bal_loss, _ = compute_load_balancing_loss(
            routing["gates"], routing["indices"], routing["full_probs"], n_experts, alpha=0.1
        )
        (task_loss + bal_loss).backward()
        optimizer.step()

    moe.eval()
    with torch.no_grad():
        _, routing = moe(X)

    assignments = routing["indices"][0, :, 0].numpy()
    confusion = np.zeros((n_clusters, n_experts))
    for cluster, expert in zip(labels, assignments):
        confusion[cluster, expert] += 1
    confusion_norm = confusion / confusion.sum(axis=1, keepdims=True)

    print("\nExpert-Cluster Affinity (after training):")
    header = "         " + "  ".join(f"E{i}" for i in range(n_experts))
    print(header)
    for i, row in enumerate(confusion_norm):
        print(f"Cluster {i}  " + "  ".join(f"{v:.2f}" for v in row))


def demo_benchmark(d_model=512, d_ff=2048, n_experts=8, top_k=2):
    dense_normal = DenseFFN(d_model, d_ff)
    dense_wide = DenseFFN(d_model, d_ff * n_experts)
    moe_layer = MoELayer(d_model, d_ff, n_experts, top_k)

    print("\nBenchmark (seq_len=64):")
    x = torch.randn(1, 64, d_model)
    t_dn, _ = benchmark_layer(dense_normal, x)
    t_dw, _ = benchmark_layer(dense_wide, x)
    t_moe, _ = benchmark_layer(moe_layer, x)
    print(f"  Dense normal: {t_dn:.2f}ms")
    print(f"  Dense wide:   {t_dw:.2f}ms")
    print(f"  MoE:          {t_moe:.2f}ms")


if __name__ == "__main__":
    demo_routing()
    demo_specialization()
    demo_benchmark()
