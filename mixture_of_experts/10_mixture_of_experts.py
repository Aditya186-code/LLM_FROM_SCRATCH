!pip install torch matplotlib numpy -q

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import time

torch.manual_seed(42)
np.random.seed(42)

plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

print("Setup complete!")

# Visualize Dense vs MoE architecture
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Dense model
ax = ax1
# Input tokens
tokens = ['Token 1', 'Token 2', 'Token 3', 'Token 4']
for i, tok in enumerate(tokens):
    ax.add_patch(plt.Rectangle((i * 2, 5), 1.5, 0.8, facecolor='#3498db', edgecolor='black'))
    ax.text(i * 2 + 0.75, 5.4, tok, ha='center', va='center', fontsize=8, fontweight='bold', color='white')

# Single FFN block
ax.add_patch(plt.Rectangle((1, 2.5), 5.5, 1.5, facecolor='#e74c3c', edgecolor='black', linewidth=2))
ax.text(3.75, 3.25, 'Dense FFN\n(ALL parameters)', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Arrows: all tokens go through the one FFN
for i in range(4):
    ax.annotate('', xy=(3.75, 4.0), xytext=(i * 2 + 0.75, 5.0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# Output
for i in range(4):
    ax.add_patch(plt.Rectangle((i * 2, 0.5), 1.5, 0.8, facecolor='#2ecc71', edgecolor='black'))
    ax.text(i * 2 + 0.75, 0.9, f'Out {i+1}', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.annotate('', xy=(i * 2 + 0.75, 1.3), xytext=(3.75, 2.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

ax.set_xlim(-0.5, 8.5)
ax.set_ylim(0, 6.5)
ax.set_title('Dense Model\n(All tokens use all parameters)', fontsize=13, fontweight='bold')
ax.axis('off')

# MoE model
ax = ax2
# Input tokens
for i, tok in enumerate(tokens):
    ax.add_patch(plt.Rectangle((i * 2.5, 6), 1.8, 0.8, facecolor='#3498db', edgecolor='black'))
    ax.text(i * 2.5 + 0.9, 6.4, tok, ha='center', va='center', fontsize=8, fontweight='bold', color='white')

# Router
ax.add_patch(plt.Rectangle((2, 4.5), 5.5, 0.8, facecolor='#f39c12', edgecolor='black', linewidth=2))
ax.text(4.75, 4.9, 'Router (Gating Network)', ha='center', va='center', fontsize=10, fontweight='bold')

# Experts
expert_colors = ['#e74c3c', '#9b59b6', '#1abc9c', '#34495e']
for i in range(4):
    ax.add_patch(plt.Rectangle((i * 2.5, 2), 1.8, 1.2, facecolor=expert_colors[i], edgecolor='black', linewidth=1.5, alpha=0.8))
    ax.text(i * 2.5 + 0.9, 2.6, f'Expert {i+1}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Routing arrows (each token goes to 2 experts)
routing = [(0, [0, 2]), (1, [1, 3]), (2, [0, 1]), (3, [2, 3])]
arrow_colors = ['#3498db', '#e67e22', '#27ae60', '#8e44ad']
for tok_i, (_, experts) in enumerate(routing):
    for exp_i in experts:
        ax.annotate('', xy=(exp_i * 2.5 + 0.9, 3.2), xytext=(tok_i * 2.5 + 0.9, 4.5),
                    arrowprops=dict(arrowstyle='->', color=arrow_colors[tok_i], lw=2, alpha=0.7))
    # Token to router
    ax.annotate('', xy=(4.75, 5.3), xytext=(tok_i * 2.5 + 0.9, 6.0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))

# Output
for i in range(4):
    ax.add_patch(plt.Rectangle((i * 2.5, 0.3), 1.8, 0.8, facecolor='#2ecc71', edgecolor='black'))
    ax.text(i * 2.5 + 0.9, 0.7, f'Out {i+1}', ha='center', va='center', fontsize=8, fontweight='bold')

ax.set_xlim(-0.5, 11)
ax.set_ylim(-0.2, 7.5)
ax.set_title('MoE Model\n(Each token routed to top-2 experts)', fontsize=13, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.show()

class Expert(nn.Module):
    """A single expert FFN.
    
    Standard Transformer FFN: Linear -> Activation -> Linear
    This is identical to the FFN in a dense model.
    """
    def __init__(self, d_model, d_ff, activation=nn.SiLU):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.activation = activation()
    
    def forward(self, x):
        return self.w2(self.activation(self.w1(x)))

# Test a single expert
d_model = 256
d_ff = 1024  # Typical: 4x d_model

expert = Expert(d_model, d_ff)
x = torch.randn(1, 10, d_model)  # batch=1, seq_len=10
out = expert(x)

expert_params = sum(p.numel() for p in expert.parameters())
print(f"Expert architecture: {d_model} -> {d_ff} -> {d_model}")
print(f"Parameters per expert: {expert_params:,}")
print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}")

class Router(nn.Module):
    """Token-level router for Mixture of Experts.
    
    For each token, produces a probability distribution over experts
    and selects the top-k experts to activate.
    """
    def __init__(self, d_model, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        
        # Gating weight matrix: maps token embedding to expert scores
        self.gate = nn.Linear(d_model, n_experts, bias=False)
    
    def forward(self, x):
        """Route tokens to experts.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            top_k_gates: Normalized weights for selected experts
                         shape: (batch, seq_len, top_k)
            top_k_indices: Which experts were selected
                          shape: (batch, seq_len, top_k)
            full_probs: Full probability distribution over all experts
                       shape: (batch, seq_len, n_experts)
        """
        # Compute gating scores
        logits = self.gate(x)  # (batch, seq_len, n_experts)
        
        # Full probability distribution
        full_probs = F.softmax(logits, dim=-1)
        
        # Select top-k experts
        top_k_values, top_k_indices = torch.topk(full_probs, self.top_k, dim=-1)
        
        # Renormalize the top-k probabilities to sum to 1
        top_k_gates = top_k_values / top_k_values.sum(dim=-1, keepdim=True)
        
        return top_k_gates, top_k_indices, full_probs

# Test the router
n_experts = 8
top_k = 2

router = Router(d_model, n_experts, top_k)
gates, indices, probs = router(x)

print(f"Input shape: {x.shape}")
print(f"Gate weights shape: {gates.shape}")
print(f"Expert indices shape: {indices.shape}")
print(f"Full probs shape: {probs.shape}")

print(f"\nSample routing for first 5 tokens:")
print(f"{'Token':>6s} | {'Expert 1 (weight)':>18s} | {'Expert 2 (weight)':>18s}")
print("-" * 55)
for t in range(5):
    e1, e2 = indices[0, t].tolist()
    g1, g2 = gates[0, t].tolist()
    print(f"{t:>6d} | Expert {e1} ({g1:.3f})       | Expert {e2} ({g2:.3f})")

class MoELayer(nn.Module):
    """Mixture of Experts layer.
    
    Replaces the standard FFN in a Transformer.
    Routes each token to top-k experts and combines their outputs.
    """
    def __init__(self, d_model, d_ff, n_experts, top_k=2):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        
        # Create n_experts independent FFNs
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(n_experts)
        ])
        
        # Router
        self.router = Router(d_model, n_experts, top_k)
    
    def forward(self, x):
        """Forward pass through MoE layer.
        
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
            routing_info: dict with routing statistics
        """
        batch_size, seq_len, d_model = x.shape
        
        # Get routing decisions
        gates, indices, full_probs = self.router(x)
        # gates: (batch, seq_len, top_k)
        # indices: (batch, seq_len, top_k)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process each expert
        # (In production, this would be batched for efficiency)
        for k in range(self.top_k):
            expert_indices = indices[:, :, k]  # (batch, seq_len)
            expert_gates = gates[:, :, k]       # (batch, seq_len)
            
            for expert_id in range(self.n_experts):
                # Find which tokens are routed to this expert
                mask = (expert_indices == expert_id)  # (batch, seq_len)
                
                if mask.any():
                    # Get the tokens for this expert
                    expert_input = x[mask]  # (n_tokens, d_model)
                    
                    # Process through expert
                    expert_output = self.experts[expert_id](expert_input)
                    
                    # Weight by gate value and add to output
                    gate_values = expert_gates[mask].unsqueeze(-1)
                    output[mask] += expert_output * gate_values
        
        # Collect routing info for analysis
        routing_info = {
            'gates': gates.detach(),
            'indices': indices.detach(),
            'full_probs': full_probs.detach(),
        }
        
        return output, routing_info

# Test the MoE layer
n_experts = 8
top_k = 2
d_ff = 1024

moe = MoELayer(d_model, d_ff, n_experts, top_k)
x = torch.randn(2, 16, d_model)  # batch=2, seq_len=16

with torch.no_grad():
    output, routing_info = moe(x)

total_params = sum(p.numel() for p in moe.parameters())
active_params = sum(p.numel() for p in moe.experts[0].parameters()) * top_k + \
                sum(p.numel() for p in moe.router.parameters())

print(f"MoE Layer: {n_experts} experts, top-{top_k} routing")
print(f"Total parameters:  {total_params:>12,}")
print(f"Active parameters: {active_params:>12,} (per token)")
print(f"Sparsity ratio:    {1 - active_params/total_params:.1%}")
print(f"\nInput:  {x.shape}")
print(f"Output: {output.shape}")

# Create diverse input tokens (simulating different token types)
torch.manual_seed(42)
n_tokens = 32

# Create 4 clusters of tokens (simulating different content types)
cluster_centers = torch.randn(4, d_model) * 3
tokens_per_cluster = n_tokens // 4

diverse_tokens = []
token_labels = []
for i, center in enumerate(cluster_centers):
    cluster_tokens = center.unsqueeze(0) + torch.randn(tokens_per_cluster, d_model) * 0.5
    diverse_tokens.append(cluster_tokens)
    token_labels.extend([f'Type {i+1}'] * tokens_per_cluster)

diverse_input = torch.cat(diverse_tokens, dim=0).unsqueeze(0)  # (1, 32, d_model)

with torch.no_grad():
    _, routing = moe(diverse_input)

# Visualize routing decisions
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: Full routing probability heatmap
ax = axes[0]
probs = routing['full_probs'][0].numpy()  # (32, 8)
im = ax.imshow(probs, aspect='auto', cmap='YlOrRd')
ax.set_xlabel('Expert ID')
ax.set_ylabel('Token Index')
ax.set_title('Router Probability Distribution\n(brighter = higher probability)')
ax.set_xticks(range(n_experts))
ax.set_xticklabels([f'E{i}' for i in range(n_experts)])
plt.colorbar(im, ax=ax)

# Add cluster boundaries
for boundary in range(1, 4):
    ax.axhline(y=boundary * tokens_per_cluster - 0.5, color='white', linewidth=2, linestyle='--')

# Add cluster labels
for i in range(4):
    ax.text(-1.5, i * tokens_per_cluster + tokens_per_cluster/2,
            f'Type {i+1}', ha='center', va='center', fontsize=9, fontweight='bold')

# Plot 2: Which experts were selected (top-2)
ax = axes[1]
selected = np.zeros((n_tokens, n_experts))
for t in range(n_tokens):
    for k in range(top_k):
        expert_id = routing['indices'][0, t, k].item()
        gate_val = routing['gates'][0, t, k].item()
        selected[t, expert_id] = gate_val

im = ax.imshow(selected, aspect='auto', cmap='Blues')
ax.set_xlabel('Expert ID')
ax.set_ylabel('Token Index')
ax.set_title(f'Top-{top_k} Expert Selection\n(intensity = gate weight)')
ax.set_xticks(range(n_experts))
ax.set_xticklabels([f'E{i}' for i in range(n_experts)])
plt.colorbar(im, ax=ax)

for boundary in range(1, 4):
    ax.axhline(y=boundary * tokens_per_cluster - 0.5, color='red', linewidth=2, linestyle='--')

# Plot 3: Expert load (how many tokens each expert processes)
ax = axes[2]
expert_loads = np.zeros(n_experts)
for k in range(top_k):
    for expert_id in range(n_experts):
        expert_loads[expert_id] += (routing['indices'][0, :, k] == expert_id).sum().item()

colors = plt.cm.Set3(np.linspace(0, 1, n_experts))
bars = ax.bar(range(n_experts), expert_loads, color=colors, edgecolor='black')
ax.axhline(y=n_tokens * top_k / n_experts, color='red', linestyle='--',
           label=f'Perfect balance ({n_tokens * top_k / n_experts:.0f})')
ax.set_xlabel('Expert ID')
ax.set_ylabel('Number of Tokens')
ax.set_title('Expert Load Distribution')
ax.set_xticks(range(n_experts))
ax.legend()

for bar, load in zip(bars, expert_loads):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
            f'{int(load)}', ha='center', fontsize=10, fontweight='bold')

plt.suptitle('Router Behavior Analysis', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

def compute_load_balancing_loss(gates, indices, full_probs, n_experts, alpha=0.01):
    """Compute the load balancing auxiliary loss.
    
    This encourages the router to distribute tokens evenly across experts.
    
    Args:
        gates: Top-k gate weights (batch, seq_len, top_k)
        indices: Top-k expert indices (batch, seq_len, top_k)
        full_probs: Full probability distribution (batch, seq_len, n_experts)
        n_experts: Number of experts
        alpha: Loss coefficient
    
    Returns:
        loss: Load balancing loss (scalar)
        load_stats: Dictionary with load statistics
    """
    batch_size, seq_len, top_k = gates.shape
    n_tokens = batch_size * seq_len
    
    # f_i: fraction of tokens dispatched to expert i
    # Count how many times each expert is selected
    expert_counts = torch.zeros(n_experts)
    for k in range(top_k):
        for expert_id in range(n_experts):
            expert_counts[expert_id] += (indices[:, :, k] == expert_id).float().sum()
    f = expert_counts / (n_tokens * top_k)  # Normalize
    
    # p_i: average routing probability for expert i
    p = full_probs.mean(dim=[0, 1])  # (n_experts,)
    
    # Load balancing loss
    loss = alpha * n_experts * (f * p).sum()
    
    # Statistics
    load_stats = {
        'expert_fractions': f,
        'expert_avg_probs': p,
        'load_imbalance': f.max() / (f.min() + 1e-8),
        'loss': loss.item(),
    }
    
    return loss, load_stats

# Compute for our example
loss, stats = compute_load_balancing_loss(
    routing['gates'], routing['indices'], routing['full_probs'], n_experts
)

print("Load Balancing Analysis:")
print(f"  Loss: {stats['loss']:.4f}")
print(f"  Load imbalance ratio: {stats['load_imbalance']:.2f}x")
print(f"  (1.0 = perfect balance)\n")

print(f"  {'Expert':>8s} | {'Token Fraction':>15s} | {'Avg Probability':>15s}")
print("  " + "-" * 50)
for i in range(n_experts):
    print(f"  Expert {i:>2d} | {stats['expert_fractions'][i]:>14.3f} | {stats['expert_avg_probs'][i]:>14.4f}")

# Visualize balanced vs imbalanced routing
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Scenario 1: Perfectly balanced
balanced_f = torch.ones(n_experts) / n_experts
axes[0].bar(range(n_experts), balanced_f.numpy(), color='green', alpha=0.7)
axes[0].axhline(y=1/n_experts, color='red', linestyle='--')
axes[0].set_title('Perfectly Balanced\n(Ideal)', fontweight='bold')
axes[0].set_xlabel('Expert ID')
axes[0].set_ylabel('Token Fraction')
axes[0].set_ylim(0, 0.5)

# Scenario 2: Current routing
axes[1].bar(range(n_experts), stats['expert_fractions'].numpy(),
            color='orange', alpha=0.7)
axes[1].axhline(y=1/n_experts, color='red', linestyle='--')
axes[1].set_title(f'Current Routing\n(Imbalance: {stats["load_imbalance"]:.1f}x)', fontweight='bold')
axes[1].set_xlabel('Expert ID')
axes[1].set_ylim(0, 0.5)

# Scenario 3: Heavily imbalanced (worst case)
imbalanced_f = torch.zeros(n_experts)
imbalanced_f[0] = 0.6
imbalanced_f[1] = 0.3
imbalanced_f[2] = 0.1
axes[2].bar(range(n_experts), imbalanced_f.numpy(), color='red', alpha=0.7)
axes[2].axhline(y=1/n_experts, color='red', linestyle='--')
axes[2].set_title('Heavily Imbalanced\n(Worst case - wastes experts)', fontweight='bold')
axes[2].set_xlabel('Expert ID')
axes[2].set_ylim(0, 0.7)

plt.suptitle('Expert Load Distribution Scenarios', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

class DenseFFN(nn.Module):
    """Standard dense FFN for comparison."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

# Compare dense vs MoE
d_model = 512
d_ff = 2048  # per expert
n_experts = 8
top_k = 2

# Dense model with equivalent capacity (8x wider FFN)
dense_wide = DenseFFN(d_model, d_ff * n_experts)  # Equivalent total params
dense_normal = DenseFFN(d_model, d_ff)  # Same per-token compute as MoE
moe_layer = MoELayer(d_model, d_ff, n_experts, top_k)

dense_wide_params = sum(p.numel() for p in dense_wide.parameters())
dense_normal_params = sum(p.numel() for p in dense_normal.parameters())
moe_params = sum(p.numel() for p in moe_layer.parameters())
moe_active = sum(p.numel() for p in moe_layer.experts[0].parameters()) * top_k + \
             sum(p.numel() for p in moe_layer.router.parameters())

print("Architecture Comparison:")
print("=" * 60)
print(f"{'':>25s} | {'Total Params':>12s} | {'Active Params':>13s}")
print("-" * 60)
print(f"{'Dense (normal FFN)':>25s} | {dense_normal_params:>12,} | {dense_normal_params:>13,}")
print(f"{'Dense (wide FFN)':>25s} | {dense_wide_params:>12,} | {dense_wide_params:>13,}")
print(f"{'MoE (8 experts, top-2)':>25s} | {moe_params:>12,} | {moe_active:>13,}")
print(f"\nMoE has {moe_params/dense_normal_params:.1f}x more total params than normal dense")
print(f"But only {moe_active/dense_normal_params:.1f}x the active params per token")
print(f"MoE has {moe_params/dense_wide_params:.1%} of wide-dense params but similar capacity")

# Benchmark wall-clock time
def benchmark_layer(layer, x, n_warmup=10, n_runs=50, is_moe=False):
    """Benchmark a layer's forward pass."""
    layer.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            if is_moe:
                layer(x)
            else:
                layer(x)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            if is_moe:
                layer(x)
            else:
                layer(x)
            times.append(time.perf_counter() - start)
    
    return np.mean(times) * 1000, np.std(times) * 1000  # ms

# Benchmark with different sequence lengths
seq_lengths = [16, 32, 64, 128, 256, 512]

results_dense_normal = []
results_dense_wide = []
results_moe = []

print("Benchmarking (this takes a moment)...")
for seq_len in seq_lengths:
    x = torch.randn(1, seq_len, d_model)
    
    t_dn, _ = benchmark_layer(dense_normal, x)
    t_dw, _ = benchmark_layer(dense_wide, x)
    t_moe, _ = benchmark_layer(moe_layer, x, is_moe=True)
    
    results_dense_normal.append(t_dn)
    results_dense_wide.append(t_dw)
    results_moe.append(t_moe)
    
    print(f"  seq_len={seq_len:>4d}: Dense={t_dn:.2f}ms, Dense-Wide={t_dw:.2f}ms, MoE={t_moe:.2f}ms")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(seq_lengths, results_dense_normal, 'b-o', label=f'Dense Normal ({dense_normal_params:,} params)', linewidth=2)
ax1.plot(seq_lengths, results_dense_wide, 'r-o', label=f'Dense Wide ({dense_wide_params:,} params)', linewidth=2)
ax1.plot(seq_lengths, results_moe, 'g-o', label=f'MoE ({moe_params:,} params, {moe_active:,} active)', linewidth=2)
ax1.set_xlabel('Sequence Length')
ax1.set_ylabel('Time (ms)')
ax1.set_title('Forward Pass Time')
ax1.legend(fontsize=9)

# Compute throughput (tokens per second)
throughput_dn = [seq / (t/1000) for seq, t in zip(seq_lengths, results_dense_normal)]
throughput_dw = [seq / (t/1000) for seq, t in zip(seq_lengths, results_dense_wide)]
throughput_moe = [seq / (t/1000) for seq, t in zip(seq_lengths, results_moe)]

ax2.plot(seq_lengths, throughput_dn, 'b-o', label='Dense Normal', linewidth=2)
ax2.plot(seq_lengths, throughput_dw, 'r-o', label='Dense Wide', linewidth=2)
ax2.plot(seq_lengths, throughput_moe, 'g-o', label='MoE', linewidth=2)
ax2.set_xlabel('Sequence Length')
ax2.set_ylabel('Tokens/Second')
ax2.set_title('Throughput Comparison')
ax2.legend()

plt.tight_layout()
plt.show()

# Simulate expert specialization by training a small MoE
# on synthetic data with clear clusters

torch.manual_seed(42)
d_model = 32
d_ff = 64
n_experts = 4
top_k = 1  # Use top-1 for clearer specialization
n_clusters = 4
samples_per_cluster = 100

# Create clustered data
centers = torch.randn(n_clusters, d_model) * 3
data_x = []
data_y = []
labels = []

for i, center in enumerate(centers):
    cluster_x = center + torch.randn(samples_per_cluster, d_model) * 0.5
    # Simple target: different transformation per cluster
    W_target = torch.randn(d_model, d_model) * 0.3
    cluster_y = cluster_x @ W_target + torch.randn(samples_per_cluster, d_model) * 0.1
    data_x.append(cluster_x)
    data_y.append(cluster_y)
    labels.extend([i] * samples_per_cluster)

X = torch.cat(data_x, dim=0).unsqueeze(0)  # (1, 400, d_model)
Y = torch.cat(data_y, dim=0).unsqueeze(0)
labels = np.array(labels)

# Create and train a small MoE
moe_small = MoELayer(d_model, d_ff, n_experts, top_k=1)
optimizer = torch.optim.Adam(moe_small.parameters(), lr=0.001)

# Training loop
losses = []
balance_losses = []

for epoch in range(200):
    optimizer.zero_grad()
    
    output, routing = moe_small(X)
    
    # Task loss
    task_loss = F.mse_loss(output, Y)
    
    # Load balancing loss
    bal_loss, _ = compute_load_balancing_loss(
        routing['gates'], routing['indices'], routing['full_probs'], n_experts, alpha=0.1
    )
    
    total_loss = task_loss + bal_loss
    total_loss.backward()
    optimizer.step()
    
    losses.append(task_loss.item())
    balance_losses.append(bal_loss.item())

# Analyze routing after training
moe_small.eval()
with torch.no_grad():
    _, routing_trained = moe_small(X)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training loss
ax = axes[0][0]
ax.plot(losses, label='Task Loss', linewidth=2)
ax.plot(balance_losses, label='Balance Loss', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Losses')
ax.legend()

# Plot 2: Expert assignment vs data cluster
ax = axes[0][1]
expert_assignments = routing_trained['indices'][0, :, 0].numpy()  # top-1 expert

# Cross-tabulation
confusion = np.zeros((n_clusters, n_experts))
for cluster, expert in zip(labels, expert_assignments):
    confusion[cluster, expert] += 1

# Normalize per cluster
confusion_norm = confusion / confusion.sum(axis=1, keepdims=True)

im = ax.imshow(confusion_norm, cmap='YlOrRd', vmin=0, vmax=1)
ax.set_xlabel('Expert ID')
ax.set_ylabel('Data Cluster')
ax.set_title('Expert-Cluster Affinity\n(darker = stronger match)')
ax.set_xticks(range(n_experts))
ax.set_yticks(range(n_clusters))
ax.set_xticklabels([f'Expert {i}' for i in range(n_experts)])
ax.set_yticklabels([f'Cluster {i}' for i in range(n_clusters)])
plt.colorbar(im, ax=ax)

# Add values
for i in range(n_clusters):
    for j in range(n_experts):
        ax.text(j, i, f'{confusion_norm[i, j]:.0%}', ha='center', va='center',
                fontsize=10, color='white' if confusion_norm[i, j] > 0.5 else 'black')

# Plot 3: Expert load over training
ax = axes[1][0]
ax.bar(range(n_experts), [np.sum(expert_assignments == i) for i in range(n_experts)],
       color=plt.cm.Set3(np.linspace(0, 1, n_experts)), edgecolor='black')
ax.axhline(y=len(labels) / n_experts, color='red', linestyle='--', label='Perfect balance')
ax.set_xlabel('Expert ID')
ax.set_ylabel('Number of Tokens')
ax.set_title('Final Expert Load Distribution')
ax.legend()

# Plot 4: 2D visualization of routing
ax = axes[1][1]
# Use PCA for visualization
from torch.linalg import svd
X_flat = X[0].detach()
X_centered = X_flat - X_flat.mean(dim=0)
U, S, Vh = svd(X_centered, full_matrices=False)
X_2d = (X_centered @ Vh[:2].T).numpy()

scatter_colors = plt.cm.Set1(np.linspace(0, 0.5, n_experts))
for expert_id in range(n_experts):
    mask = expert_assignments == expert_id
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[scatter_colors[expert_id]], 
               label=f'Expert {expert_id}', alpha=0.5, s=20)

ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_title('Token Routing in 2D (PCA projection)')
ax.legend(fontsize=8)

plt.suptitle('Expert Specialization After Training', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# Visualize the MoE memory-compute tradeoff
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Model comparison
models = {
    'LLaMA-7B\n(Dense)':    {'total_params': 7, 'active_params': 7, 'type': 'dense'},
    'LLaMA-13B\n(Dense)':   {'total_params': 13, 'active_params': 13, 'type': 'dense'},
    'Mixtral-8x7B\n(MoE)':  {'total_params': 47, 'active_params': 13, 'type': 'moe'},
    'LLaMA-70B\n(Dense)':   {'total_params': 70, 'active_params': 70, 'type': 'dense'},
    'Hypothetical\nMoE-8x70B': {'total_params': 280, 'active_params': 70, 'type': 'moe'},
}

names = list(models.keys())
total = [m['total_params'] for m in models.values()]
active = [m['active_params'] for m in models.values()]
bar_colors = ['#1f77b4' if m['type'] == 'dense' else '#ff7f0e' for m in models.values()]

x_pos = np.arange(len(names))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, total, width, label='Total Parameters', color=bar_colors, alpha=0.4)
bars2 = ax1.bar(x_pos + width/2, active, width, label='Active per Token', color=bar_colors, alpha=0.9)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(names, fontsize=9)
ax1.set_ylabel('Parameters (Billions)')
ax1.set_title('Total vs Active Parameters')
ax1.legend()

# Add annotations
for i, (t, a) in enumerate(zip(total, active)):
    if t != a:
        ax1.annotate(f'{a/t:.0%} active', xy=(i, a), fontsize=8,
                     ha='center', va='bottom', color='red', fontweight='bold')

# Memory vs compute comparison
ax2.scatter([m['total_params'] for m in models.values()],
            [m['active_params'] for m in models.values()],
            c=bar_colors, s=200, zorder=5, edgecolors='black')

for name, m in zip(names, models.values()):
    ax2.annotate(name.replace('\n', ' '), (m['total_params'], m['active_params']),
                 textcoords='offset points', xytext=(5, 5), fontsize=8)

# Perfect efficiency line (dense models)
ax2.plot([0, 300], [0, 300], 'k--', alpha=0.3, label='Dense (active=total)')
ax2.set_xlabel('Total Parameters (B) = Memory Required')
ax2.set_ylabel('Active Parameters (B) = Compute Required')
ax2.set_title('Memory vs Compute: Dense vs MoE')
ax2.legend()

legend_patches = [
    mpatches.Patch(color='#1f77b4', label='Dense'),
    mpatches.Patch(color='#ff7f0e', label='MoE'),
]
ax2.legend(handles=legend_patches + [plt.Line2D([0], [0], color='k', linestyle='--', alpha=0.3, label='Dense (active=total)')],
           fontsize=9)

plt.tight_layout()
plt.show()

# Simulate expert offloading
def simulate_offloading_latency(n_experts, top_k, expert_size_mb, 
                                 pcie_bandwidth_gbps=16, gpu_compute_ms=1):
    """Simulate the latency of MoE with expert offloading.
    
    Args:
        n_experts: Total number of experts
        top_k: Number of active experts per token
        expert_size_mb: Size of each expert in MB
        pcie_bandwidth_gbps: PCIe transfer bandwidth in GB/s
        gpu_compute_ms: Compute time per expert on GPU in ms
    """
    # Transfer time for top-k experts
    transfer_time_ms = top_k * expert_size_mb / (pcie_bandwidth_gbps * 1000) * 1000
    
    # Compute time
    compute_time_ms = top_k * gpu_compute_ms
    
    return {
        'transfer_ms': transfer_time_ms,
        'compute_ms': compute_time_ms,
        'total_ms': transfer_time_ms + compute_time_ms,
        'gpu_memory_mb': top_k * expert_size_mb,  # Only active experts in VRAM
        'total_memory_mb': n_experts * expert_size_mb,  # All experts in RAM
    }

# Compare scenarios for Mixtral-8x7B-like model
expert_size = 800  # ~800MB per expert at FP16

scenarios = [
    ('All on GPU', 8, 2, expert_size, 32, 1, False),
    ('Offload (PCIe 4.0)', 8, 2, expert_size, 16, 1, True),
    ('Offload (PCIe 3.0)', 8, 2, expert_size, 8, 1, True),
]

print("MoE Serving Scenarios (Mixtral-8x7B-like):")
print("=" * 70)
print(f"{'Scenario':>25s} | {'GPU Mem':>8s} | {'Transfer':>10s} | {'Compute':>10s} | {'Total':>10s}")
print("-" * 70)

for name, n_exp, k, size, bw, comp, is_offload in scenarios:
    if is_offload:
        r = simulate_offloading_latency(n_exp, k, size, bw, comp)
    else:
        r = {
            'gpu_memory_mb': n_exp * size,
            'transfer_ms': 0,
            'compute_ms': k * comp,
            'total_ms': k * comp,
        }
    
    print(f"{name:>25s} | {r['gpu_memory_mb']/1024:>6.1f}GB | {r['transfer_ms']:>8.1f}ms | "
          f"{r['compute_ms']:>8.1f}ms | {r['total_ms']:>8.1f}ms")

class CapacityMoELayer(nn.Module):
    """MoE layer with expert capacity limits.
    
    TODO: Implement this
    - Each expert can process at most `capacity_factor * (n_tokens / n_experts)` tokens
    - Tokens that exceed capacity are either dropped or sent to a second-choice expert
    - Typical capacity_factor: 1.25
    """
    def __init__(self, d_model, d_ff, n_experts, top_k=2, capacity_factor=1.25):
        super().__init__()
        self.capacity_factor = capacity_factor
        # TODO: Implement
        pass
    
    def forward(self, x):
        # TODO: Implement capacity-limited routing
        pass

class NoisyRouter(nn.Module):
    """Router with noisy top-k gating.
    
    TODO: Implement this
    - During training: add Gaussian noise to logits before top-k
    - During inference: use standard (deterministic) routing
    - The noise std should be learnable or tunable
    """
    def __init__(self, d_model, n_experts, top_k=2, noise_std=1.0):
        super().__init__()
        # TODO: Implement
        pass

def moe_memory_calculator(n_layers, d_model, d_ff, n_experts, n_heads, 
                          n_kv_heads, seq_len, batch_size, dtype_bytes=2):
    """Calculate total memory for serving an MoE model.
    
    TODO: Implement this
    Account for:
    - Attention weights (shared, not MoE)
    - Expert weights (n_experts per layer)
    - Router weights
    - KV cache
    - Activation memory
    """
    pass

# Test with Mixtral-8x7B configuration
