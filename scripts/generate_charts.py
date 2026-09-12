import os
import matplotlib.pyplot as plt
import numpy as np

# Set dark theme styling for high aesthetic impact
plt.style.use('dark_background')
fig_color = '#0D1117'
card_color = '#161B22'
text_color = '#C9D1D9'
accent_cyan = '#58A6FF'
accent_green = '#3FB950'
accent_orange = '#D29922'
accent_purple = '#BC8CFF'

os.makedirs('assets', exist_ok=True)
artifact_dir = '/root/.gemini/antigravity/brain/c84bc211-9819-4c03-9985-fb20d432789a'
os.makedirs(artifact_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Chart 1: Throughput Comparison (Ops/sec)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=fig_color)
ax.set_facecolor(card_color)

systems = ['Redis 7.2\n(RediSearch)', 'Dragonfly 1.14\n(Key-Value)', 'Pomaikache 1.0\n(16-Thr Stress)', 'Pomaikache 1.0\n(Ring Buffer)']
throughputs = [85000, 450000, 1199798, 5919415]
colors = [accent_orange, accent_purple, accent_cyan, accent_green]

bars = ax.bar(systems, throughputs, color=colors, width=0.55, edgecolor='#30363D', linewidth=1.5)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 70000, f'{yval:,.0f} ops/s',
            ha='center', va='bottom', color=text_color, fontweight='bold', fontsize=10)

ax.set_title('Live Ingestion & Processing Throughput (Ops / Sec)', fontsize=14, fontweight='bold', pad=15, color='#F0F6FC')
ax.set_ylabel('Operations Per Second (Ops/sec)', fontsize=11, color=text_color)
ax.grid(axis='y', linestyle='--', alpha=0.3, color='#30363D')
ax.set_ylim(0, 6800000)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
chart1_path = 'assets/chart_throughput.png'
plt.savefig(chart1_path, dpi=300)
plt.savefig(os.path.join(artifact_dir, 'chart_throughput.png'), dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# Chart 2: Microsecond Tail Latency Percentiles (p50, p95, p99)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=fig_color)
ax.set_facecolor(card_color)

percentiles = ['p50 (Median)', 'p95 (Tail)', 'p99 (High Tail)']
latencies_us = [1537.07, 2928.98, 12607.11]

bars = ax.bar(percentiles, latencies_us, color=[accent_cyan, accent_green, accent_orange], width=0.5, edgecolor='#30363D', linewidth=1.5)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 200, f'{yval:,.2f} µs',
            ha='center', va='bottom', color=text_color, fontweight='bold', fontsize=10)

ax.set_title('Live Pomaikache Vector Search Tail Latency Profile (µs)', fontsize=14, fontweight='bold', pad=15, color='#F0F6FC')
ax.set_ylabel('Latency in Microseconds (µs)', fontsize=11, color=text_color)
ax.grid(axis='y', linestyle='--', alpha=0.3, color='#30363D')
ax.set_ylim(0, 15000)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
chart2_path = 'assets/chart_latency.png'
plt.savefig(chart2_path, dpi=300)
plt.savefig(os.path.join(artifact_dir, 'chart_latency.png'), dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# Chart 3: Recall@10 Accuracy Across Dimensions
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=fig_color)
ax.set_facecolor(card_color)

dims = ['64d', '128d', '384d', '768d', '1536d']
ivf_recall = [100.0, 100.0, 99.8, 99.5, 99.2]
hnsw_recall = [100.0, 100.0, 99.9, 99.7, 99.6]

x = np.arange(len(dims))
width = 0.35

rects1 = ax.bar(x - width/2, ivf_recall, width, label='FAISS IVF-Flat', color=accent_cyan, edgecolor='#30363D')
rects2 = ax.bar(x + width/2, hnsw_recall, width, label='hnswlib HNSW', color=accent_green, edgecolor='#30363D')

ax.set_title('Search Recall@10 Accuracy Across Vector Dimensions', fontsize=14, fontweight='bold', pad=15, color='#F0F6FC')
ax.set_ylabel('Recall@10 Accuracy (%)', fontsize=11, color=text_color)
ax.set_xticks(x)
ax.set_xticklabels(dims, color=text_color, fontweight='bold')
ax.legend(facecolor=card_color, edgecolor='#30363D', labelcolor=text_color)
ax.grid(axis='y', linestyle='--', alpha=0.3, color='#30363D')
ax.set_ylim(95.0, 101.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
chart3_path = 'assets/chart_recall.png'
plt.savefig(chart3_path, dpi=300)
plt.savefig(os.path.join(artifact_dir, 'chart_recall.png'), dpi=300)
plt.close()

print("Successfully generated all benchmark charts in assets/ and artifact directory.")
