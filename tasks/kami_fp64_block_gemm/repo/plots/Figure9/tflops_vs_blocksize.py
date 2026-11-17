import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib.patches import Ellipse

sns.set_theme(style='whitegrid',
              rc={
                  'font.family': 'Times New Roman',
                  'font.weight': 'bold'
              })

marker_size = 12
line_width = 3

font_size = 19

file_paths = {
    "1d": "../../logs/tflops_vs_blocksize/remap_5090_fp16_1d.csv",
    "2d": "../../logs/tflops_vs_blocksize/remap_5090_fp16_2d.csv",
    "3d": "../../logs/tflops_vs_blocksize/remap_5090_fp16_3d.csv"
}

dataframes = {}
for key, path in file_paths.items():
    dataframes[key] = pd.read_csv(path,
                                  header=None,
                                  names=[
                                      "dimension", "mnk", "n", "k",
                                      "NUM_RANK1", "NUM_RANK2", "tflops",
                                      "Block size"
                                  ])

all_mnk_values = set()
for df in dataframes.values():
    all_mnk_values.update(df["mnk"].unique())

block_order = [32, 64, 96, 128, 256, 512]
x_vals = list(range(len(block_order)))

plt.figure(figsize=(8, 3))

plt.xlabel("Block size", fontsize=font_size, weight='bold')
plt.ylabel("TFLOPS", fontsize=font_size, weight='bold')
plt.ylim(100, 500)

markers = {'1d': 'o', '2d': 's', '3d': '^'}

for key in ["1d", "2d", "3d"]:
    df = dataframes[key]
    sub_df = df[df["mnk"] == 64][["Block size", "tflops"]]
    if not sub_df.empty:
        sub_df = (sub_df.drop_duplicates(
            subset=["Block size"]).set_index("Block size").reindex(block_order)
                  )
        plt.plot(x_vals,
                 sub_df["tflops"],
                 marker=markers[key],
                 label=key,
                 markersize=marker_size,
                 linewidth=line_width)

        plt.legend(labels=['KAMI-1D', 'KAMI-2D', 'KAMI-3D'],
                   fontsize=font_size,
                   loc='lower center',
                   ncol=3,
                   bbox_to_anchor=(0.5, 0.89),
                   columnspacing=1.2,
                   handletextpad=0.5,
                   frameon=False,
                   handlelength=2.5)

plt.xticks(x_vals, block_order, fontsize=font_size)
plt.yticks(fontsize=font_size)

plt.tight_layout()

plt.grid(True, linestyle='--')
ax = plt.gca()
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')

plt.savefig("tflops_vs_blocksize.pdf", bbox_inches='tight', pad_inches=0.02)

stats_dict = {}
for dim in ['1d', '2d', '3d']:
    df = dataframes[dim]
    df = df[df["mnk"] == 64][["Block size", "tflops"]]
    print(dim, df)
    stats = {
        'variance': df['tflops'].var(),
        'std_dev': df['tflops'].std(),
        'cv': (df['tflops'].std() / df['tflops'].mean()) * 100
    }
    stats_dict[dim] = stats
    print(f"{dim.upper()}:")
    print(f"Variance: {stats['variance']:.2f}")
    print(f"Standard Deviation: {stats['std_dev']:.2f}")
    print(f"Coefficient of Variation: {stats['cv']:.2f}%")
    print("-" * 30)

df_1d = dataframes['1d'][dataframes['1d']['mnk'] == 64]
df_2d = dataframes['2d'][dataframes['2d']['mnk'] == 64]
df_3d = dataframes['3d'][dataframes['3d']['mnk'] == 64]

tflops_1d_max = (df_1d[["Block size", "tflops"
                        ]].drop_duplicates(subset=["Block size"]).set_index(
                            "Block size").reindex(block_order)["tflops"].max())

tflops_2d_max = (df_2d[["Block size", "tflops"
                        ]].drop_duplicates(subset=["Block size"]).set_index(
                            "Block size").reindex(block_order)["tflops"].max())

tflops_3d_max = (df_3d[["Block size", "tflops"
                        ]].drop_duplicates(subset=["Block size"]).set_index(
                            "Block size").reindex(block_order)["tflops"].max())

print(f"Max TFLOPS for 1D: {tflops_1d_max:.2f}")
print(f"Max TFLOPS for 2D: {tflops_2d_max:.2f}")
print(f"Max TFLOPS for 3D: {tflops_3d_max:.2f}")

speedup = (df_2d[df_2d['Block size'] == 64]['tflops'].values[0] /
           df_1d[df_1d['Block size'] == 64]['tflops'].values[0]) * 100
print(
    f"When block size is 64, 2D performance is {speedup:.2f}% of 1D performance"
)
