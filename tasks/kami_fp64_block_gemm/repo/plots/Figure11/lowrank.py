import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

sns.set_theme(style='whitegrid',
              rc={
                  'font.family': 'Times New Roman',
                  'font.weight': 'bold'
              })

marker_size = 12
line_width = 3
font_size = 20
legend_font_size = 20

kami_paths = {
    "KAMI-1D": "../../logs/block_gemm/lowrank/H200/fp16_block_lowrank_H200.csv"
}
cublasdx_path = "../../logs/block_gemm/lowrank/H200/cublasdx.csv"
cutlass_path = "../../logs/block_gemm/lowrank/H200/CUTLASS.csv"

cublasdx_df = pd.read_csv(cublasdx_path, header=None)
cublasdx_df.columns = ["m", "n", "k", "blocksize", "gflops", "tflops"]
cublasdx_df["dimension"] = "cuBLASDx"

kami_dfs = []
for path in kami_paths.values():
    df = pd.read_csv(path, header=None)
    df.columns = [
        "dimension", "m", "n", "k", "num1", "NUM_ALLOC_RANK_BLOCK", "tflops",
        "blocksize"
    ]
    df["dimension"] = "KAMI"
    kami_dfs.append(df)
kami_df = pd.concat(kami_dfs, ignore_index=True)
kami_df = (kami_df.drop_duplicates(subset=["dimension", "m", "n", "k"]).
           loc[:, ["dimension", "m", "n", "k", "tflops"]])
cutlass_df = pd.read_csv(cutlass_path, header=None)
cutlass_df.columns = [
    "m", "n", "k", "blocksize", "gridsize", "gflops", "tflops"
]
cutlass_df = cutlass_df[cutlass_df["m"] == cutlass_df["n"]]
cutlass_df["dimension"] = "CUTLASS"
cutlass_df = (cutlass_df.drop_duplicates(subset=["dimension", "m", "n", "k"]).
              loc[:, ["dimension", "m", "n", "k", "tflops"]])

all_data = pd.concat([kami_df, cublasdx_df, cutlass_df], ignore_index=True)

m_n_values = [16, 32, 64, 128, 192]
k_values = [16, 32]

colors = {"KAMI": "#9fcff3", "cuBLASDx": "#f4a7a7", "CUTLASS": "#c3e2b4"}
edges = {"KAMI": "#2b5dad", "cuBLASDx": "#a33d3d", "CUTLASS": "#4b7d3a"}
hatch_styles = {
    "cuBLASDx": ('\\\\', edges["cuBLASDx"]),
    "KAMI": ('//', edges["KAMI"]),
    "CUTLASS": ('xx', edges["CUTLASS"])
}
unique_dims = ["cuBLASDx", "CUTLASS", "KAMI"]

y_max = all_data[(all_data["m"] == all_data["n"])
                 & (all_data["m"].isin(m_n_values))]["tflops"].max()
y_max = (int(y_max / 0.5) + 1) * 0.5

for k in k_values:
    k_data = all_data[(all_data["k"] == k) & (all_data["m"] == all_data["n"]) &
                      (all_data["m"].isin(m_n_values))]
    if k_data.empty:
        continue

    fig, ax = plt.subplots(figsize=(4, 2))
    bar_width = 0.25
    m_values = sorted(k_data["m"].unique())
    positions = list(range(len(m_values)))
    n_bars = len(unique_dims)
    total_width = bar_width * n_bars
    start_pos = -total_width / 2 + bar_width / 2

    for i, dim in enumerate(unique_dims):
        dim_data = k_data[k_data["dimension"] == dim]
        dim_data = (dim_data.drop_duplicates(
            subset=["m"]).set_index("m")["tflops"].reindex(m_values,
                                                           fill_value=0))
        bar_pos = [p + start_pos + i * bar_width for p in positions]
        bars = ax.bar(bar_pos,
                      dim_data,
                      bar_width,
                      facecolor=colors[dim],
                      edgecolor=edges[dim],
                      linewidth=1.2,
                      hatch=hatch_styles[dim][0])
        for bar in bars:
            bar.set_hatch(hatch_styles[dim][0])
            bar.set_edgecolor(hatch_styles[dim][1])

    ax.set_xlabel('Matrix order (m & n)', fontsize=font_size, weight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels([str(m) for m in m_values], fontsize=font_size)

    ax.tick_params(axis='x', labelsize=font_size)

    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    if k == 16:
        ax.set_yticklabels([f"{int(t)}" for t in yticks], fontsize=font_size)
        ax.set_ylabel('TFLOPS', fontsize=font_size, weight='bold')
    else:
        ax.set_yticklabels(['' for _ in yticks], fontsize=font_size)

    ax.yaxis.grid(True, linestyle='-', linewidth=1.0, color='lightgray')
    ax.set_ylim(0, 660)

    for spine in ax.spines.values():
        spine.set_color('black')

    plt.grid(True, linestyle='--')
    plt.savefig(f"lowrank_H200_k_{k}.pdf",
                bbox_inches='tight',
                pad_inches=0.02)

for k in [16, 32]:
    print(f"\n=================== k = {k} ===================")
    k_data = all_data[(all_data["k"] == k) & (all_data["m"] == all_data["n"]) &
                      (all_data["m"].isin(m_n_values))]
    grouped = k_data.pivot(index='m', columns='dimension', values='tflops')

    if "KAMI" in grouped and "cuBLASDx" in grouped:
        speedup_cublasdx = grouped["KAMI"] / grouped["cuBLASDx"]
        print(f"\nKAMI vs cuBLASDx @ k={k}")
        print("m\tKAMI\tcuBLASDx\tSpeedup")
        for m in speedup_cublasdx.index:
            kami_val = grouped.loc[m, "KAMI"]
            dx_val = grouped.loc[m, "cuBLASDx"]
            print(
                f"{m}\t{kami_val:.2f}\t{dx_val:.2f}\t\t{kami_val / dx_val:.2f}x"
            )
        print(f"Average speedup: {speedup_cublasdx.mean():.2f}x")
        print(f"Maximum speedup: {speedup_cublasdx.max():.2f}x")

    if "KAMI" in grouped and "CUTLASS" in grouped:
        speedup_cutlass = grouped["KAMI"] / grouped["CUTLASS"]
        print(f"\nKAMI vs CUTLASS @ k={k}")
        print("m\tKAMI\tCUTLASS\t\tSpeedup")
        for m in speedup_cutlass.index:
            kami_val = grouped.loc[m, "KAMI"]
            cutlass_val = grouped.loc[m, "CUTLASS"]
            print(
                f"{m}\t{kami_val:.2f}\t{cutlass_val:.2f}\t\t{kami_val / cutlass_val:.2f}x"
            )
        print(f"Average speedup: {speedup_cutlass.mean():.2f}x")
        print(f"Maximum speedup: {speedup_cutlass.max():.2f}x")

fig = plt.figure(figsize=(0, 0))

colors = {"KAMI": "#9fcff3", "cuBLASDx": "#f4a7a7", "CUTLASS": "#c3e2b4"}
edges = {"KAMI": "#2b5dad", "cuBLASDx": "#a33d3d", "CUTLASS": "#4b7d3a"}
hatch_styles = {
    "cuBLASDx": ('\\\\', edges["cuBLASDx"]),
    "KAMI": ('//', edges["KAMI"]),
    "CUTLASS": ('xx', edges["CUTLASS"])
}

legend_elements = [
    Patch(facecolor=colors[dim],
          edgecolor=edges[dim],
          hatch=hatch_styles[dim][0],
          label=dim,
          linewidth=1.2) for dim in ["cuBLASDx", "CUTLASS", "KAMI"]
]

legend = plt.legend(handles=legend_elements,
                    loc='center',
                    ncol=3,
                    frameon=False,
                    fontsize=20,
                    labelspacing=0.2,
                    columnspacing=0.6,
                    handletextpad=0.3,
                    borderpad=0.1)

plt.axis('off')

plt.savefig("lowrank_legend.pdf",
            bbox_inches='tight',
            pad_inches=0,
            transparent=True)