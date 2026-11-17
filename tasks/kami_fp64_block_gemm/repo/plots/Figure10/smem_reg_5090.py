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

color_pattern = sns.color_palette()

df = pd.read_csv('../../logs/reg_smem/reg_smem_5090.csv')
df['SMEM_K_SPLIT_BLOCK'] = df['M_BLOCK'] / 16.0 - df['REG_K_SPLIT_BLOCK']

plt_m_block = [32, 64, 128]
plt_m_block = [32, 64, 96, 128]

plt.figure(figsize=(8, 3))

colors = ['blue', 'red', 'green', 'orange']
markers = ['o', 's', '^', 'D']

for i, m_block in enumerate(plt_m_block):

    df_m_block = df[df['M_BLOCK'] == m_block]

    df_m_block = df_m_block.sort_values('REG_K_SPLIT_BLOCK')

    x_values = (df_m_block['SMEM_K_SPLIT_BLOCK'] / (m_block / 16.0)) * 100

    plt.plot(x_values,
             df_m_block['Tflops'],
             marker=markers[i],
             linestyle='-',
             color=color_pattern[i],
             label=f'{m_block}',
             markersize=marker_size,
             linewidth=line_width)

ax = plt.gca()

all_x_values = []
for m_block in plt_m_block:
    df_m_block = df[df['M_BLOCK'] == m_block]
    x_values = (df_m_block['SMEM_K_SPLIT_BLOCK'] / (m_block / 16.0)) * 100
    all_x_values.extend(x_values.unique())
all_x_values = sorted(list(set([float(x) for x in all_x_values])))
ax.set_xticks(all_x_values)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))

plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)

plt.xlabel('Shared memory ratio', fontsize=font_size, weight='bold')
plt.ylabel('TFLOPS', fontsize=font_size, weight='bold')

plt.legend(
    loc='lower center',
    bbox_to_anchor=(0.63, 0.89),
    fontsize=font_size,
    ncol=4,
    columnspacing=0.6,
    handletextpad=0.5,
    frameon=False,
    handlelength=2.5,
)

plt.text(0, 1.066, 'Matrix order:', transform=ax.transAxes, fontsize=font_size)

plt.ylim(180, 530)

plt.grid(True, linestyle='--')
ax = plt.gca()
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')

plt.tight_layout()

plt.savefig('smem_reg_cooperation.pdf', bbox_inches='tight', pad_inches=0.02)

df_32 = df[df['M_BLOCK'] == 32]

tflops_reg1 = df_32[df_32['REG_K_SPLIT_BLOCK'] == 1]['Tflops'].values[0]
tflops_reg2 = df_32[df_32['REG_K_SPLIT_BLOCK'] == 2]['Tflops'].values[0]

percentage = (tflops_reg1 / tflops_reg2) * 100
print(
    f"For m=32, TFLOPS at REG_K_SPLIT_BLOCK=1 is {percentage:.2f}% of TFLOPS at REG_K_SPLIT_BLOCK=2"
)

df_128 = df[df['M_BLOCK'] == 128]

tflops_base = df_128[df_128['REG_K_SPLIT_BLOCK'] == 8]['Tflops'].values[0]

for reg_k in sorted(df_128['REG_K_SPLIT_BLOCK'].unique()):
    tflops = df_128[df_128['REG_K_SPLIT_BLOCK'] == reg_k]['Tflops'].values[0]
    ratio = tflops / tflops_base
    print(
        f"For REG_K_SPLIT_BLOCK={reg_k}, TFLOPS is {ratio:.2f}x of baseline (REG_K_SPLIT_BLOCK=8)"
    )
