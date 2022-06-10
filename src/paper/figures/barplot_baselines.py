try:
    import debug_settings
except:
    pass
import matplotlib
from src.common.export_image import export_image 
from src.common.plotting import *
import seaborn as sns
sns.set_style("whitegrid", {
    'grid.linestyle': '--', "lines.linewidth": 0.5
 })
matplotlib_latex_conf(pgf_backend=True, thesis=True)
matplotlib.rcParams.update(GetMplStyleConfigStandard())
import os
import numpy as np
import pandas as pd
import shutil
import itertools
from src.evaluation.bark.analysis.custom_viewer import CustomViewer, \
      CreateAxisCollectionDownwards, THESIS_TEXTWIDTH, THESIS_TEXTHEIGHT

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from bark.benchmark.benchmark_runner import BenchmarkResult, BenchmarkRunner
# drawing params:
err_width_bars = .5
capsize = 0
pallette = itertools.cycle(sns.color_palette("cubehelix_r", 4))
hue_order = ["RSBG", "MDP", "RMDP", "Cooperative"]
hue_color_dict = {}
for idx, hue in enumerate(hue_order):
    hue_color_dict[hue] = next(pallette)

estimator = lambda x: sum(x==1)*100.0/len(x)
estimator_time = lambda x: np.mean(x)*0.2
estimator_normalized = lambda x: np.mean(x)*100.0

data_file = "benchmark_results_run_benchmark_rcrsbg_baselines_no_risk_final"
filepath = "results/benchmark/rcrsbg"
benchmark_result = BenchmarkResult.load(os.path.join(filepath, data_file))
df = benchmark_result.get_data_frame()
df["limit"] = "Max. Iterations 20k"

df["max_steps"] = df.Terminal.apply(lambda x: "max_steps" in x and (not "collision" in x))
df["success"] = df.Terminal.apply(lambda x: "success" in x and (not "collision" in x) and (not "max_steps" in x) and (not "out_of_drivable" in x))
df["collision"] = df.Terminal.apply(lambda x: ("collision_other" in x))
df["time"] = df.step * 0.2
df.loc[df["safe_dist_stat"]> 0.0, "violated"] = 1.0
df.loc[df["safe_dist_stat"] == 0.0, "violated"] = 0.0
df["safe_dist_stat_norm"] = df["safe_dist_stat"] / df.step

df["search_time"] = df["BehaviorUctBase::Mcts::MaxSearchTime"]
df["num_iter"] = df["BehaviorUctBase::Mcts::MaxNumIterations"]

df.loc[df["search_time"] == 1000, "limit"] = "Max. Search Time $\\SI{1}{\\second}$"
df.loc[df["num_iter"] == 20000, "limit"] = "Max. Iterations 20k"


# figure definition

fs = latex_to_mp_figsize(latex_width_pt=THESIS_TEXTWIDTH, scale=1.0, height_scale=THESIS_TEXTHEIGHT/THESIS_TEXTWIDTH*0.4)
figure = plt.figure(figsize=fs)
thesis_default_style = GetMplStyleConfigStandardThesis()
matplotlib.rcParams.update(thesis_default_style)
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.0], height_ratios=[0.5, 0.5], wspace = 0.06, hspace = 0.08,
        left = 0.1, right = 1.0, bottom = 0.2, top = 0.95)#, width_ratios=[3, 1]) 
axes = []
for idx_hor in range(0, 2):
    axes.append([])
    for idx_ver in range(0, 2):
        if idx_hor == 0:
            share_y=None
        else:
            share_y = axes[0][idx_ver]
        matplotlib.rcParams.update(thesis_default_style)
        axes[idx_hor].append(plt.subplot(gs[2 * idx_ver + idx_hor], sharey=share_y))

def format_upper_axis(axis):
    axis.xaxis.set_ticklabels([])
    axis.set_xlabel("")

def format_all_bars(axis):
    space = 0.005
    bar_width=0.2
    num_locations = 3

    plt.setp(axis.patches, linewidth=0.2)

    hatches = itertools.cycle(['xxx', '///'])
    for i, bar in enumerate(axis.patches):
        if i % num_locations == 0:
            hatch = next(hatches)
        if bar.get_height() > 0.1:
            bar.set_hatch(hatch)

# freeway
df_free = df.loc[df["scen_set"].str.contains("freeway")]
# barplot - Goal Reached
sns.barplot(x="limit", y="success", hue="behavior", hue_order=hue_order, estimator=estimator, data=df_free, errwidth=err_width_bars, capsize=capsize, palette=hue_color_dict, ax=axes[0][0])

#barplot - Collision
sns.barplot(x="limit", y="collision_other", hue="behavior", estimator=estimator_normalized, hue_order=hue_order, data=df_free, errwidth=err_width_bars, capsize=capsize, palette=hue_color_dict, ax=axes[0][1])



# rural
# barplot - Goal Reached
df_rural = df.loc[df["scen_set"] == "rural_left_turn_no_risk_super_dense"]

sns.barplot(x="limit", y="success", hue="behavior", hue_order=hue_order,  estimator=estimator, data=df_rural, errwidth=err_width_bars, palette=hue_color_dict, capsize=capsize, ax=axes[1][0])
# barplot - Num Steps

#barplot - Collision
sns.barplot(x="limit", y="collision_other", hue="behavior", estimator=estimator_normalized, hue_order=hue_order, data=df_rural, errwidth=err_width_bars, capsize=capsize, palette=hue_color_dict, ax=axes[1][1])


# x y labels
axes[0][0].set_ylabel("$P_\\text{suc} [\%]$")
axes[0][1].set_ylabel("$P_\\text{col}$ [\%]")
axes[0][1].set_xlabel("Planning algorithm")

axes[1][1].set_xlabel("Planning algorithm")
axes[1][1].xaxis.set_label_coords(0.5, -0.16)
axes[0][1].xaxis.set_label_coords(0.5, -0.16)

axes[1][0].set_ylabel("")
axes[1][1].set_ylabel("")
axes[1][1].set_ylabel("")

axes[0][0].get_yaxis().set_label_coords(-0.07, 0.5)
axes[0][1].get_yaxis().set_label_coords(-0.07, 0.5)

# remove upper xticks
for vert_idx in range(0, 1):
    for hor_idx in range(0, 2):
        format_upper_axis(axes[hor_idx][vert_idx])

# hatches
for vert_idx in range(0, 2):
    for hor_idx in range(0, 2):
        format_all_bars(axes[hor_idx][vert_idx])
hatch_order = ["\\\\", "//", "o", "--", ".", "xx"]
num_locations = 4
for vert_idx in range(0, 2):
  for hor_idx in range(0, 2):
    hatches = itertools.cycle(hatch_order)
    for i, bar in enumerate(axes[hor_idx][vert_idx].patches):
      if i % num_locations == 0:
          hatch = next(hatches)
      if bar.get_height() > 0.1:
          bar.set_hatch(hatch)
        
# set ylims
axes[0][0].set_ylim((0, 80))
#axes[0][1].set_ylim((0, 5.8))
axes[0][1].set_ylim((0, 25.0))
##axes[0][2].set_yticklabels(["0", "0.5", "1"])


# delete legends
keep = [1, 0]
for vert_idx in range(0, 2):
    for hor_idx in range(0, 2):
        if not (vert_idx == keep[0] and hor_idx == keep[1]):
            try:
                axes[hor_idx][vert_idx].get_legend().remove()
            except:
                pass

#reverse legend order
handles, labels = axes[keep[1]][keep[0]].get_legend_handles_labels()
labels = [l.replace("Local","") for l in labels]
axes[keep[1]][keep[0]].get_legend().remove()
leg = axes[keep[1]][keep[0]].legend(handles, labels, bbox_to_anchor=(0.3, -0.4, 1.3, .09), handletextpad=0.1,
         loc=7, ncol=6, mode="expand", borderaxespad=0., frameon=False, handlelength=2)
for idx, patch in enumerate(axes[keep[1]][keep[0]].get_legend().get_patches()):
    patch.set_height(7)
    patch.set_y(patch.get_y()-1)
    hatch = hatch_order[idx]
    patch.set_hatch(hatch)


# move xticks and delete middle xticks
axes[1][0].yaxis.tick_right()
axes[1][1].yaxis.tick_right()

for vert_idx in range(0, 2):
    for hor_idx in range(0, 2):
        axes[hor_idx][vert_idx].tick_params(axis='both', which='major', pad=2.5, length=0)
        if hor_idx > 0:
            plt.setp(axes[hor_idx][vert_idx].get_yticklabels(), visible=False)
        else:
            axes[hor_idx][vert_idx].set_yticklabels(["{:2.0f}".format(label) for label in axes[hor_idx][vert_idx].get_yticks()])

# axes[0][2].set_yticklabels(["0", "0.5", "1"])

axes[0][0].set_title("Freeway enter", pad=3)
axes[1][0].set_title("Left turn", pad=3)
plt.show(block=True)
export_image(filename=f"barplot_baselines.pdf", fig=figure, type="pgf", bbox_inches='tight',
              pad_inches=0, transparent=True)
home_dir = os.path.expanduser("~")
shutil.copy(f"barplot_baselines.pdf", os.path.join(home_dir, "development/thesis/results/figures"))
