try:
    import debug_settings
except:
    pass
import matplotlib
from src.common.export_image import export_image 
from src.common.plotting import *
import seaborn as sns
import shutil

import os
import numpy as np
import pandas as pd
import itertools

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from bark.benchmark.benchmark_runner import BenchmarkResult, BenchmarkRunner
from src.evaluation.bark.analysis.custom_viewer import CustomViewer, \
      CreateAxisCollectionDownwards, THESIS_TEXTWIDTH, THESIS_TEXTHEIGHT
# drawing params:
err_width_bars = .5
capsize = 0
pallette = itertools.cycle(sns.color_palette("cubehelix_r", 6))
hue_order = ["RCRSBGLocal", "RCRSBGLocalFullInfo", "RSBG", "MDP", "RMDP", "Cooperative"]
hue_color_dict = {}
for idx, hue in enumerate(hue_order):
    hue_color_dict[hue] = next(pallette)

estimator = lambda x: sum(x==1)*100.0/len(x)
estimator_time = lambda x: sum(x)*0.2/len(x)
from colour import Color

data_file = "benchmark_results_run_benchmark_rcrsbg_params_final_full"
filepath = "results/benchmark/rcrsbg"
benchmark_result = BenchmarkResult.load(os.path.join(filepath, data_file))
df_full = benchmark_result.get_data_frame()

data_file = "benchmark_results_run_benchmark_rcrsbg_params_final_rcrsbg"
filepath = "results/benchmark/rcrsbg"
benchmark_result = BenchmarkResult.load(os.path.join(filepath, data_file))
df_rcrsbg = benchmark_result.get_data_frame()

data_file = "benchmark_results_run_benchmark_rcrsbg_baselines_with_risk_final"
filepath = "results/benchmark/rcrsbg"
benchmark_result = BenchmarkResult.load(os.path.join(filepath, data_file))
df_baselines = benchmark_result.get_data_frame()

df = pd.concat([df_full, df_rcrsbg, df_baselines], ignore_index=True)
df["max_steps"] = df.Terminal.apply(lambda x: "max_steps" in x and (not "collision" in x))
df["success"] = df.Terminal.apply(lambda x: "success" in x and (not "collision" in x) and (not "max_steps" in x) and (not "out_of_drivable" in x))
df["collision"] = df.Terminal.apply(lambda x: ("collision_other" in x))
df["time"] = df.step * 0.2
df.loc[df["safe_dist_stat"]> 0.0, "violated"] = 1.0
df.loc[df["safe_dist_stat"] == 0.0, "violated"] = 0.0
# figure definition

sns.set_style("whitegrid", {
    'grid.linestyle': '--', "lines.linewidth": 0.5
 })
matplotlib_latex_conf(pgf_backend=True, thesis=True)

fs = latex_to_mp_figsize(latex_width_pt=THESIS_TEXTWIDTH, scale=1.0, height_scale=THESIS_TEXTHEIGHT/THESIS_TEXTWIDTH*0.4)
thesis_default_style = GetMplStyleConfigStandardThesis()
matplotlib.rcParams.update(thesis_default_style)
plt.figure(figsize=fs)
gs = gridspec.GridSpec(3, 1, width_ratios=[1], height_ratios=[0.6, 0.3, 0.4], wspace = 0.02, hspace = 0.12,
        left = 0.1, right = 1.0, bottom = 0.1, top = 0.95)#, width_ratios=[3, 1]) 
axes = []
for idx_hor in range(0, 1):
    axes.append([])
    for idx_ver in range(0, 3):
        if idx_hor == 0:
            share_y=None
        else:
            share_y = axes[0][idx_ver]
        axes[idx_hor].append(plt.subplot(gs[idx_ver, :], sharey=share_y))

def format_upper_axis(axis):
    axis.xaxis.set_ticklabels([])
    axis.set_xlabel("")

data_frame_freeway = df[df.scen_set == "freeway_enter"]
data_frame_rural = df[df.scen_set == "rural_left_turn"]

# freeway
# barplot - Goal Reached
sns.barplot(x="risk", y="success", hue="behavior", hue_order=hue_order, estimator=estimator, data=data_frame_freeway, errwidth=err_width_bars, capsize=capsize, palette=hue_color_dict, ax=axes[0][0])
# barplot - Num Steps
data_num_steps = data_frame_freeway[(data_frame_freeway.success == 1) ] 
sns.barplot(x="risk", y="step", hue="behavior", hue_order=hue_order, data=data_num_steps, estimator=estimator_time, errwidth=err_width_bars, capsize=capsize, palette=hue_color_dict, ax=axes[0][1])

#barplot - Collision
sns.barplot(x="risk", y="collision_other", hue="behavior", estimator=estimator, hue_order=hue_order, data=data_frame_freeway, errwidth=err_width_bars, capsize=capsize, palette=hue_color_dict, ax=axes[0][2])


# x y labels
axes[0][2].set_xlabel("")
axes[0][2].xaxis.set_label_coords(0.5, -0.17)
axes[0][0].set_ylabel("$P_\\text{suc} [\%]$")
axes[0][1].set_ylabel("$\overline{T}_\\text{suc}$ [s]")
axes[0][2].set_ylabel("$P_\\text{col}$ [\%]")
axes[0][2].set_xlabel("Specified risk \SRisklevel")

# axes[1][2].set_xlabel("Specified risk \SRisklevel")
# axes[1][2].xaxis.set_label_coords(0.5, -0.55)
# axes[1][0].set_ylabel("")
# axes[1][1].set_ylabel("")
# axes[1][2].set_ylabel("")

axes[0][0].get_yaxis().set_label_coords(-0.05, 0.6)
axes[0][1].get_yaxis().set_label_coords(-0.05, 0.5)
axes[0][2].get_yaxis().set_label_coords(-0.05, 0.3)

# remove upper xticks
for vert_idx in range(0, 2):
    for hor_idx in range(0, 1):
        format_upper_axis(axes[hor_idx][vert_idx])

# hatches
hatch_order = ["\\\\", "//", "o", "--", ".", "xx"]
num_locations = 7
for vert_idx in range(0, 3):
  for hor_idx in range(0, 1):
    hatches = itertools.cycle(hatch_order)
    for i, bar in enumerate(axes[hor_idx][vert_idx].patches):
      if i % num_locations == 0:
          hatch = next(hatches)
      if bar.get_height() > 0.1:
          bar.set_hatch(hatch)
        
# set ylims
axes[0][0].set_ylim((0, 100))
#axes[0][1].set_ylim((0, 5.8))
#axes[0][2].set_ylim((0, 7.0))
##axes[0][2].set_yticklabels(["0", "0.5", "1"])


# delete legends
keep = [1, 0]
for vert_idx in range(0, 3):
    for hor_idx in range(0, 1):
        if not (vert_idx == keep[0] and hor_idx == keep[1]):
            try:
                axes[hor_idx][vert_idx].get_legend().remove()
            except:
                pass

handles, labels = axes[keep[1]][keep[0]].get_legend_handles_labels()
labels = [l.replace("Local","") for l in labels]
labels = [l.replace("RCRSBG","RC-RSBG") for l in labels]
axes[keep[1]][keep[0]].get_legend().remove()
leg = axes[keep[1]][keep[0]].legend(handles, labels, bbox_to_anchor=(0.0, -2.55, 1.0, .09), loc=7, ncol=3, mode="expand", borderaxespad=0., frameon=False, handlelength=3)
for idx, patch in enumerate(axes[keep[1]][keep[0]].get_legend().get_patches()):
    patch.set_height(7)
    patch.set_y(patch.get_y()-1)
    hatch = hatch_order[idx]
    patch.set_hatch(hatch)


# move xticks and delete middle xticks
# axes[1][0].yaxis.tick_right()
# axes[1][1].yaxis.tick_right()
# axes[1][2].yaxis.tick_right()
for vert_idx in range(0, 3):
    for hor_idx in range(0, 1):
        axes[hor_idx][vert_idx].tick_params(axis='both', which='major', pad=2.5, length=0)
        if hor_idx > 0:
            plt.setp(axes[hor_idx][vert_idx].get_yticklabels(), visible=False)
        else:
            axes[hor_idx][vert_idx].set_yticklabels(["{:2.0f}".format(label) for label in axes[hor_idx][vert_idx].get_yticks()])

# axes[0][2].set_yticklabels(["0", "0.5", "1"])

#axes[1][0].set_title("Left turn", pad=3)
plt.show(block=True)
export_image(filename="barplot_risk_freeway.pgf", fig=plt.gcf(), bbox_inches='tight',
                pad_inches=0)
home_dir = os.path.expanduser("~")
shutil.copy(f"barplot_risk_freeway.pdf", os.path.join(home_dir, "development/thesis/results/figures"))
