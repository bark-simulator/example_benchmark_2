try:
    import debug_settings
except:
    pass

import os
import logging
import pandas as pd
logging.basicConfig()

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shutil
import itertools

from src.common.plotting import *
from src.common.export_image import export_image

from load.benchmark_database import BenchmarkDatabase
from serialization.database_serializer import DatabaseSerializer
from bark.benchmark.benchmark_runner import BenchmarkResult

from bark.runtime.commons.parameters import ParameterServer
from bark.core.models.behavior import *

from src.evaluation.bark.behavior_configuration.behavior_configs import *
from src.evaluation.bark.behavior_configuration.behavior_configurate import \
        dump_defaults, create_behavior_configs

from bark.runtime.viewer.matplotlib_viewer import MPViewer
from src.evaluation.bark.analysis.custom_viewer import CustomViewer, \
      CreateAxisCollectionDownwards, THESIS_TEXTWIDTH, THESIS_TEXTHEIGHT

from bark.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import add_config_reader_module
add_config_reader_module("bark_mcts.runtime.scenario.behavior_space_sampling")

err_width_bars = .5
capsize = 0
pallette = itertools.cycle(sns.color_palette("cubehelix_r", 4))
hue_order = ["RCRSBGLocal", "RSBG"]
hue_color_dict = {}
for idx, hue in enumerate(hue_order):
    hue_color_dict[hue] = next(pallette)

estimator = lambda x: sum(x==1)*100.0/len(x)
estimator_normalized = lambda x: np.mean(x)*100.0

data_file1 = "benchmark_results_run_benchmark_parallel_mcts_final_1"
filepath = "results/benchmark/perform"
benchmark_result = BenchmarkResult.load(os.path.join(filepath, data_file1))
df1 = benchmark_result.get_data_frame()
df1 = df1[df1["exploration"] == "normal"] # exploration as risk evaluation works best for rcrsbg, no differences for rsbg
df1["search_time"] = df1["BehaviorUctBase::Mcts::MaxSearchTime"]
df1["num_mcts"] = df1["BehaviorUctBase::Mcts::NumParallelMcts"]

data_file2 = "benchmark_results_run_benchmark_parallel_mcts_final_1_single_core" # adds num parallel = 1
filepath = "results/benchmark/perform"
benchmark_result = BenchmarkResult.load(os.path.join(filepath, data_file2))
df2 = benchmark_result.get_data_frame()
df2 = df2[df2["risk"] == 0.1]
df2["num_mcts"] = df2["BehaviorUctBase::Mcts::NumParallelMcts"]

data_file3 = "benchmark_results_run_benchmark_rcrsbg_params_final_rcrsbg"
filepath = "results/benchmark/rcrsbg"
benchmark_result = BenchmarkResult.load(os.path.join(filepath, data_file3))
df3 = benchmark_result.get_data_frame()
df3 = df3[df3["risk"] == 0.1]
df3 = df3[df3["behavior"] == "RCRSBGLocal"]
df3["num_mcts"] = 0

data_file4 = "benchmark_results_run_benchmark_rcrsbg_baselines_with_risk_final"
filepath = "results/benchmark/rcrsbg"
benchmark_result = BenchmarkResult.load(os.path.join(filepath, data_file4))
df4 = benchmark_result.get_data_frame()
df4 = df4[df4["risk"] == 0.1]
df4 = df4[df4["behavior"] == "RSBG"]
df4["num_mcts"] = 0

df = pd.concat([df1, df2, df3, df4], ignore_index=True)

df["max_steps"] = df.Terminal.apply(lambda x: "max_steps" in x and (not "collision" in x))
df["success"] = df.Terminal.apply(lambda x: "success" in x and (not "collision" in x) and (not "max_steps" in x) and (not "out_of_drivable" in x))
df["collision"] = df.Terminal.apply(lambda x: ("collision_other" in x))
df["time"] = df.step * 0.2
df["safe_dist_stat_norm"] = df["safe_dist_stat"] / df.step

def format_upper_axis(axis):
    axis.xaxis.set_ticklabels([])
    axis.set_xlabel("")

df["avg_dyn_violate"] = df.safe_dist_dyn / df.step


# ------------- Figure Creation -----------
sns.set_style("whitegrid", {
    'grid.linestyle': '--'
 })
matplotlib_latex_conf(pgf_backend=False, thesis=True)

fs = latex_to_mp_figsize(latex_width_pt=THESIS_TEXTWIDTH, scale=1.0, height_scale=THESIS_TEXTHEIGHT/THESIS_TEXTWIDTH*0.3)
thesis_default_style = GetMplStyleConfigStandardThesis()
matplotlib.rcParams.update(thesis_default_style)
figure = plt.figure(figsize=fs)
gs = gridspec.GridSpec(3, 2, width_ratios=[1,1], height_ratios=[0.5, 0.1, 0.4], wspace = 0.06, hspace = 0.08,
        left = 0.1, right = 1.0, bottom = 0.2, top = 0.95)#, width_ratios=[3, 1]) 
axes = []
for idx_hor in range(0, 2):
    axes.append([])
    for idx_ver in range(0, 3):
        if idx_hor == 0:
            share_y=None
        else:
            share_y = axes[0][idx_ver]
        matplotlib.rcParams.update(thesis_default_style)
        axes[idx_hor].append(plt.subplot(gs[idx_ver, idx_hor], sharey=share_y))

df_freeway = df.loc[(df["scen_set"] == "freeway_enter")]
df_left_turn = df.loc[(df["scen_set"] == "rural_left_turn_risk")]

# freeway
# barplot - Goal Reached
sns.barplot(x="num_mcts", y="success", hue="behavior", hue_order=hue_order, estimator=estimator, data=df_freeway, errwidth=err_width_bars, capsize=capsize, palette=hue_color_dict, ax=axes[0][0])

sns.barplot(x="num_mcts", y="collision_other", hue="behavior", hue_order=hue_order, estimator=estimator, data=df_freeway, errwidth=err_width_bars, capsize=capsize, palette=hue_color_dict, ax=axes[0][1])

flierprops = dict(markerfacecolor='0.75', markersize=1.0,
              linestyle='none')

sns.boxplot(data=df_freeway,
      x="num_mcts", y="avg_dyn_violate",
      hue="behavior",ax=axes[0][2], hue_order=hue_order, linewidth=0.3, palette=hue_color_dict, flierprops=flierprops, zorder=2)

sns.barplot(x="num_mcts", y="success", hue="behavior", hue_order=hue_order, estimator=estimator, data=df_left_turn, errwidth=err_width_bars, capsize=capsize, palette=hue_color_dict, ax=axes[1][0])

# rural
# barplot - Goal Reached
sns.barplot(x="num_mcts", y="collision_other", hue="behavior", hue_order=hue_order, estimator=estimator, data=df_left_turn, errwidth=err_width_bars, capsize=capsize, palette=hue_color_dict, ax=axes[1][1])

sns.boxplot(data=df_left_turn,
      x="num_mcts", y="avg_dyn_violate",
      hue="behavior",ax=axes[1][2], hue_order=hue_order, linewidth=0.3, palette=hue_color_dict, flierprops=flierprops, zorder=2)


# x y labels
axes[0][0].set_ylabel("$P_\\text{suc} [\%]$")
axes[0][1].set_ylabel("$P_\\text{col} [\%]$")
axes[0][2].set_ylabel("Observed risk $\\beta^{*}$")

axes[0][2].set_xlabel("Num. Parallel MCTS, $\\maxsearchtime{=}\\SI{200}{\\milli\\second}$")
axes[1][2].set_xlabel("Num. Parallel MCTS, $\\maxsearchtime{=}\\SI{200}{\\milli\\second}$")
axes[1][2].xaxis.set_label_coords(0.6, -0.28)
axes[0][2].xaxis.set_label_coords(0.6, -0.28)

axes[1][0].set_ylabel("")
axes[1][1].set_ylabel("")
axes[1][2].set_ylabel("")

axes[0][0].get_yaxis().set_label_coords(-0.07, 0.5)
axes[0][1].get_yaxis().set_label_coords(-0.07, 0.55)
axes[0][2].get_yaxis().set_label_coords(-0.07, 0.3)



# remove upper xticks
for vert_idx in range(0, 2):
    for hor_idx in range(0, 2):
        format_upper_axis(axes[hor_idx][vert_idx])
       

# # hatches
# for vert_idx in range(0, 3):
#     for hor_idx in range(0, 2):
#         format_all_bars(axes[hor_idx][vert_idx])
hatch_order = ["\\\\", "//", "o", "--", ".", "xx"]
num_locations = 5
for vert_idx in range(0, 3):
  for hor_idx in range(0, 2):
    hatches = itertools.cycle(hatch_order)
    for i, bar in enumerate(axes[hor_idx][vert_idx].patches):
      if i % num_locations == 0:
          hatch = next(hatches)
      if bar.get_height() > 0.1:
          bar.set_hatch(hatch)
        
# set ylims
axes[0][0].set_ylim((0, 100))
#axes[0][1].set_ylim((0, 5.8))
axes[0][2].set_ylim((0.0, 0.6))
axes[0][2].set_yticks([0.1, 0.4])
axes[0][2].set_yticklabels(["0.1", "0.4"])

axes[0][1].set_ylim([-0.3, 0.5])
axes[0][1].set_yticks([0.0])
axes[0][1].set_yticklabels(["0"])

axes[1][2].set_xticklabels(["$\\maxsearchiterations{=}20\\text{k}$", "1", "4", "16", "64"])
axes[0][2].set_xticklabels(["$\\maxsearchiterations{=}20\\text{k}$", "1", "4", "16", "64"])

# delete legends
keep = [0, 0]
for vert_idx in range(0, 3):
    for hor_idx in range(0, 2):
        if not (vert_idx == keep[0] and hor_idx == keep[1]):
            try:
                axes[hor_idx][vert_idx].get_legend().remove()
            except:
                pass

#reverse legend order
handles, labels = axes[keep[1]][keep[0]].get_legend_handles_labels()
labels = [l.replace("Local","") for l in labels]
labels = [l.replace("RCRSBG","RC-RSBG") for l in labels]
axes[keep[1]][keep[0]].get_legend().remove()
leg = axes[keep[1]][keep[0]].legend(handles, labels, bbox_to_anchor=(0.4, -1.68, 1.2, .09), loc=7, ncol=3, mode="expand", borderaxespad=0., frameon=False, handlelength=3)
for idx, patch in enumerate(axes[keep[1]][keep[0]].get_legend().get_patches()):
    patch.set_height(7)
    patch.set_y(patch.get_y()-1)
    hatch = hatch_order[idx]
    patch.set_hatch(hatch)


# move xticks and delete middle xticks
axes[1][0].yaxis.tick_right()
#axes[1][1].yaxis.tick_right()
for vert_idx in range(0, 3):
    for hor_idx in range(0, 2):
        axes[hor_idx][vert_idx].tick_params(axis='both', which='major', pad=2.5, length=0)
        axes[hor_idx][vert_idx].axvline(x=0.5, linewidth=0.8, color= axes[hor_idx][vert_idx].spines['left']._edgecolor)
        if hor_idx > 0:
            plt.setp(axes[hor_idx][vert_idx].get_yticklabels(), visible=False)
        elif vert_idx < 2:
            axes[hor_idx][vert_idx].set_yticklabels(["{:2.0f}".format(label) for label in axes[hor_idx][vert_idx].get_yticks()])
        else:
            axes[hor_idx][vert_idx].set_yticklabels(["{:1.1f}".format(label) for label in axes[hor_idx][vert_idx].get_yticks()])

# axes[0][2].set_yticklabels(["0", "0.5", "1"])

axes[0][0].set_title("Freeway enter", pad=3)
axes[1][0].set_title("Left turn", pad=3)
plt.show(block=False)

export_image(filename=f"barplot_parallel_mcts.pdf", fig=figure, type="pgf", bbox_inches='tight',
              pad_inches=0.03, transparent=True)
home_dir = os.path.expanduser("~")
shutil.copy(f"barplot_parallel_mcts.pdf", os.path.join(home_dir, "development/thesis/results/figures"))