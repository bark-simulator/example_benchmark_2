try:
    import debug_settings
except:
    pass

import os
import logging
import pandas as pd
logging.basicConfig()
import numpy as np

from src.common.plotting import *
from src.common.export_image import export_image

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import shutil
from os.path import expanduser
import itertools

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

def format_upper_axis(axis):
    axis.xaxis.set_ticklabels([])
    axis.set_xlabel("")

err_width_bars = .5
capsize = 0
pallette = itertools.cycle(sns.color_palette("cubehelix_r", 2))
hue_order = ["RSBG_wo_risk", "IntentRSBG_wo_risk"]
hue_color_dict = {}
for hue in hue_order:
    hue_color_dict[hue] = next(pallette)
estimator = lambda x: sum(x==1)*100.0/len(x)
estimator_time = lambda x: np.mean(x)*0.2
estimator_normalized = lambda x: np.mean(x)*100.0

data_file = "benchmark_results_run_benchmark_intent_compare_final"
filepath = "results/benchmark/behavior_spaces"
benchmark_result = BenchmarkResult.load(os.path.join(filepath, data_file))
df = benchmark_result.get_data_frame()

df["max_steps"] = df.Terminal.apply(lambda x: "max_steps" in x and (not "collision" in x))
df["success"] = df.Terminal.apply(lambda x: "success" in x and (not "collision" in x) and (not "max_steps" in x) and (not "out_of_drivable" in x))
df["time"] = df.step * 0.2
df["safe_dist_stat_norm"] = df["safe_dist_stat"] / df.step
sns.set_style("whitegrid", {
    'grid.linestyle': '--'
 })

matplotlib_latex_conf(pgf_backend=False, thesis=True)

fs = latex_to_mp_figsize(latex_width_pt=THESIS_TEXTWIDTH, scale=1.0, height_scale=THESIS_TEXTHEIGHT/THESIS_TEXTWIDTH*0.4)
figure = plt.figure(figsize=fs)
thesis_default_style = GetMplStyleConfigStandardThesis()
matplotlib.rcParams.update(thesis_default_style)
gs = gridspec.GridSpec(2, 2, width_ratios=[1,1], height_ratios=[0.5, 0.5], wspace = 0.06, hspace = 0.08,
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

df_intent = df.loc[df["scen_set"].str.contains("intent")]
df_no_intent = df.loc[~df["scen_set"].str.contains("intent")]

#freeway
#barplot - Goal Reached
sns.barplot(x="scen_set", y="success", hue="behavior", hue_order=hue_order, estimator=estimator, data=df_intent, errwidth=err_width_bars, capsize=capsize, palette=hue_color_dict, ax=axes[0][0])

#barplot - Collision
sns.barplot(x="scen_set", y="collision_other", hue="behavior", estimator=estimator, hue_order=hue_order, data=df_intent, errwidth=err_width_bars, capsize=capsize, palette=hue_color_dict, ax=axes[0][1])

# rural
# barplot - Goal Reached
sns.barplot(x="scen_set", y="success", hue="behavior", hue_order=hue_order, estimator=estimator, data=df_no_intent, errwidth=err_width_bars, capsize=capsize, palette=hue_color_dict, ax=axes[1][0])

#barplot - Collision
sns.barplot(x="scen_set", y="collision_other", hue="behavior", hue_order=hue_order, data=df_no_intent, estimator=estimator, errwidth=err_width_bars, capsize=capsize, palette=hue_color_dict, ax=axes[1][1])



# x y labels
axes[0][0].set_ylabel("$P_\\text{suc} [\%]$")
axes[0][1].set_ylabel("$P_\\text{col}$ [\%]")
axes[1][1].set_xlabel("Scenario type")
axes[0][1].set_xlabel("Scenario type")
axes[1][1].xaxis.set_label_coords(0.5, -0.14)
axes[0][1].xaxis.set_label_coords(0.5, -0.14)

scen_set_to_label = {"rural_left_turn_no_risk" : "Left turn",
                     "rural_left_turn_no_risk_intent" : "Left turn",
                     "freeway_enter" : "Freeway enter",
                     "freeway_enter_intent" : "Freeway enter"}
xlabels = axes[0][0].get_xticklabels()
xlabels = [scen_set_to_label[label.get_text()] for label in xlabels]
axes[1][1].set_xticklabels(xlabels)
axes[0][1].set_xticklabels(xlabels)

axes[1][0].set_ylabel("")
axes[1][1].set_ylabel("")

axes[0][0].get_yaxis().set_label_coords(-0.07, 0.5)
axes[0][1].get_yaxis().set_label_coords(-0.07, 0.5)

# remove upper xticks
for vert_idx in range(0, 1):
    for hor_idx in range(0, 2):
        format_upper_axis(axes[hor_idx][vert_idx])

# # hatches
# for vert_idx in range(0, 3):
#     for hor_idx in range(0, 2):
#         format_all_bars(axes[hor_idx][vert_idx])
hatch_order = ["\\\\", "//", "o", "--", ".", "xx"]
num_locations = 2
for vert_idx in range(0, 2):
  for hor_idx in range(0, 2):
    hatches = itertools.cycle(hatch_order)
    for i, bar in enumerate(axes[hor_idx][vert_idx].patches):
      if i % num_locations == 0:
          hatch = next(hatches)
      if bar.get_height() > 0.1:
          bar.set_hatch(hatch)
        
# set ylims
axes[0][0].set_ylim((0, 100))
axes[0][1].set_ylim((0, 15.0))
##axes[0][2].set_yticklabels(["0", "0.5", "1"])


# delete legends
keep = [1,0]
for vert_idx in range(0, 2):
    for hor_idx in range(0, 2):
        if not (vert_idx == keep[0] and hor_idx == keep[1]):
            try:
                axes[hor_idx][vert_idx].get_legend().remove()
            except:
                pass

#reverse legend order
handles, labels = axes[keep[1]][keep[0]].get_legend_handles_labels()
labels = [l.replace("_wo_risk","") for l in labels]
axes[keep[1]][keep[0]].get_legend().remove()
leg = axes[keep[1]][keep[0]].legend(handles, labels, bbox_to_anchor=(0.4, -0.4, 1.0, .09), loc=7, ncol=4, mode="expand", borderaxespad=0., frameon=False, handlelength=3)
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

axes[0][0].set_title("With intent simulation", pad=3)
axes[1][0].set_title("W/o intent simulation", pad=3)
plt.show(block=False)

export_image(filename=f"barplot_compare_intents.pdf", fig=figure, type="pgf", bbox_inches='tight',
              pad_inches=0.03, transparent=True)
home_dir = os.path.expanduser("~")
shutil.copy(f"barplot_compare_intents.pdf", os.path.join(home_dir, "development/thesis/results/figures"))