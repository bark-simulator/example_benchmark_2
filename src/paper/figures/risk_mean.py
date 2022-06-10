try:
    import debug_settings
except:
    pass

from src.common.export_image import export_image 
from src.common.plotting import *
import seaborn as sns

import os
import numpy as np
import pandas as pd
import math
import itertools
import shutil

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from bark.benchmark.benchmark_runner import BenchmarkResult, BenchmarkRunner
from src.evaluation.bark.analysis.custom_viewer import CustomViewer, \
      CreateAxisCollectionDownwards, THESIS_TEXTWIDTH, THESIS_TEXTHEIGHT

err_width_bars = .5
capsize = 0
pallette = itertools.cycle(sns.color_palette("cubehelix_r", 6))
hue_order = ["RCRSBGLocal", "RCRSBGLocalFullInfo", "RSBG"]
hue_color_dict = {}
for idx, hue in enumerate(hue_order):
    hue_color_dict[hue] = next(pallette)

err_width_bars = .5
capsize = 0
pallette = itertools.cycle(sns.color_palette("cubehelix_r", 6))
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
df["avg_dyn_violate"] = df.safe_dist_dyn / df.step

def waiting_time(p_max_steps, p_success, t_mean_success, t_max_sim):
  # caculates expected waiting time for a given scenario with mean success time and max simulation time
  # based on geometric series t_wait = sum_{k=0}^{infty}(tmax_sim*k+t_mean_success)*p_success*p_max_steps^k
  # is simpliefied into to series sum_{k=0}^{infty}k*p^k and sum_{k=0}^{infty} p^k with convergence of sums
  if p_success == 0.0:
    return np.nan
  t_wait = p_success*(t_max_sim*p_max_steps/((1-p_max_steps)**2)+ t_mean_success*1/(1-p_max_steps))
  return t_wait

sns.set_style("whitegrid", {
    'grid.linestyle': '--', "lines.linewidth": 0.5
 })
matplotlib_latex_conf(pgf_backend=False, thesis=True)

def waiting_time_df(row):
  return waiting_time(row["max_steps"], row["success"], row["step"]*0.2, 30*0.2)
matplotlib.rcParams.update({"lines.linewidth" : 1.0})
# figure definition
fs = latex_to_mp_figsize(latex_width_pt=THESIS_TEXTWIDTH, scale=1.0, height_scale=THESIS_TEXTHEIGHT/THESIS_TEXTWIDTH*0.4)
figure = plt.figure(figsize=fs)
thesis_default_style = GetMplStyleConfigStandardThesis()
matplotlib.rcParams.update(thesis_default_style)

data_frame = df.replace(to_replace={"freeway_enter" : "Freeway enter", "rural_left_turn_risk" : "Left turn"})
data_frame_steps = data_frame[data_frame.success == 1]
data_frame_freeway = data_frame[data_frame.scen_set == "Freeway enter"]
data_frame_rural = data_frame[data_frame.scen_set == "Left turn"]
data_frame_freeway_steps = data_frame_steps[data_frame_steps.scen_set == "Freeway enter"]
data_frame_rural_steps = data_frame_steps[data_frame_steps.scen_set == "Left turn"]
plt.figure(figsize=fs)
gs = gridspec.GridSpec(2, 2, width_ratios=[1,1], height_ratios=[1.0, 0.5], wspace = 0.02, hspace = 0.05,
        left = 0.05, right = 0.95, bottom = 0.1, top = 0.95)#, width_ratios=[3, 1]) 
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1], sharey=ax1)

ax3 = plt.subplot(gs[2], sharex=ax1)
ax4 = plt.subplot(gs[3], sharex=ax2, sharey=ax3)


flierprops = dict(markerfacecolor='0.75', markersize=1.0,
              linestyle='none')

box_width = 0.5
num_boxes = 7
x =[-box_width + box_idx*(2*box_width) for box_idx in range(0, num_boxes+1)]
y=[0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0]

ax1.fill_between(x,y, step="post", alpha=0.3, zorder=0.0, color="lightgrey")
ax2.fill_between(x,y, step="post", alpha=0.3, zorder=0.0, color="lightgrey")
sns.boxplot(data=data_frame_freeway,
      x="risk", y="avg_dyn_violate",
      hue="behavior",ax=ax1, hue_order=hue_order, linewidth=0.3, palette=hue_color_dict, flierprops=flierprops, zorder=2)

sns.boxplot(data=data_frame_rural,
      x="risk", y="avg_dyn_violate",
      hue="behavior",ax=ax2, hue_order=hue_order, linewidth=0.3, palette=hue_color_dict, flierprops=flierprops, zorder=2)


data_frame_freeway["risk_replaced"] = data_frame_freeway["risk"].replace({0.01: 0.0, 0.1 : 1.0, 0.2: 2.0, 0.4: 3.0, 0.6: 4.0, 0.8: 5.0, 1.0: 6.0})
data_frame_rural["risk_replaced"] = data_frame_rural["risk"].replace({0.01: 0.0, 0.1 : 1.0, 0.2: 2.0, 0.4: 3.0, 0.6: 4.0, 0.8: 5.0, 1.0: 6.0})


sns.lineplot(data=data_frame_freeway,
      x="risk_replaced", y="avg_dyn_violate",
      hue="behavior",style="behavior", ax=ax3, hue_order=hue_order, linewidth=3.0, palette=hue_color_dict, style_order=hue_order, ci=99.999999)

sns.lineplot(data=data_frame_rural,
      x="risk_replaced", y="avg_dyn_violate",
      hue="behavior",style="behavior", ax=ax4, hue_order=hue_order, linewidth=3.0, palette=hue_color_dict, style_order=hue_order, ci=99.999999)


angle=45
ax1.set_title("Freeway enter")
ax1.set_ylabel("Observed risk $\\beta^{*}$")
ax1.set_xlabel("")
ax2.set_xlabel("")
ax4.set_xlabel("Specified risk $\\beta$")
ax3.plot([0.0, 1, 2, 3, 4, 5.0, 6], [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0], linestyle='--', color= "darkgrey", linewidth=2.0)
ax4.plot([0.0, 1, 2, 3, 4,  5.0, 6], [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0], linestyle='--', color= "darkgrey", linewidth=2.0)
ax3.text(2.2, 0.27, "Equality", size=7,  rotation=angle, rotation_mode='anchor')
ax4.text(2.2, 0.27, "Equality", size=7,  rotation=angle, rotation_mode='anchor')

#ax2.plot([0.0, 5.0], [0.1, 1.0], linestyle='--', color= "darkgrey", linewidth=1.0)
ax3.set_xlabel("Specified risk $\\beta$")
ax3.xaxis.set_label_coords(0.5, -0.17)
ax4.xaxis.set_label_coords(0.5, -0.17)
ax2.set_ylabel("")
ax4.set_ylabel("")
ax3.set_ylabel("Observed risk $\\beta^{*}$")
ax2.set_title("Left turn")
#ax1.set(ylim=(0.0, 0.8))
#ax1.set(xlim=(0.0, 1.0))

#ax2.set(xlim=(0.1, 1.0))
#ax4.set_yscale("log")
ax3.set_yticks([0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.set_yticks([0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.set_yticklabels(["0.01", "0.1", "0.2", "0.4", "0.6", "0.8", "1.0"])
ax3.set_yticklabels(["0.01", "0.1", "0.2", "0.4", "0.6", "0.8", "1.0"])
ax2.set(ylim=(0.0, 0.8))

ax3.set(ylim=(0.0, 0.5))
ax4.set(ylim=(0.0, 0.5))
ax1.yaxis.set_label_coords(-0.10, 0.6)
ax3.yaxis.set_label_coords(-0.10, 0.45)
plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax4.get_yticklabels(), visible=False)
ax4.tick_params(axis='both', which='major', pad=2.5, length=0)
ax3.tick_params(axis='both', which='major', pad=2.5, length=0)
ax1.tick_params(axis='both', which='major', pad=2.5, length=0)
ax2.tick_params(axis='both', which='major', pad=2.5, length=0)
handles, labels = ax1.get_legend_handles_labels()
labels = [l.replace("Local","") for l in labels]
labels = [l.replace("RCRSBG","RC-RSBG") for l in labels]
ax1.get_legend().remove()
ax2.get_legend().remove()
ax3.get_legend().remove()
ax4.get_legend().remove()
leg = ax3.legend(handles, labels, bbox_to_anchor=(0.15, -0.5, 1.5, .09), handletextpad=0.1,
         loc=7, ncol=4, mode="expand", borderaxespad=0., frameon=False, handlelength=1.0)

plt.show(block=False)
export_image(filename="risk_mean.pdf", fig=plt.gcf(), bbox_inches='tight',
                pad_inches=0)
home_dir = os.path.expanduser("~")
shutil.copy(f"risk_mean.pdf", os.path.join(home_dir, "development/thesis/results/figures"))
