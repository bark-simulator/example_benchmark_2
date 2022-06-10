try:
    import debug_settings
except:
    pass

import os
import math
import logging
logging.basicConfig()

from src.common.plotting import *
from src.common.export_image import export_image 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
      CreateAxisCollectionGrid

from src.evaluation.bark.analysis.custom_video_renderer import CustomVideoRenderer
from bark.benchmark.benchmark_analyzer import BenchmarkAnalyzer
from bark.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import add_config_reader_module
add_config_reader_module("bark_mcts.runtime.scenario.behavior_space_sampling")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("scen_set", help="display a square of a given number",
                    type=str)
parser.add_argument("world_indices", help="display a square of a given number",
                    type=str)
parser.add_argument("scen_idx", help="display a square of a given number",
                    type=int)
parser.add_argument("risk", help="display a square of a given number",
                    type=float)
args = parser.parse_args()

result_file = "histories_rcrsbg_baselines"
filepath = "results/benchmark/rcrsbg"
num_nearest = 4
step_time = 0.2
render_time = 0.2

# scen_type = "freeway_enter"
# result_file = "histories_rcrsbg"
# filepath = "results/benchmark/rcrsbg"
# scenario_idx = 0
# world_indices_to_plot = [2, 10]
# risk = 0.1

scen_type = args.scen_set
result_file = "histories_rcrsbg"
filepath = "results/benchmark/rcrsbg"
scenario_idx = args.scen_idx
world_indices_to_plot = [int(item) for item in args.world_indices.split(',')]
risk = args.risk


matplotlib_latex_conf(pgf_backend=False)

hd_scale = 1.0
resolution = (1920*hd_scale, 1080*hd_scale)
dpi = 300
figure = plt.figure(figsize=(resolution[0] / dpi, resolution[1] / dpi), dpi=dpi)

result = BenchmarkResult.load(os.path.join(filepath, result_file))
analyzer = BenchmarkAnalyzer(benchmark_result=result)

configs_mdp = analyzer.find_configs(criteria = {"out_of_drivable": lambda x : not x, "scen_set" : lambda x: x=="freeway_enter",  \
                                               "behavior": lambda x: x=="MDP", "risk" : lambda x: x==0.4})

configs_coop = analyzer.find_configs(criteria = {"out_of_drivable": lambda x : not x, "scen_set" : lambda x: x=="freeway_enter",  \
                                               "behavior": lambda x: x=="Cooperative", "risk" : lambda x: x==0.4 }, scenarios_as_in_configs=configs_mdp)

configs_rsbg = analyzer.find_configs(criteria = {"out_of_drivable": lambda x : not x, "scen_set" : lambda x: x=="freeway_enter",  \
                                               "behavior": lambda x: x=="RSBG", "risk" : lambda x: x==0.4 }, scenarios_as_in_configs=configs_coop)

configs_rcrsbg = analyzer.find_configs(criteria = {"success": lambda x : x, "scen_set" : lambda x: x=="freeway_enter", \
                                               "behavior": lambda x: x=="RCRSBGLocal", "risk" : lambda x: x==0.4, "step" : lambda x: x>20}, scenarios_as_in_configs=configs_rsbg)

config_idx_lists = analyzer.make_scenarios_congruent(configs_idx_lists = [configs_rcrsbg, configs_rsbg, configs_mdp, configs_coop])

total_confs = []
for l in config_idx_lists:
    total_confs.extend(l) 
result.load_histories(total_confs)
result.load_benchmark_configs(total_confs)
histories = result.get_histories()

results_variations = [("RC-RSBG Planner, $\\SRisklevel=0.4$", config_idx_lists[0]), 
      ("RSBG Planner, $\\SRisklevel=0.4$", config_idx_lists[1]),
      ("MDP Planner, $\\SRisklevel=0.4$", config_idx_lists[2]),
      ("Cooperative Planner, $\\SRisklevel=0.4$", config_idx_lists[3])]

SetMplStyle("traffic", GetMplStyleConfigTrafficVideo())
SetMplStyle("beliefs", GetMplStyleConfigPolicyVideo())
SetMplStyle("policy", GetMplStyleConfigPolicyVideo())

axis_collection = CreateAxisCollectionGrid(figure=figure, vert = 2, hor = 2 ,\
             parts_info_axis = 0, parts_main_axis = 10, parts_belief_axis = 3, parts_policy_axis=7,\
             grid_spec_main = {"wspace" : 0.15, "hspace" : 0.0, \
           "left" : 0.05, "right" : 0.95, "bottom" : 0.05, "top" : 0.95}, split_type="vertical")

params_drawing = ParameterServer(filename="src/evaluation/bark/database_configuration/visualization_params/scenario_draw.json")
viewer = CustomViewer(
  params=ParameterServer(),
  center= [100.0, 1.8],
  enforce_x_length=True,
  x_length = 120.0,
  use_world_bounds=False)

video_renderer = CustomVideoRenderer(world_step_time = render_time)

def clear_figure_content(figure):
  for txt in figure.texts:
    txt.set_visible(False)

agent_color_map = viewer.agent_color_map
for config_idx in config_idx_list:
  scenarios_ended = False
  time_idx = 3
  while not scenarios_ended:
    scenarios_ended = True
    for result_idx, result_variation in enumerate(results_variations):
      params = ParameterServer()
      axis_types = axis_collection.GetAxisType(config_plot_idx=result_idx % 2, \
            time_plot_idx=math.floor(result_idx / 2))

      axis_types.main_axis.cla()

      result_name = result_variation[0]
      current_config = result_variation[1][config_idx] 
      result_history = histories[current_config]
      viewer.axes = axis_types.main_axis
      
      if time_idx >= len(result_history):
        plotted_time_idx = len(result_history) - 1
      else:
        scenarios_ended = False
        plotted_time_idx = time_idx
      scenario = result_history[plotted_time_idx]
      loaded_params = ParameterServer(json=scenario._json_params)
      loaded_params.AppendParamServer(params_drawing, overwrite=True)
      viewer.initialize_params(loaded_params)
      viewer.agent_color_map = agent_color_map
      world = scenario.GetWorldState()
      ego_behavior = world.agents[scenario.eval_agent_ids[0]].behavior_model
      viewer.drawWorld(world=world,
                              eval_agent_ids=scenario.eval_agent_ids, debug_text=False)
      behavior_plan = None
      axis_types.main_axis.text(0.5, 0.65, "$t={:.2f}\,\\text{{s}}$".format(plotted_time_idx*step_time), horizontalalignment='center', size = 5,
        verticalalignment='center', transform=axis_types.main_axis.transAxes)
      # legend
      hor = result_idx % 2
      ver = math.floor(result_idx / 2)
      axis_types.main_axis.text(0.5, 0.8, result_name, horizontalalignment='center', size = 7,
        verticalalignment='center', transform=axis_types.main_axis.transAxes)
    time_idx += 1

    plt.draw()
    video_renderer.DumpFrame(axis_types.main_axis.get_figure())
    clear_figure_content(figure)

video_renderer.export_video(filename="./video")