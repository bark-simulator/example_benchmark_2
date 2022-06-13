import os

try:
    import debug_settings
except:
    pass

from load.benchmark_database import BenchmarkDatabase
from serialization.database_serializer import DatabaseSerializer
from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.viewer.video_renderer import VideoRenderer

from bark.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import add_config_reader_module
add_config_reader_module("bark_mcts.runtime.scenario.behavior_space_sampling")

num_test_scenarios_per_scenario_set = 5
num_steps_per_scenario = 20

params = ParameterServer()
params["Visualization"]["Agents"]["DrawEvalGoals"] = True
params["Visualization"]["Agents"]["Color"]["UseColormapForOtherAgents"] = True
viewer = MPViewer(params=params,
                  use_world_bounds=True)
renderer = VideoRenderer(renderer=viewer, world_step_time=0.2)


dbs = DatabaseSerializer(test_scenarios=num_test_scenarios_per_scenario_set, test_world_steps=num_steps_per_scenario,
                         num_serialize_scenarios=num_test_scenarios_per_scenario_set, visualize_tests=True, viewer=renderer)
test_result = dbs.process("src/evaluation/bark/database_configuration/database", filter_sets="**/*light_dense.json")

renderer.export_video(filename="./database")
