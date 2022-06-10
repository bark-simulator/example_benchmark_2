try:
    import debug_settings
except:
    pass

import os
import sys
import logging
import matplotlib.pyplot as plt
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

logging.info("Running on process with ID: {}".format(os.getpid()))
import bark.core.commons

from load.benchmark_database import BenchmarkDatabase
from serialization.database_serializer import DatabaseSerializer
from bark.benchmark.benchmark_runner_mp import BenchmarkRunnerMP, BenchmarkRunner

from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.viewer.viewer import generatePoseFromState
from bark.runtime.viewer.video_renderer import VideoRenderer

from bark.core.models.dynamic import StateDefinition
from bark.core.models.behavior import BehaviorIDMClassic, BehaviorMacroActionsFromParamServer

from bark.runtime.commons.parameters import ParameterServer
from src.evaluation.bark.analysis.custom_viewer import CustomViewer
from src.evaluation.bark.behavior_configuration.behavior_configs import *
from src.evaluation.bark.behavior_configuration.behavior_configurate import \
        dump_defaults, create_behavior_configs, create_mcts_params, \
        create_benchmark_configs, create_evaluation_configs, get_terminal_criteria

from bark.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import add_config_reader_module
add_config_reader_module("bark_mcts.runtime.scenario.behavior_space_sampling")

log_folder = os.path.abspath(os.path.join(os.getcwd(), "logs"))
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
logging.info("Logging into: {}".format(log_folder))
bark.core.commons.GLogInit(sys.argv[0], log_folder, 0, True, "behavior*=5")

# reduced max steps and scenarios for testing
max_steps = 40
num_scenarios = 10

logging.getLogger().setLevel(logging.INFO)

if not os.path.exists("src"):
  logging.info("changing directory")
  os.chdir("run_benchmark_interact.runfiles/phd")


dbs = DatabaseSerializer(test_scenarios=2, test_world_steps=20, num_serialize_scenarios=num_scenarios)
dbs.process("src/evaluation/bark/database_configuration/database", filter_sets="**/[i]*/free*intent.json")
local_release_filename = dbs.release(version="tmp2")

db = BenchmarkDatabase(database_root=local_release_filename)

evaluators, param_servers_persisted = create_evaluation_configs()
terminal_when = get_terminal_criteria()


param_mappings= [ {"BehaviorUctBase::ConstantActionIndex" : -1,
                  "BehaviorUctBase::Mcts::MaxNumIterations" : 10000,
                  "BehaviorUctBase::Mcts::MaxNumNodes" : 4000,
                  "BehaviorUctBase::Mcts::MaxSearchTime" : 200000000,
                  "BehaviorUctBase::Mcts::NumParallelMcts" : 1,
                  "BehaviorUctBase::Mcts::CostConstrainedStatistic::Kappa" : 10.0,
                  "BehaviorUctBase::Mcts::CostConstrainedStatistic::ActionFilterFactor" : 3.5,
                  "BehaviorUctBase::Mcts::CostConstrainedStatistic::GradientUpdateScaling" : 1.0,
                    "BehaviorUctBase::Mcts::State::EvaluatorParams::EvaluatorStaticSafeDist::LateralSafeDist" : 0.5,
                    "BehaviorUctBase::Mcts::State::EvaluatorParams::EvaluatorStaticSafeDist::LongitudinalSafeDist" : 0.5}
                  ]
                  
# param_mappings = [{
#   "BehaviorUctBase::Mcts::State::EvaluationParameters::AddSafeDist" : False,
#   "BehaviorUctBase::Mcts::MaxSearchTime" : 10000000,
#   "BehaviorUctBase::Mcts::MaxNumIterations" : 10000
# }]

benchmark_configs, param_servers = \
         create_benchmark_configs(db, num_scenarios, 
          [
        "BehaviorConfigRSBG_wo_risk",
       #   "BehaviorConfigRCRSBGLocal",
        #   "BehaviorConfigMDP",
           # "BehaviorConfigCooperative"
         #   "BehaviorConfigRandomMacro",
          # "BehaviorConfigSBGFullInfo"
          ], {
               "rural_left_turn_risk" : "1D_desired_gap_urban.json",
               "freeway_enter" : "1D_desired_gap_urban.json"}, [0.1], [16],
                param_mappings=param_mappings)
benchmark_runner = BenchmarkRunner(benchmark_database = db,
                                    evaluators = evaluators,
                                    terminal_when = terminal_when,
                                   benchmark_configs = benchmark_configs,
                                    num_scenarios=num_scenarios,
                                   log_eval_avg_every = 1,
                                   checkpoint_dir = "checkpoints",
                                    merge_existing = False)
                                  
viewer = CustomViewer(
  params=ParameterServer(),
  center= [100.0, 1.8],
  enforce_x_length=False,
  x_length = 120.0,
  use_world_bounds=False,
  use_wait_for_click=False)
result = benchmark_runner.run(maintain_history=False, viewer=viewer)
#result = benchmark_runner.run_benchmark_config(config_idx=2, viewer=viewer)
result.dump("./scenarios_examples", dump_configs = True, dump_histories=True)