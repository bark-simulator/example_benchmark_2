try:
    import debug_settings
except:
    pass

import os
import sys
import logging
import matplotlib.pyplot as plt
import math
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
from src.evaluation.bark.behavior_configuration.behavior_configs import *
from src.evaluation.bark.behavior_configuration.behavior_configurate import \
        dump_defaults, create_behavior_configs, create_mcts_params, \
        create_benchmark_configs, create_evaluation_configs, get_terminal_criteria
from src.common.pyhelpers import get_ray_init_config

from bark.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import add_config_reader_module
add_config_reader_module("bark_mcts.runtime.scenario.behavior_space_sampling")

log_folder = os.path.abspath(os.path.join(os.getcwd(), "logs"))
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
logging.info("Logging into: {}".format(log_folder))
bark.core.commons.GLogInit(sys.argv[0], log_folder, 0, True, "behavior*=3")

# reduced max steps and scenarios for testing
num_scenarios = 10


logging.getLogger().setLevel(logging.INFO)

if not os.path.exists("src"):
  logging.info("changing directory")
  os.chdir("run_benchmark_parallel_mcts.runfiles/phd")

database_tmp_dir = None
try:
  database_tmp_dir = os.environ["SLURM_SUBMIT_DIR"]
except:
  logging.info("Using standard tmp database dir /tmp")


dbs = DatabaseSerializer(test_scenarios=2, test_world_steps=20, num_serialize_scenarios=num_scenarios)
dbs.process("src/evaluation/bark/database_configuration/database", filter_sets="**/[c]*/*.json")
local_release_filename = dbs.release(version="tmp2", tmp_dir=database_tmp_dir)

db = BenchmarkDatabase(database_root=local_release_filename, tmp_dir=database_tmp_dir)

def create_param_mappings_increase_parallelization(parallel_mcts_param_list, base_max_iterations):
  param_mappings = []
  for mcts_parallel in parallel_mcts_param_list:
    param_config = {"BehaviorUctBase::Mcts::NumParallelMcts" : mcts_parallel, 
                  "BehaviorUctBase::Mcts::MaxSearchTime" : 200,
                  "BehaviorUctBase::Mcts::MaxNumNodes" : 4000}
    param_mappings.append(param_config)
  return param_mappings

def add_param_variations(param_mapping, variations):
  new_mapping = []
  for mapping in param_mapping:
    for variation_dict in variations:
      mapping_new = mapping.copy()
      for key, param in variation_dict.items():
        mapping_new[key] = param
      new_mapping.append(mapping_new)
  return new_mapping

variations = [  {  "BehaviorUctBase::Mcts::CostConstrainedStatistic::Kappa" : 10.0,
                  "BehaviorUctBase::Mcts::UctStatistic::ExplorationConstant" : 0.7,
                  "BehaviorUctBase::Mcts::CostConstrainedStatistic::ActionFilterFactor" : 3.5,
                  "BehaviorUctBase::Mcts::State::EvaluatorParams::EvaluatorStaticSafeDist::LateralSafeDist" : 0.0,
                 "BehaviorUctBase::Mcts::State::EvaluatorParams::EvaluatorStaticSafeDist::LongitudinalSafeDist" : 0.0,
                 "exploration" : "normal"}]

param_mappings = create_param_mappings_increase_parallelization(
    parallel_mcts_param_list=[1, 4, 16, 64],
    base_max_iterations=1600,
)
param_mappings = add_param_variations(param_mappings, variations)


benchmark_configs, param_servers = \
         create_benchmark_configs(db, num_scenarios, 
          [
           "BehaviorConfigRSBG",
           "BehaviorConfigRCRSBGLocal"
          ], {
               "rural_left_turn_risk" : "1D_desired_gap_urban.json",
               "freeway_enter" : "1D_desired_gap_urban.json"
                }, [0.1], [16],
                param_mappings=param_mappings)

evaluators, param_servers_persisted = create_evaluation_configs(add_mcts_infos=True)
terminal_when = get_terminal_criteria()

benchmark_runner = BenchmarkRunnerMP(benchmark_database = db,
                                    evaluators = evaluators,
                                    terminal_when = terminal_when,
                                    benchmark_configs = benchmark_configs,
                                    num_scenarios=num_scenarios,
                                   glog_init_settings={"vlevel": 0},
                                   log_eval_avg_every = 5,
                                   checkpoint_dir = "checkpoints",
                                    merge_existing = False,
                                    ray_init_args=get_ray_init_config())
benchmark_runner.clear_checkpoint_dir()

result = benchmark_runner.run(maintain_history=False, checkpoint_every=10000000) 

print(result.get_data_frame())
result.dump(os.path.join("./benchmark_results"), dump_histories=False, dump_configs=False)

result_loaded = result.load(os.path.join("./benchmark_results"), load_histories=False, load_configs=False)
data_frame = result.get_data_frame()

data_frame["max_steps"] = data_frame.Terminal.apply(lambda x: "max_steps" in x and (not "collision" in x))
data_frame["success"] = data_frame.Terminal.apply(lambda x: "success" in x and (not "collision" in x) and (not "max_steps" in x))

data_frame["avg_dyn_violate"] = data_frame.safe_dist_dyn / data_frame.step

data_frame.fillna(-1)
#dfg = data_frame.fillna(-1).groupby(["behavior", "scen_set", "update", "risk", "filter", "kappa", "num_hypothesis"]).mean()
dfg = data_frame.fillna(-1).groupby(["behavior", "scen_set", "risk", "num_hypothesis", "BehaviorUctBase::Mcts::NumParallelMcts"]).mean()
print(dfg.to_string())