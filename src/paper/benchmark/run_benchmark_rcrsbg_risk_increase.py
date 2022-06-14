try:
    import debug_settings
except:
    pass

from itertools import filterfalse
import os
import sys
import logging
import matplotlib.pyplot as plt
import pandas as pd
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
import math

logging.info("Running on process with ID: {}".format(os.getpid()))
import bark.core.commons

from load.benchmark_database import BenchmarkDatabase
from serialization.database_serializer import DatabaseSerializer
from bark.benchmark.benchmark_runner_mp import BenchmarkRunnerMP, BenchmarkRunner, BenchmarkResult

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
from bark.benchmark.benchmark_analyzer import BenchmarkAnalyzer


from bark.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import add_config_reader_module
add_config_reader_module("bark_mcts.runtime.scenario.behavior_space_sampling")

log_folder = os.path.abspath(os.path.join(os.getcwd(), "logs"))
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
logging.info("Logging into: {}".format(log_folder))
bark.core.commons.GLogInit(sys.argv[0], log_folder, 0, True, "behavior*=3")

# reduced max steps and scenarios for testing
num_scenarios = 200
hypotheses_split = [16]
risk_levels = [0.01, 0.1, 0.2, 0.4, 0.8, 1.0]
variations_kappa = [10.0]   
variations_filter = [3.5]

logging.getLogger().setLevel(logging.INFO)

if not os.path.exists("src"):
  logging.info("changing directory")
  os.chdir("run_benchmark_rcrsbg_params.runfiles/phd")

database_tmp_dir = None
try:
  database_tmp_dir = os.environ["SLURM_SUBMIT_DIR"]
except:
  logging.info("Using standard tmp database dir /tmp")


dbs = DatabaseSerializer(test_scenarios=2, test_world_steps=20, num_serialize_scenarios=num_scenarios)
dbs.process("src/evaluation/bark/database_configuration/database", filter_sets="**/*mid_dense.json")
local_release_filename = dbs.release(version="tmp2", tmp_dir=database_tmp_dir)
db = BenchmarkDatabase(database_root=local_release_filename, tmp_dir=database_tmp_dir)


def add_param_variations(param_mapping, variations_kappa, variations_filter):
  new_mapping = []
  for mapping in param_mapping:
    for variation_kappa in variations_kappa:
      for variation_filter in variations_filter:
        mapping_new = mapping.copy()
        mapping_new["BehaviorUctBase::Mcts::CostConstrainedStatistic::Kappa"] = variation_kappa
        mapping_new["BehaviorUctBase::Mcts::CostConstrainedStatistic::ActionFilterFactor"] = variation_filter
        new_mapping.append(mapping_new)
  return new_mapping

param_mappings = [ {"BehaviorUctBase::Mcts::MaxNumIterations" : 20000,
                    "BehaviorUctBase::Mcts::MaxNumNodes" : 4000,
                    "BehaviorUctBase::Mcts::State::EvaluatorParams::EvaluatorStaticSafeDist::LateralSafeDist" : 0.0,
                    "BehaviorUctBase::Mcts::State::EvaluatorParams::EvaluatorStaticSafeDist::LongitudinalSafeDist" : 0.0}]
param_mappings = add_param_variations(param_mappings, variations_kappa = variations_kappa, variations_filter=variations_filter)
benchmark_configs, param_servers = \
         create_benchmark_configs(db, num_scenarios, 
          [
         "BehaviorConfigRCRSBGLocal",
      #   "BehaviorConfigRCRSBGLocalFullInfo"
          ],{
               "highway_light" : "1D_desired_gap_urban.json",
               "highway_mid" : "1D_desired_gap_urban.json"}, risk_levels, hypotheses_split ,
                param_mappings=param_mappings)

evaluators, param_servers_persisted = create_evaluation_configs()
terminal_when = get_terminal_criteria()

if "slurm_num_cpus" in os.environ:
    num_cpus = int(os.environ["slurm_num_cpus"])
    memory = int(os.environ["slurm_memory"])*1000*1024*1024
    logging.info("Cpus={} and memory={} based on environment variables.".format(num_cpus, memory))
else:
    memory = 40*1000*1024*1024 # 32gb
    num_cpus=10

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
result = benchmark_runner.run(maintain_history=False, checkpoint_every=1000000) 

print(result.get_data_frame())
result.dump(os.path.join("./benchmark_results"), dump_histories=False, dump_configs=False)

result_loaded = result.load(os.path.join("./benchmark_results"), load_histories=False, load_configs=False)
data_frame = result.get_data_frame()

data_frame["max_steps"] = data_frame.Terminal.apply(lambda x: "max_steps" in x and (not "collision" in x))
data_frame["success"] = data_frame.Terminal.apply(lambda x: "success" in x and (not "collision" in x) and (not "max_steps" in x))

data_frame["safe_violate"] = data_frame.apply(lambda x: (x.safe_dist_dyn > 0) or (x.safe_dist_stat > 0), axis=1)
data_frame["avg_dyn_violate"] = data_frame.safe_dist_dyn / data_frame.step

data_frame.fillna(-1)
#dfg = data_frame.fillna(-1).groupby(["behavior", "scen_set", "update", "risk", "filter", "kappa", "num_hypothesis"]).mean()
dfg = data_frame.fillna(-1).groupby(["behavior", "scen_set", "risk", "num_hypothesis"]).mean()
print(dfg.to_string())


