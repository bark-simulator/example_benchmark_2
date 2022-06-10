
try:
    import debug_settings
except:
    pass


import unittest
import pickle
import os

from bark.runtime.commons.parameters import ParameterServer
from src.evaluation.bark.behavior_configuration.behavior_configs import *
from src.evaluation.bark.behavior_configuration.behavior_configurate import \
        dump_defaults, create_behavior_configs, create_mcts_params, create_benchmark_configs, \
        COMMON_PARAM_FILE, OVERWRITTEN_PARAM_FILE, PARAMS_ALG_DEP, PARAMS_SCEN_DEP

from load.benchmark_database import BenchmarkDatabase
from serialization.database_serializer import DatabaseSerializer
from bark.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import add_config_reader_module
add_config_reader_module("bark_mcts.runtime.scenario.behavior_space_sampling")


def pickle_unpickle(object):
    return pickle.loads(
      pickle.dumps(object))

class MctsConfigTests(unittest.TestCase):
  def test_dump_defaults(self):
    dump_defaults("./")

  def test_create_behavior_configs(self):
    mcts_params = ParameterServer()
    behavior_space_params = ParameterServer(filename="src/evaluation/bark/behavior_configuration" \
            "/behavior_spaces/1D_desired_gap_urban.json", log_if_default=False)
    behavior_configs, params = create_behavior_configs(
        mcts_params,
        behavior_space_params,
        hypothesis_splits=[2, 4],
        behavior_config_types=[
            "BehaviorConfigRSBG",
            "BehaviorConfigRMDP",
            "BehaviorConfigMDP",
            "BehaviorConfigSBG",
            "BehaviorConfigRSBGFullInfo",
            "BehaviorConfigSBGFullInfo_wo_risk",
            "BehaviorConfigRCRSBGLocal",
            "BehaviorConfigCooperative"
        ])
    self.assertEqual(len(behavior_configs), 2*3+5)

  def test_create_mcts_params(self):
    mcts_params = create_mcts_params("cooperative", risk_spec=.012323, scenario_type="freeway_enter")
    commons_params = ParameterServer(filename=COMMON_PARAM_FILE)
    freeway_params = ParameterServer(filename=os.path.join(PARAMS_SCEN_DEP, "freeway_enter.json"))
    self.assertEqual(mcts_params["BehaviorUctBase"]["Mcts"]["MaxNumIterations"], \
            commons_params["BehaviorUctBase"]["Mcts"]["MaxNumIterations"])
    self.assertEqual(mcts_params["BehaviorUctBase"]["Mcts"]["State"]["EvaluatorParams"]["EvaluatorDynamicSafeDistLong"]["MaxOtherDecceleration"], \
            commons_params["BehaviorUctBase"]["Mcts"]["State"]["EvaluatorParams"]["EvaluatorDynamicSafeDistLong"]["MaxOtherDecceleration"])
    self.assertEqual(mcts_params["BehaviorUctBase"]["Mcts"]["State"]["EvaluatorParams"]["EvaluatorStaticSafeDist"]["LateralSafeDist"], \
            freeway_params["BehaviorUctBase"]["Mcts"]["State"]["EvaluatorParams"]["EvaluatorStaticSafeDist"]["LateralSafeDist"])
    self.assertEqual(mcts_params["BehaviorUctBase"]["Mcts"]["State"]["CollisionReward"], - 1.0)
    self.assertEqual(mcts_params["BehaviorUctBase"]["Mcts"]["State"]["GoalReward"], .012323)
    mcts_params = create_mcts_params("rsbg", risk_spec=0.23534445, scenario_type="rural_left_turn")
    self.assertEqual(mcts_params["BehaviorUctBase"]["Mcts"]["State"]["CollisionReward"], - 1.0)
    self.assertEqual(mcts_params["BehaviorUctBase"]["Mcts"]["State"]["GoalReward"], 0.23534445)
    self.assertEqual(mcts_params["BehaviorUctBase"]["Mcts"]["State"]["GoalCost"], -0.23534445)
    mcts_params = create_mcts_params("risk", risk_spec=0.4323434, scenario_type="overtaking_decision")
    self.assertEqual(mcts_params["BehaviorUctBase"]["Mcts"]["State"]["CollisionReward"], 0.0)
    self.assertEqual(mcts_params["BehaviorUctBase"]["Mcts"]["State"]["CollisionCost"], 1.0)
    self.assertEqual(mcts_params["BehaviorUctBase"]["Mcts"]["State"]["SafeDistViolatedCost"], 1.0)
    self.assertEqual(mcts_params["BehaviorUctRiskConstraint"]["DefaultAvailableRisk"], 0.4323434)

  def test_create_benchmark_configs(self):
    cwd = os.getcwd()
    num_scenarios = 4
    dbs = DatabaseSerializer(test_scenarios=2, test_world_steps=20, num_serialize_scenarios=num_scenarios)
    dbs.process("src/evaluation/bark/database_configuration/database", filter_sets="**/**[nr]/*.json")
    local_release_filename = dbs.release(version="tmp2")
    db = BenchmarkDatabase(database_root=local_release_filename)
    
    benchmark_configs, param_servers = \
         create_benchmark_configs(db, num_scenarios, 
          [
            "BehaviorConfigSBGFullInfo_wo_risk",
            "BehaviorConfigRSBG",
            "BehaviorConfigSBG",
            "BehaviorConfigRMDP",
            "BehaviorConfigRCRSBGLocal",
            "BehaviorConfigCooperative"
          ], {"highway" : "1D_desired_gap_no_prior.json"}, 0.1, [32])

    def check_equality_algorithm_params(self, behavior, algorithm_param_file):
        result, unequal_params = behavior.params.HasEqualParamsAs(
        ParameterServer(filename=algorithm_param_file))
        if result:
            print(unequal_params.ConvertToDict())
        self.assertFalse(result)
    risk_count = 0
    coop_count = 0
    sbg_count = 0
    for benchmark_config in benchmark_configs:
        behavior = benchmark_config.behavior_config.behavior
        check_equality_algorithm_params(self, behavior, COMMON_PARAM_FILE)
        if benchmark_config.behavior_config.algorithm_type() == "cooperative":
            check_equality_algorithm_params(self, behavior, os.path.join(PARAMS_ALG_DEP, "uct_cooperative.json"))
            coop_count +=1
        elif benchmark_config.behavior_config.algorithm_type() == "sbg":
            check_equality_algorithm_params(self, behavior, os.path.join(PARAMS_ALG_DEP, "uct_rsbg.json"))
            sbg_count += 1
        elif benchmark_config.behavior_config.algorithm_type() == "risk":
            check_equality_algorithm_params(self, behavior, os.path.join(PARAMS_ALG_DEP, "uct_risk.json"))
            risk_count += 1
        else:
            raise ValueError("Unknown algorithm type {}".format(  
                benchmark_config.behavior_config.algorithm_type))

        if "overtaking" in benchmark_config.scenario_set_name:
            check_equality_algorithm_params(self, behavior, os.path.join(PARAMS_SCEN_DEP, "overtaking_decision.json"))
        elif "rural" in benchmark_config.scenario_set_name:
            check_equality_algorithm_params(self, behavior, os.path.join(PARAMS_SCEN_DEP, "rural_left_turn.json"))
        elif "urban" in benchmark_config.scenario_set_name:
            check_equality_algorithm_params(self, behavior, os.path.join(PARAMS_SCEN_DEP, "urban_left_turn.json"))
        elif "freeway" in benchmark_config.scenario_set_name:
            check_equality_algorithm_params(self, behavior, os.path.join(PARAMS_SCEN_DEP, "freeway_enter.json"))
        else:
            raise ValueError("Unknown scenario type")
    self.assertEqual(len(benchmark_configs), num_scenarios*4*6 )
    self.assertEqual(risk_count, num_scenarios*4*1 )
    self.assertEqual(sbg_count, num_scenarios*4*4 )
    self.assertEqual(coop_count, num_scenarios*4*1 )
    os.chdir(cwd)
    

if __name__ == '__main__':
  unittest.main()