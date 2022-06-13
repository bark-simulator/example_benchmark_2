from collections import defaultdict
import os
import glob

from bark.runtime.commons.parameters import ParameterServer
from bark.benchmark.benchmark_result import BenchmarkConfig
from bark.benchmark.benchmark_runner import EvaluationConfig
from bark.core.world.evaluation import *

from bark_mcts.models.behavior.hypothesis.behavior_space.behavior_space import BehaviorSpace
from src.evaluation.bark.behavior_configuration.behavior_configs import *
from bark.core.models.behavior import *

MCTS_PARAMS_ROOT = "src/evaluation/bark/behavior_configuration/mcts_params"
PARAMS_ALG_DEP = os.path.join(MCTS_PARAMS_ROOT, "algorithm_dependent")
PARAMS_SCEN_DEP = os.path.join(MCTS_PARAMS_ROOT, "scenario_dependent")
COMMON_PARAM_FILE = os.path.join(MCTS_PARAMS_ROOT, "common/common.json")
OVERWRITTEN_PARAM_FILE = os.path.join(MCTS_PARAMS_ROOT, "common/overwritten.json")
BEHAVIOR_SPACE_ROOT = "src/evaluation/bark/behavior_configuration/behavior_spaces"
SCENARIO_TYPES = ["highway_light", "highway_mid"]
MAX_STEPS_FREEWAY_MID = 30
MAX_STEPS_FREEWAY_LIGHT = 20

def dump_defaults(dir):
    params = ParameterServer()
    behavior = BehaviorUCTHypothesis(params, [])
    params.Save(os.path.join(dir, "uct_hypothesis.json"))

    params = ParameterServer()
    behavior = BehaviorUCTRiskConstraint(params, [], None)
    params.Save(os.path.join(dir, "uct_risk_constraint.json"))

    params = ParameterServer()
    behavior = BehaviorUCTCooperative(params)
    params.Save(os.path.join(dir, "uct_risk_cooperative.json"))

    params = ParameterServer()
    behavior_space = BehaviorSpace(params)
    params.Save(os.path.join(dir, "behavior_space.json"))

def create_cover_and_hypothesis_params(behavior_space_params, hypothesis_splits):
    behavior_space = BehaviorSpace(behavior_space_params)
    [cover_hypothesis], cover_params = behavior_space.create_cover_hypothesis()
    hypothesis_set_collections = \
                behavior_space.create_multiple_hypothesis_sets(hypothesis_splits)
    return cover_hypothesis, hypothesis_set_collections, cover_params

def create_behavior_configs(mcts_params,
                        behavior_space_params,
                        hypothesis_splits,
                        behavior_config_types,
                        param_descriptions=None,
                        hypothesis_set_collections=None,
                        cover_hypothesis=None,
                        intent_hypothesis=None,
                        imitation_agents=None):
    params_persisted = []

    if behavior_space_params:
      cover_hypothesis, hypothesis_set_collections, params_cover = \
          create_cover_and_hypothesis_params(behavior_space_params, hypothesis_splits)
      params_persisted.append(params_persisted)

    param_descriptions = param_descriptions or {}
    behavior_configs = []
    if not isinstance(mcts_params, list):
        mcts_params = [({}, mcts_params)]
    for mcts_param_pair in mcts_params:
        mcts_param_server = mcts_param_pair[1]
        param_descriptions_mcts = mcts_param_pair[0] 
        for behavior_config_type in behavior_config_types:
            tmp_param_descriptions = {**param_descriptions_mcts, **param_descriptions}
            config = None
            if behavior_config_type == "BehaviorConfigRMDP" or behavior_config_type == "BehaviorConfigMDP" or \
                behavior_config_type == "BehaviorConfigRSBGFullInfo" or behavior_config_type == "BehaviorConfigSBGFullInfo" \
                or behavior_config_type == "BehaviorConfigRCSBGLocalFullInfo" or behavior_config_type == "BehaviorConfigRCRSBGLocalFullInfo" or \
                  behavior_config_type == "BehaviorConfigRSBGFullInfo_wo_risk" or behavior_config_type == "BehaviorConfigSBGFullInfo_wo_risk":
                config = eval("{}(cover_hypothesis, mcts_param_server, param_descriptions=tmp_param_descriptions)".format(behavior_config_type))
                behavior_configs.append(config)
            elif behavior_config_type == "BehaviorConfigCooperative":
                config = eval("{}(mcts_param_server, param_descriptions=tmp_param_descriptions)".format(behavior_config_type))
                behavior_configs.append(config)
            elif behavior_config_type == "BehaviorConfigIntentRSBG" or behavior_config_type=="BehaviorConfigIntentRSBG_wo_risk":
                config = eval("{}(intent_hypothesis, mcts_param_server, param_descriptions=tmp_param_descriptions)".format(behavior_config_type))
                behavior_configs.append(config)
                params_persisted
            elif behavior_config_type == "BehaviorConfigSBG" or behavior_config_type == "BehaviorConfigRSBG" or \
                behavior_config_type == "BehaviorConfigRCRSBGLocal" or \
                behavior_config_type == "BehaviorConfigRSBG_wo_risk" or \
                behavior_config_type == "BehaviorConfigSBG_wo_risk"  or \
                behavior_config_type == "BehaviorConfigRSBG_wo_risk_wh":
                for num_hypothesis, (hypothesis_set, params_sets) in hypothesis_set_collections.items():
                    tmp_param_descriptions_hyp = {"num_hypothesis": len(hypothesis_set), **tmp_param_descriptions}
                    config = eval("{}(hypothesis_set, mcts_param_server, param_descriptions=tmp_param_descriptions_hyp)".format(behavior_config_type))
                    behavior_configs.append(config)
                    params_persisted.extend(params_sets)
            elif behavior_config_type == "BehaviorConfigRandomMacro":
              config = eval("{}(mcts_param_server, param_descriptions=tmp_param_descriptions)".format(behavior_config_type))
              behavior_configs.append(config)
            elif behavior_config_type == "BehaviorConfigRCRSBGNHeuristic":
              for num_hypothesis, (hypothesis_set, params_sets) in hypothesis_set_collections.items():
                tmp_param_descriptions_hyp = {"num_hypothesis": len(hypothesis_set), **tmp_param_descriptions}
                for agent_name, imitation_agent in imitation_agents.items():
                  tmp_param_descriptions_nheuristic = {"imitation_agent" : agent_name, **tmp_param_descriptions_hyp}
                  config = eval("{}(hypothesis_set, mcts_param_server, imitation_agent=imitation_agent,\
                                                   param_descriptions=tmp_param_descriptions_nheuristic)".format(behavior_config_type))
                  behavior_configs.append(config)
                  params_persisted.extend(params_sets)
            else:
                raise ValueError("Unknown config type: {}.".format(behavior_config_type))

    return behavior_configs, params_persisted

def adjust_risk_spec(mcts_params, risk_spec, algorithm_type):
  search_params = mcts_params["BehaviorUctBase"]["Mcts"]
  state_params = mcts_params["BehaviorUctBase"]["Mcts"]["State"]

  max_depth = mcts_params["BehaviorUctBase"]["Mcts"]["MaxSearchDepth"]
  predict_k = state_params["PredictionK"]
  predict_alpha = state_params["PredictionAlpha"]

  if "risk" in algorithm_type:
    mcts_params["BehaviorUctRiskConstraint"]["DefaultAvailableRisk"] = [risk_spec, 0.0] # safe dist / collision
    state_params["GoalReward"] = 1.0
    state_params["CollisionReward"] = 0.0
    state_params["SafeDistViolatedReward"] = 0.0
    state_params["DrivableCollisionReward"] = 0.0

    state_params["GoalCost"] = 0.0
    state_params["CollisionCost"] = 1.0
    state_params["SafeDistViolatedCost"] = 1.0
  elif "cooperative" in algorithm_type or "sbg" in algorithm_type or "mdp" in algorithm_type:
    state_params["GoalReward"] = 0.1
    planning_time = 0.0
    for depth in range(0, max_depth): 
      planning_time += predict_k*pow(depth, predict_alpha)
    # if I violate in 10 steps, 10*risk_spec I get the same reward as for reaching the goal
    safe_violated_step_cost = state_params["GoalReward"]/(risk_spec*planning_time)
    state_params["CollisionReward"] = -1.0
    state_params["SafeDistViolatedReward"] = -safe_violated_step_cost if risk_spec >= 0.0 else 0.0

    state_params["GoalCost"] = - state_params["GoalReward"]
    state_params["CollisionCost"] = - state_params["CollisionReward"]
    state_params["SafeDistViolatedCost"] = - state_params["SafeDistViolatedReward"]
    state_params["DrivableCollisionCost"] = - state_params["DrivableCollisionReward"]

  if "cooperative" in algorithm_type:
    search_params["ReturnUpperBound"] = state_params["GoalReward"]*5
    search_params["ReturnLowerBound"] = state_params["CollisionReward"] 
    search_params["LowerCostBound"] = 0.0
    search_params["UpperCostBound"] = 0.0

  if "sbg" in algorithm_type or "mdp" in algorithm_type:
    search_params["ReturnUpperBound"] = state_params["GoalReward"]
    search_params["ReturnLowerBound"] = state_params["CollisionReward"] + state_params["DrivableCollisionReward"]
    search_params["LowerCostBound"] = - search_params["ReturnUpperBound"]
    search_params["UpperCostBound"] = - search_params["ReturnLowerBound"]

  if "risk" in algorithm_type:
    search_params["ReturnUpperBound"] = 1.0
    search_params["ReturnLowerBound"] = 0.0
    search_params["LowerCostBound"] = 0.0
    search_params["UpperCostBound"] = 1.0
  
  if "risk" in algorithm_type or "sbg" in algorithm_type:
    state_params["CooperationFactor"] = -1000.0


def create_mcts_params(algorithm_type, risk_spec = None, scenario_type = None):
  if "mdp" in algorithm_type: algorithm_type = "sbg"
  algorithm_param_file = glob.glob(os.path.join(PARAMS_ALG_DEP, "*{}*.json".format(algorithm_type)))[0]
  scenario_param_file = os.path.join(PARAMS_SCEN_DEP, "default.json")
  if scenario_type:
     scenario_param_file = glob.glob(os.path.join(PARAMS_SCEN_DEP, "*{}*.json".format(scenario_type)))[0]

  mcts_params = ParameterServer(filename = COMMON_PARAM_FILE, log_if_default=True)
  mcts_params.AppendParamServer(ParameterServer(filename=OVERWRITTEN_PARAM_FILE), overwrite=False)
  mcts_params.AppendParamServer(ParameterServer(filename=algorithm_param_file), overwrite=True)
  mcts_params.AppendParamServer(ParameterServer(filename=scenario_param_file), overwrite=True)

  if risk_spec:
    adjust_risk_spec(mcts_params, risk_spec, algorithm_type)

  return mcts_params

def create_intent_set(behavior_space_name):
  without_extension = os.path.splitext(behavior_space_name)[0]
  intent_behavior_space_param_files = glob.glob(os.path.join(BEHAVIOR_SPACE_ROOT, f"{without_extension}_intent*"))
  hypothesis_set = []
  params = []
  for intent_behavior_space in intent_behavior_space_param_files:
    behavior_space_params_param_file = ParameterServer(filename=intent_behavior_space)
    behavior_space = BehaviorSpace(behavior_space_params_param_file)
    [cover_hypothesis], cover_params = behavior_space.create_cover_hypothesis()
    hypothesis_set.append(cover_hypothesis)
    params.append(cover_params)
  return hypothesis_set, params


def create_benchmark_configs(database, num_scenarios, behavior_config_types, behavior_space_mapping,
                             risk_spec, hypothesis_splits, param_mappings=None, imitation_agents=None):
  algorithm_types = ["cooperative", "risk", "sbg", "mdp"]
  benchmark_configs = []
  param_servers = []
  benchmark_config_idx = 0

  if not isinstance(risk_spec, list):
    risk_spec = [risk_spec]
  mcts_algorithm_params = {}
  for algorithm_type in algorithm_types:
    mcts_algorithm_params[algorithm_type] = {}
    for scenario_type in SCENARIO_TYPES:
      mcts_algorithm_params[algorithm_type][scenario_type] = {}
      for risk in risk_spec:
        mcts_algorithm_params[algorithm_type][scenario_type][risk] = \
           create_mcts_params(algorithm_type, risk, scenario_type)

  hypothesis_pre_calc = {}
  intents_pre_calc = {}
  for scenario_type, behavior_space_name in behavior_space_mapping.items():
    behavior_space_param_file = os.path.join(BEHAVIOR_SPACE_ROOT, behavior_space_name)
    behavior_space_params = ParameterServer(filename=behavior_space_param_file)
    hypothesis_pre_calc[scenario_type] = create_cover_and_hypothesis_params(behavior_space_params, hypothesis_splits)
    intents_pre_calc[scenario_type] = create_intent_set(behavior_space_name)
    param_servers.extend(hypothesis_pre_calc[scenario_type][2])
    param_servers.extend(intents_pre_calc[scenario_type][1])
  for scenario_generator, scenario_set_name, scenario_set_param_desc in database:
    scenario_types_matched = [ scenario_type for scenario_type in SCENARIO_TYPES if scenario_type in scenario_set_name.lower()]
    if len(scenario_types_matched) == 0 or len(scenario_types_matched) > 1:
      raise ValueError("Scenario type matching failed: {} matched to scenario set {}".format(
                          scenario_types_matched, scenario_set_name))
    for scenario, scenario_idx in scenario_generator:
      if num_scenarios and scenario_idx >= num_scenarios:
        break
      for behavior_config_type in behavior_config_types:
        algorithm_types_matched = [ algorithm_type for algorithm_type in algorithm_types if algorithm_type in eval("{}.algorithm_type()".format(behavior_config_type))]
        if len(algorithm_types_matched) == 0 or len(algorithm_types_matched) > 1:
          raise ValueError("Algorithm type matching failed: {} matched to scenario set {}".format(
                              algorithm_types_matched, behavior_config_type))
        for risk in risk_spec:
          mcts_params = mcts_algorithm_params[algorithm_types_matched[0]][scenario_types_matched[0]][risk]


          if not param_mappings:
            param_mappings = [{}]
          for param_mapping in param_mappings:
            current_mcts_params = mcts_params.clone()
            for param, value in param_mapping.items():
              current_mcts_params[param] = value 
            behavior_configs, params_persisted = create_behavior_configs([({**param_mapping, "risk" : risk}, current_mcts_params)], None, None,
                  [behavior_config_type], None, hypothesis_set_collections=hypothesis_pre_calc[scenario_types_matched[0]][1], \
                       cover_hypothesis=hypothesis_pre_calc[scenario_type][0], intent_hypothesis=intents_pre_calc[scenario_type][0],
                      imitation_agents=imitation_agents)

            for behavior_config in behavior_configs: 
              for scenario, scenario_idx in scenario_generator:
                if num_scenarios and scenario_idx >= num_scenarios:
                  break
                benchmark_config = \
                    BenchmarkConfig(
                        len(benchmark_configs),
                        behavior_config,
                        scenario,
                        scenario_idx,
                        scenario_set_name,
                        scenario_set_param_desc
                    )
                benchmark_configs.append(benchmark_config)
              param_servers.extend(params_persisted)
  return benchmark_configs, param_servers


class EvaluatorNumIterations(BaseEvaluator):
  def __init__(self):
    super(EvaluatorNumIterations, self).__init__()
    self.iterations_list = []

  def SetAgentId(self, agent_id):
    self._agent_id = agent_id

  def Evaluate(self, observed_world):
    if not self._agent_id in observed_world.agents:
      return 0
    ego_behavior = observed_world.agents[self._agent_id].behavior_model
    if ego_behavior.last_num_iterations != 0:
      self.iterations_list.append(ego_behavior.last_num_iterations)
      return sum(self.iterations_list) / len(self.iterations_list)
    else:
      return 0

  def __setstate__(self, l):
    self.iterations_list = l

  def __getstate__(self):
    return self.iterations_list

class EvaluatorNumNodes(BaseEvaluator):
  def __init__(self):
    super(EvaluatorNumNodes, self).__init__()
    self.nodes_list = []

  def SetAgentId(self, agent_id):
    self._agent_id = agent_id

  def Evaluate(self, observed_world):
    if not self._agent_id in observed_world.agents:
      return 0
    ego_behavior = observed_world.agents[self._agent_id].behavior_model
    if ego_behavior.last_num_iterations != 0:
      self.nodes_list.append(ego_behavior.last_num_nodes)
      return sum(self.nodes_list) / len(self.nodes_list)
    else:
      return 0

  def __setstate__(self, l):
    self.nodes_list = l

  def __getstate__(self):
    return self.nodes_list



EvaluationConfig.AddEvaluationModule("src.evaluation.bark.behavior_configuration.behavior_configurate")

def create_evaluation_configs(add_mcts_infos=False):
  default_config = {"success" : "EvaluatorGoalReached", "collision_other" : "EvaluatorCollisionEgoAgent",
        "max_steps": "EvaluatorStepCount", "out_of_drivable" : "EvaluatorDrivableArea",
        "planning_time" : "EvaluatorPlanningTime"}
  if add_mcts_infos:
    default_config.update({"num_nodes" : EvaluatorNumNodes(),"num_iterations" : EvaluatorNumIterations()} )
  evaluation_config = EvaluationConfig(default_config)
  param_servers_persisted = []
  default_scenario_params = ParameterServer(filename=COMMON_PARAM_FILE, log_if_default=True)
  for scenario_type in SCENARIO_TYPES:
    params_evaluators = default_scenario_params.clone()
    scenario_param_file = glob.glob(os.path.join(PARAMS_SCEN_DEP, "*{}*.json".format(scenario_type)))[0]
    params_evaluators.AppendParamServer(ParameterServer(filename=scenario_param_file, log_if_default=True))
    params_evaluators = params_evaluators["BehaviorUctBase"]["Mcts"]["State"]["EvaluatorParams"]
    scenario_eval_config = {**default_config,
                      "out_of_drivable" : {"type" : "EvaluatorSafeDistDrivableArea", "params" : params_evaluators},
                      "safe_dist_stat" : {"type" : "EvaluatorStaticSafeDist", "params" : params_evaluators},
                      "safe_dist_dyn" : {"type" : "EvaluatorDynamicSafeDist", "params" : params_evaluators} }
    evaluation_config.AddEvaluatorConfig(scenario_eval_config, scenario_type)
    param_servers_persisted.append(params_evaluators)
  return evaluation_config, param_servers_persisted


def get_terminal_criteria(max_steps=None):
  return {"highway_mid" : {"collision_other" : lambda x: x,"max_steps": lambda x : (x> max_steps) if max_steps else (x > MAX_STEPS_FREEWAY_MID),  "success" : lambda x: x},
          "highway_light" : {"collision_other" : lambda x: x,"max_steps": lambda x : (x> max_steps) if max_steps else (x> MAX_STEPS_FREEWAY_LIGHT),  "success" : lambda x: x}}
  

