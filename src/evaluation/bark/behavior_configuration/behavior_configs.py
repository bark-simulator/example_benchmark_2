


from bark_mcts.models.behavior.hypothesis.behavior_space.behavior_space import BehaviorSpace
from bark.benchmark.benchmark_runner import BehaviorConfig

from bark.core.models.behavior import *

# Provides configs for K, in a single behavior space (same hypothesis definitions)
# MDP, RMDP, SBG (2...K), RSBG (2...K), SBGFullInfo, RSBGFullInfo
#

class BehaviorConfigRSBG(BehaviorConfig):
    def __init__(self, hypothesis_set, mcts_params, param_descriptions=None):
        behavior_name="RSBG"
        self.local_mcts_params = mcts_params.clone()
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["CostBasedActionSelection"] = True
        self.local_mcts_params["BehaviorUctHypothesis"]["PredictionSettings"]["UseTrueBehaviorsAsHypothesis"] = False
        behavior = BehaviorUCTHypothesis(self.local_mcts_params, hypothesis_set)
        super(BehaviorConfigRSBG, self).__init__(behavior_name=behavior_name,
                                          behavior=behavior,
                                          param_descriptions=param_descriptions)
    @staticmethod
    def algorithm_type():
      return "sbg"

class BehaviorConfigRSBG_wo_risk(BehaviorConfig):
    def __init__(self, hypothesis_set, mcts_params, param_descriptions=None):
        behavior_name="RSBG_wo_risk"
        self.local_mcts_params = mcts_params.clone()
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["CostBasedActionSelection"] = True
        self.local_mcts_params["BehaviorUctHypothesis"]["PredictionSettings"]["UseTrueBehaviorsAsHypothesis"] = False
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["State"]["EvaluationParameters"]["AddSafeDist"] = False
        behavior = BehaviorUCTHypothesis(self.local_mcts_params, hypothesis_set)
        super(BehaviorConfigRSBG_wo_risk, self).__init__(behavior_name=behavior_name,
                                          behavior=behavior,
                                          param_descriptions=param_descriptions)
    @staticmethod
    def algorithm_type():
      return "sbg"

class BehaviorConfigRSBG_wo_risk_wh(BehaviorConfig):
    def __init__(self, hypothesis_set, mcts_params, param_descriptions=None):
        behavior_name="RSBG_wo_risk_wh"
        self.local_mcts_params = mcts_params.clone()
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["CostBasedActionSelection"] = True
        self.local_mcts_params["BehaviorUctHypothesis"]["PredictionSettings"]["UseTrueBehaviorsAsHypothesis"] = False
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["State"]["EvaluationParameters"]["AddSafeDist"] = False
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["ProgressiveWidening"]["HypothesisBased"] = True
        behavior = BehaviorUCTHypothesis(self.local_mcts_params, hypothesis_set)
        super(BehaviorConfigRSBG_wo_risk_wh, self).__init__(behavior_name=behavior_name,
                                          behavior=behavior,
                                          param_descriptions=param_descriptions)
    @staticmethod
    def algorithm_type():
      return "sbg"

class BehaviorConfigRMDP(BehaviorConfig):
    def __init__(self, hypothesis, mcts_params, param_descriptions=None):
        behavior_name="RMDP"
        self.local_mcts_params = mcts_params.clone()
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["CostBasedActionSelection"] = True
        self.local_mcts_params["BehaviorUctHypothesis"]["PredictionSettings"]["UseTrueBehaviorsAsHypothesis"] = False
        behavior = BehaviorUCTHypothesis(self.local_mcts_params, [hypothesis])
        super(BehaviorConfigRMDP, self).__init__(behavior_name=behavior_name,
                                          behavior=behavior,
                                          param_descriptions=param_descriptions)
    @staticmethod
    def algorithm_type():
      return "sbg"

class BehaviorConfigMDP(BehaviorConfig):
    def __init__(self, hypothesis, mcts_params, param_descriptions=None):
        behavior_name="MDP"
        self.local_mcts_params = mcts_params.clone()
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["CostBasedActionSelection"] = False
        self.local_mcts_params["BehaviorUctHypothesis"]["PredictionSettings"]["UseTrueBehaviorsAsHypothesis"] = False
        behavior = BehaviorUCTHypothesis(self.local_mcts_params, [hypothesis])
        super(BehaviorConfigMDP, self).__init__(behavior_name=behavior_name,
                                          behavior=behavior,
                                          param_descriptions=param_descriptions)
    @staticmethod
    def algorithm_type():
      return "sbg"

class BehaviorConfigSBG(BehaviorConfig):
    def __init__(self, hypothesis_set, mcts_params, param_descriptions=None):
        behavior_name="SBG"
        self.local_mcts_params = mcts_params.clone()
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["CostBasedActionSelection"] = False
        self.local_mcts_params["BehaviorUctHypothesis"]["PredictionSettings"]["UseTrueBehaviorsAsHypothesis"] = False
        behavior = BehaviorUCTHypothesis(self.local_mcts_params, hypothesis_set)
        super(BehaviorConfigSBG, self).__init__(behavior_name=behavior_name,
                                          behavior=behavior,
                                          param_descriptions=param_descriptions)
    @staticmethod
    def algorithm_type():
      return "sbg"

class BehaviorConfigSBG_wo_risk(BehaviorConfig):
    def __init__(self, hypothesis_set, mcts_params, param_descriptions=None):
        behavior_name="SBG_wo_risk"
        self.local_mcts_params = mcts_params.clone()
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["CostBasedActionSelection"] = False
        self.local_mcts_params["BehaviorUctHypothesis"]["PredictionSettings"]["UseTrueBehaviorsAsHypothesis"] = False
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["State"]["EvaluationParameters"]["AddSafeDist"] = False
        behavior = BehaviorUCTHypothesis(self.local_mcts_params, hypothesis_set)
        super(BehaviorConfigSBG_wo_risk, self).__init__(behavior_name=behavior_name,
                                          behavior=behavior,
                                          param_descriptions=param_descriptions)
    @staticmethod
    def algorithm_type():
      return "sbg"

class BehaviorConfigRSBGFullInfo(BehaviorConfig):
    def __init__(self, hypothesis_set, mcts_params, param_descriptions=None):
        behavior_name="RSBGFullInfo"
        self.local_mcts_params = mcts_params.clone()
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["CostBasedActionSelection"] = True
        self.local_mcts_params["BehaviorUctHypothesis"]["PredictionSettings"]["UseTrueBehaviorsAsHypothesis"] = True
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["ProgressiveWidening"]["HypothesisBased"] = False
        behavior = BehaviorUCTHypothesis(self.local_mcts_params, [])
        super(BehaviorConfigRSBGFullInfo, self).__init__(behavior_name=behavior_name,
                                          behavior=behavior,
                                          param_descriptions=param_descriptions)
    @staticmethod
    def algorithm_type():
      return "sbg"

class BehaviorConfigRSBGFullInfo_wo_risk(BehaviorConfig):
    def __init__(self, hypothesis_set, mcts_params, param_descriptions=None):
        behavior_name="RSBGFullInfo_wo_risk"
        self.local_mcts_params = mcts_params.clone()
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["CostBasedActionSelection"] = True
        self.local_mcts_params["BehaviorUctHypothesis"]["PredictionSettings"]["UseTrueBehaviorsAsHypothesis"] = True
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["ProgressiveWidening"]["HypothesisBased"] = False
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["State"]["EvaluationParameters"]["AddSafeDist"] = False
        behavior = BehaviorUCTHypothesis(self.local_mcts_params, [])
        super(BehaviorConfigRSBGFullInfo_wo_risk, self).__init__(behavior_name=behavior_name,
                                          behavior=behavior,
                                          param_descriptions=param_descriptions)
    @staticmethod
    def algorithm_type():
      return "sbg"
      
class BehaviorConfigSBGFullInfo_wo_risk(BehaviorConfig):
    def __init__(self, hypothesis_set, mcts_params, param_descriptions=None):
        behavior_name="SBGFullInfo_wo_risk"
        self.local_mcts_params = mcts_params.clone()
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["CostBasedActionSelection"] = False
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["ProgressiveWidening"]["HypothesisBased"] = False
        self.local_mcts_params["BehaviorUctHypothesis"]["PredictionSettings"]["UseTrueBehaviorsAsHypothesis"] = True
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["State"]["EvaluationParameters"]["AddSafeDist"] = False
        behavior = BehaviorUCTHypothesis(self.local_mcts_params, [])
        super(BehaviorConfigSBGFullInfo_wo_risk, self).__init__(behavior_name=behavior_name,
                                          behavior=behavior,
                                          param_descriptions=param_descriptions)
    @staticmethod
    def algorithm_type():
      return "sbg"
      
class BehaviorConfigRCRSBGLocal(BehaviorConfig):
    def __init__(self, hypothesis_set, mcts_params, param_descriptions=None):
        behavior_name="RCRSBGLocal"
        self.local_mcts_params = mcts_params.clone()
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["CostBasedActionSelection"] = True
        self.local_mcts_params["BehaviorUctHypothesis"]["PredictionSettings"]["UseTrueBehaviorsAsHypothesis"] = False
        self.local_mcts_params["BehaviorUctRiskConstraint"]["EstimateScenarioRisk"] = False
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["CostConstrainedStatistic"]["ExplorationReductionFactor"] = 0.0
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["CostConstrainedStatistic"]["ExplorationReductionOffset"] = 0.0
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["CostConstrainedStatistic"]["ExplorationReductionInit"] = 0.0
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["CostConstrainedStatistic"]["MinVisitsPolicyReady"] = -1
        behavior = BehaviorUCTRiskConstraint(self.local_mcts_params, hypothesis_set, None)
        super(BehaviorConfigRCRSBGLocal, self).__init__(behavior_name=behavior_name,
                                          behavior=behavior,
                                          param_descriptions=param_descriptions)
    @staticmethod
    def algorithm_type():
      return "risk"
      
class BehaviorConfigRCSBGLocalFullInfo(BehaviorConfig):
    def __init__(self, hypothesis_set, mcts_params, param_descriptions=None):
        behavior_name="RCSBGLocalFullInfo"
        self.local_mcts_params = mcts_params.clone()
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["CostBasedActionSelection"] = False
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["ProgressiveWidening"]["HypothesisBased"] = False
        self.local_mcts_params["BehaviorUctHypothesis"]["PredictionSettings"]["UseTrueBehaviorsAsHypothesis"] = True
        self.local_mcts_params["BehaviorUctRiskConstraint"]["EstimateScenarioRisk"] = False
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["CostConstrainedStatistic"]["ExplorationReductionFactor"] = 0.0
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["CostConstrainedStatistic"]["ExplorationReductionOffset"] = 0.0
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["CostConstrainedStatistic"]["ExplorationReductionInit"] = 0.0
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["CostConstrainedStatistic"]["MinVisitsPolicyReady"] = -1
        behavior = BehaviorUCTRiskConstraint(self.local_mcts_params, [], None)
        super(BehaviorConfigRCSBGLocalFullInfo, self).__init__(behavior_name=behavior_name,
                                          behavior=behavior,
                                          param_descriptions=param_descriptions)
    @staticmethod
    def algorithm_type():
      return "risk"
      
class BehaviorConfigRCRSBGLocalFullInfo(BehaviorConfig):
    def __init__(self, hypothesis_set, mcts_params, param_descriptions=None):
        behavior_name="RCRSBGLocalFullInfo"
        self.local_mcts_params = mcts_params.clone()
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["CostBasedActionSelection"] = True
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["ProgressiveWidening"]["HypothesisBased"] = False
        self.local_mcts_params["BehaviorUctHypothesis"]["PredictionSettings"]["UseTrueBehaviorsAsHypothesis"] = True
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["CostConstrainedStatistic"]["ExplorationReductionFactor"] = 0.0
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["CostConstrainedStatistic"]["ExplorationReductionOffset"] = 0.0
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["CostConstrainedStatistic"]["ExplorationReductionInit"] = 0.0
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["CostConstrainedStatistic"]["MinVisitsPolicyReady"] = -1
        self.local_mcts_params["BehaviorUctRiskConstraint"]["EstimateScenarioRisk"] = False
        behavior = BehaviorUCTRiskConstraint(self.local_mcts_params, [], None)
        super(BehaviorConfigRCRSBGLocalFullInfo, self).__init__(behavior_name=behavior_name,
                                          behavior=behavior,
                                          param_descriptions=param_descriptions)
    @staticmethod
    def algorithm_type():
      return "risk"
  
class BehaviorConfigRCSBGGlobal(BehaviorConfig):
    def __init__(self, hypothesis_set, scenario_risk_function, mcts_params, param_descriptions=None):
        behavior_name="RCRSBGGlobal"
        self.local_mcts_params = mcts_params.clone()
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["CostBasedActionSelection"] = True
        self.local_mcts_params["BehaviorUctHypothesis"]["PredictionSettings"]["UseTrueBehaviorsAsHypothesis"] = False
        self.local_mcts_params["BehaviorUctRiskConstraint"]["EstimateScenarioRisk"] = True
        behavior = BehaviorUCTRiskConstraint(self.local_mcts_params, hypothesis_set, scenario_risk_function)
        super(BehaviorConfigRCRSBGGlobal, self).__init__(behavior_name=behavior_name,
                                          behavior=behavior,
                                          param_descriptions=param_descriptions)
    @staticmethod
    def algorithm_type():
      return "risk"
      

class BehaviorConfigCooperative(BehaviorConfig):
    def __init__(self, mcts_params, param_descriptions=None):
        behavior_name="Cooperative"
        self.local_mcts_params = mcts_params.clone()
        behavior = BehaviorUCTCooperative(self.local_mcts_params)
        super(BehaviorConfigCooperative, self).__init__(behavior_name=behavior_name,
                                          behavior=behavior,
                                          param_descriptions=param_descriptions)
    @staticmethod
    def algorithm_type():
      return "cooperative"

class BehaviorConfigIntentRSBG(BehaviorConfig):
    def __init__(self, intent_set, mcts_params, param_descriptions=None):
        behavior_name="IntentRSBG"
        self.local_mcts_params = mcts_params.clone()
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["CostBasedActionSelection"] = True
        self.local_mcts_params["BehaviorUctHypothesis"]["PredictionSettings"]["UseTrueBehaviorsAsHypothesis"] = False
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["BeliefTracker"]["PosteriorType"] = 0
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["BeliefTracker"]["HistoryLength"] = 5
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["BeliefTracker"]["ProbabilityDiscount"] = 1.0
        behavior = BehaviorUCTHypothesis(self.local_mcts_params, intent_set)
        super(BehaviorConfigIntentRSBG, self).__init__(behavior_name=behavior_name,
                                          behavior=behavior,
                                          param_descriptions=param_descriptions)
    @staticmethod
    def algorithm_type():
      return "sbg"

class BehaviorConfigIntentRSBG_wo_risk(BehaviorConfig):
    def __init__(self, intent_set, mcts_params, param_descriptions=None):
        behavior_name="IntentRSBG_wo_risk"
        self.local_mcts_params = mcts_params.clone()
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["CostBasedActionSelection"] = True
        self.local_mcts_params["BehaviorUctHypothesis"]["PredictionSettings"]["UseTrueBehaviorsAsHypothesis"] = False
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["BeliefTracker"]["PosteriorType"] = 0
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["BeliefTracker"]["HistoryLength"] = 5
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["BeliefTracker"]["ProbabilityDiscount"] = 1.0
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["State"]["EvaluationParameters"]["AddSafeDist"] = False
        behavior = BehaviorUCTHypothesis(self.local_mcts_params, intent_set)
        super(BehaviorConfigIntentRSBG_wo_risk, self).__init__(behavior_name=behavior_name,
                                          behavior=behavior,
                                          param_descriptions=param_descriptions)
    @staticmethod
    def algorithm_type():
      return "sbg"

class BehaviorConfigRandomMacro(BehaviorConfig):
    def __init__(self, mcts_params, param_descriptions=None):
        behavior_name="Random"
        self.local_mcts_params = mcts_params.clone()
        behavior_macro_tmp = BehaviorMacroActionsFromParamServer(self.local_mcts_params["BehaviorUctBase"]["EgoBehavior"])
        behavior = BehaviorRandomMacroActions(self.local_mcts_params["BehaviorUctBase"]["EgoBehavior"], \
                                              behavior_macro_tmp.GetMotionPrimitives())
        super(BehaviorConfigRandomMacro, self).__init__(behavior_name=behavior_name,
                                          behavior=behavior,
                                          param_descriptions=param_descriptions)
    @staticmethod
    def algorithm_type():
      return "risk"


class BehaviorConfigRCRSBGNHeuristic(BehaviorConfig):
    def __init__(self, hypothesis_set, mcts_params, imitation_agent, param_descriptions=None):
        behavior_name="RCRSBGNHeuristic"
        self.local_mcts_params = mcts_params.clone()
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["CostBasedActionSelection"] = True
        self.local_mcts_params["BehaviorUctBase"]["Mcts"]["HypothesisStatistic"]["ProgressiveWidening"]["HypothesisBased"] = False
        self.local_mcts_params["BehaviorUctHypothesis"]["PredictionSettings"]["UseTrueBehaviorsAsHypothesis"] = False
        self.local_mcts_params["BehaviorUctRiskConstraint"]["EstimateScenarioRisk"] = False
        observer = imitation_agent.observer
        model_file_name = imitation_agent.get_script_filename()
        nn_to_value_converter = imitation_agent.nn_to_value_converter
        behavior = BehaviorUCTNHeuristicRiskConstraint(self.local_mcts_params, hypothesis_set, None, \
                                model_file_name, observer, nn_to_value_converter)
        assert(behavior.ego_behavior.IsEqualTo(imitation_agent.motion_primitive_behavior))
        super(BehaviorConfigRCRSBGNHeuristic, self).__init__(behavior_name=behavior_name,
                                          behavior=behavior,
                                          param_descriptions=param_descriptions)
    @staticmethod
    def algorithm_type():
      return "risk_nheuristic"
      

                                          
