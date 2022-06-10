import numpy as np
import logging
import random
import math
import colorsys
from collections import defaultdict
from src.common.plotting import *
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.core.geometry import Point2d, Polygon2d
from bark.core.models.behavior import BehaviorUCTRiskConstraint
from bark.core.models.dynamic import StateDefinition, SingleTrackModel
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcol
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

import numpy as np

THESIS_TEXTWIDTH = 423.37708
THESIS_TEXTHEIGHT = 646.40027

from mpl_toolkits.mplot3d import art3d


def format_eval_result(eval_result):
  np.set_printoptions(precision=2, suppress=True)
  if isinstance(eval_result, list) or isinstance(eval_result, tuple):
    stringified_list = ",".join([format_eval_result(i) for i in eval_result])
    formatted_result = f"[{stringified_list}]"
  elif isinstance(eval_result, float):
    formatted_result = "{:.2f}".format(eval_result)
  else:
    formatted_result = "{}".format(eval_result)
  return formatted_result
    
def stringify_eval_results(eval_dict):
  stringified_dict = {}
  for item, val in eval_dict.items():
    stringified_dict[item] = format_eval_result(val)
  return stringified_dict

class AxisTypes():
  def __init__(self, main_axis, policy_axis, belief_axis):
    self.main_axis = main_axis
    self.policy_axis = policy_axis
    self.belief_axis = belief_axis

class AxisCollection():
  def __init__(self):
    self.collection = defaultdict(dict)

  def AddAxisTypes(self, config_plot_idx, time_plot_idx, axis_types):
    self.collection[config_plot_idx][time_plot_idx] = axis_types

  def GetAxisType(self, config_plot_idx, time_plot_idx):
    return self.collection[config_plot_idx][time_plot_idx]


def GetAxisTypeCollectionFromGridSpecSplitHorizontal(gs, parts_info_axis, parts_main_axis):
  inner_grid = gs.subgridspec(2, parts_info_axis + parts_main_axis, wspace=0.2, hspace=0.3)
  matplotlib.rcParams.update(GetMplStyle("traffic"))
  main_axis =plt.subplot(inner_grid[:, parts_info_axis+1:])
  main_axis.axis("off")    
  belief_axis = None
  policy_axis = None
  if parts_info_axis > 0:
    matplotlib.rcParams.update(GetMplStyle("beliefs"))
    belief_axis = ax1 = plt.subplot(inner_grid[0, 0:parts_info_axis+1:])
    matplotlib.rcParams.update(GetMplStyle("policy"))
    policy_axis = ax1 = plt.subplot(inner_grid[1, 0:parts_info_axis+1:])
  return AxisTypes(main_axis, policy_axis, belief_axis)

def GetAxisTypeCollectionFromGridSpecSplitVertical(gs, parts_info_axis, parts_main_axis, 
                         parts_belief_axis, parts_policy_axis, grid_spec_sub={"wspace" : 0.2, "hspace" : 0.0}):
  inner_grid = gs.subgridspec(parts_info_axis + parts_main_axis, parts_belief_axis + parts_policy_axis, **grid_spec_sub)
  matplotlib.rcParams.update(GetMplStyle("traffic"))
  main_axis =plt.subplot(inner_grid[parts_info_axis+1:, :])
  main_axis.axis("off")    
  belief_axis = None
  policy_axis = None
  if parts_info_axis > 0:
    matplotlib.rcParams.update(GetMplStyle("beliefs"))
    belief_axis = ax1 = plt.subplot(inner_grid[0:parts_info_axis+1:, 0:parts_belief_axis])
    matplotlib.rcParams.update(GetMplStyle("policy"))
    policy_axis = ax1 = plt.subplot(inner_grid[0:parts_info_axis+1:, parts_belief_axis+1:])
  return AxisTypes(main_axis, policy_axis, belief_axis)
   
def GetAxisTypeCollectionFromGridSpec(gs, **kwargs):
  split_type = kwargs.pop("split_type", "horizontal")
  if split_type == "horizontal":
    return GetAxisTypeCollectionFromGridSpecSplitHorizontal(gs, **kwargs)
  elif split_type == "vertical":
    return GetAxisTypeCollectionFromGridSpecSplitVertical(gs, **kwargs)

def CreateAxisCollectionGrid(figure, vert, hor, **kwargs):
  # first start with the outer spec
  grid_spec_main = kwargs.pop("grid_spec_main", {"wspace" : 0.1, "hspace" : 0.1, \
           "left" : 0.00, "right" : 1.0, "bottom" : 0.0, "top" : 1.0})
  gs = figure.add_gridspec(vert, hor, width_ratios=[1]*hor,
  height_ratios=[1]*vert, **grid_spec_main)

  axis_collection = AxisCollection()
  for vert_idx in range(0, vert+1):
    top = grid_spec_main["top"]
    bottom = grid_spec_main["bottom"]
    height = bottom - top
    l = matplotlib.lines.Line2D([grid_spec_main["left"], grid_spec_main["right"]], [top + vert_idx*height/vert, top + vert_idx*height/vert], \
                    transform=figure.transFigure, figure=figure, \
                        color="lightgrey", linewidth=0.5)
    figure.lines.extend([l])
  for hor_idx in range(0, hor+1):
    left = grid_spec_main["left"]
    right = grid_spec_main["right"]
    width = right - left
    l = matplotlib.lines.Line2D([left+hor_idx*width/hor, left+hor_idx*width/hor], [grid_spec_main["bottom"], grid_spec_main["top"]], \
                    transform=figure.transFigure, figure=figure, \
                        color="lightgrey", linewidth=0.5)
    figure.lines.extend([l])
  for hor_idx in range(0, hor):
    for vert_idx in range(0, vert):
      axis_types = GetAxisTypeCollectionFromGridSpec(gs[vert_idx, hor_idx], **kwargs)
      axis_collection.AddAxisTypes(config_plot_idx = hor_idx, time_plot_idx = vert_idx, \
                axis_types = axis_types)
  return axis_collection

def CreateAxisCollectionDownwards(figure, num_configs, num_time_steps, **kwargs):
  # first start with the outer spec
  grid_spec_main = kwargs.pop("grid_spec_main", {"wspace" : 0.1, "hspace" : 0.1, \
           "left" : 0.00, "right" : 1.0, "bottom" : 0.0, "top" : 1.0})
  gs = figure.add_gridspec(num_time_steps, num_configs, width_ratios=[1]*num_configs,
  height_ratios=[1]*num_time_steps, **grid_spec_main)


  axis_collection = AxisCollection()
  for config_plot_idx in range(0, num_configs):
    l = matplotlib.lines.Line2D([1.0/num_configs, 1.0/num_configs], [grid_spec_main["bottom"], grid_spec_main["top"]], \
                         transform=figure.transFigure, figure=figure, \
                            color="lightgrey", linewidth=0.5)
    figure.lines.extend([l])
    for time_plot_idx in range(0, num_time_steps):
      axis_types = GetAxisTypeCollectionFromGridSpec(gs[time_plot_idx, config_plot_idx], **kwargs)
      axis_collection.AddAxisTypes(config_plot_idx = config_plot_idx, time_plot_idx = time_plot_idx, \
                axis_types = axis_types)
  return axis_collection


class CustomViewer(MPViewer):
    def __init__(self,
                 params=None,
                 axis_policy = None,
                 axis_beliefs = None,
                 use_wait_for_click=False,
                 plot3d = False,
                 **kwargs):
        super(CustomViewer, self).__init__(params, **kwargs)
        self.axis_beliefs = axis_beliefs
        self.axis_policy = axis_policy
        self.belief_draw_region = None
        self.belief_draw_ranges = []
        self.use_wait_for_click = use_wait_for_click
        self.plot3d = plot3d
        if use_wait_for_click:
          self.InitWaitForClick()

    def initialize_params(self, params, kwargs={}):
      super(CustomViewer, self).initialize_params(params, kwargs)
      self.setupColorMap()

    def setupColorMap(self):
      N = 25
      colors = []
      for x in range(N):
        HSV = [x/N, 0.5, 0.5]
        colors.append(colorsys.hsv_to_rgb(*HSV))
      self.max_agents_color_map = 25
      split_shuffle = 5
      chunks = [colors[i::split_shuffle] for i in range(0, split_shuffle)]
      colors = []
      for idx, chunk in enumerate(chunks): colors.extend(chunk)
      for idx, color in enumerate(colors): self.agent_color_map[idx] = color
      #random.Random(10).shuffle(self.agent_color_map)

    def ExtractActionMapping(self, behavior_macro_actions):
      primitives = behavior_macro_actions.GetMotionPrimitives()
      primitives_names = [primitive.name.replace("Primitive", ""). \
            replace("ConstAcc", "").replace("StayLane","").replace("ChangeTo", "Change").replace(": ", "") for primitive in primitives]
      primitives_names = [primitive.replace("Lefta=0", "Left").replace("Righta=0", "Right") for primitive in primitives_names]
      return primitives_names

    def UpdateScenarioText(self, text):
      self.current_scenario_text = text

    def drawWorld(self, world, *args, **kwargs):
      self.current_time = world.time
      super(CustomViewer, self).drawWorld(world, *args, **kwargs)
      #ego_agent = world.agents[args[0][0]]
      #super(CustomViewer, self).drawLaneCorridor(ego_agent.road_corridor.lane_corridors[0], "green")

    def drawEvalResults(self, evaluation_dict):
      evaluation_dict = stringify_eval_results(evaluation_dict)
      super(CustomViewer, self).drawEvalResults(evaluation_dict)
      if self.use_wait_for_click:
        self.WaitForClick()
        while self.WaitingForClick():
          plt.pause(0.05)

    def getNearestIds(self, world, ego_id, num):
      state = world.agents[ego_id].state
      point = Point2d(state[int(StateDefinition.X_POSITION)], \
                    state[int(StateDefinition.Y_POSITION)])
      nearest_agents = world.GetNearestAgents(point, num+1)
      return [idx for idx, agent in nearest_agents.items() if idx !=ego_id]

    def drawBeliefsHypothesis1D(self, behavior, others_ids, xlabel):
      if not self.belief_draw_ranges:
        self.ExtractBeliefDrawParameters1D(behavior)
      for other_id in others_ids:
        color = self.agent_color_map[other_id]
        plot_y = behavior.current_beliefs[other_id]
        plot_y.append(plot_y[-1])
        plot_x = self.belief_draw_ranges 
        self.axis_beliefs.step(plot_x, plot_y, where="post", color=color, linewidth=0.01)
        self.axis_beliefs.tick_params(axis='both', which='major', pad=2.5, length=0)
        self.axis_beliefs.fill_between(plot_x, plot_y, step="post", alpha=0.2, color=color)
        self.axis_beliefs.set_xlabel(xlabel)
        self.axis_beliefs.xaxis.set_label_coords(0.9, -0.07)
        self.axis_beliefs.set_ylabel("$\\text{Pr}(\hbst{}{}|\cdot)$")
        self.axis_beliefs.yaxis.set_label_coords(-0.02, 0.5)
        self.axis_beliefs.spines["top"].set_linewidth(0.0)
        self.axis_beliefs.spines["right"].set_linewidth(0.0)
        self.axis_beliefs.spines["left"].set_linewidth(0.5)
        self.axis_beliefs.spines["bottom"].set_linewidth(0.5)
        plt.setp(self.axis_beliefs.get_xticklabels(), visible=False)
        plt.setp(self.axis_beliefs.get_yticklabels(), visible=False)

    def drawPolygon2d(self, polygon, color, alpha, facecolor=None, linewidth=1, zorder=10, hatch=''):
      if not self.plot3d:
        super(CustomViewer, self).drawPolygon2d(polygon, color, alpha, facecolor, linewidth, zorder, hatch)
      else:
        self.drawPolygon3d(polygon, color, alpha, facecolor, linewidth, zorder, hatch)

    def drawPolygon3d(self, polygon, color, alpha, facecolor=None, linewidth=1, zorder=10, hatch=''):
      points = polygon.ToArray()
      polygon_draw = matplotlib.patches.Polygon(
          points,
          True,
          zorder = zorder,
          facecolor=self.getColor(facecolor),
          edgecolor=self.getColor(color),
          alpha=alpha,
          linewidth=linewidth,
          hatch=hatch)
      t_start = self.axes.transData
      polygon_draw.set_transform(t_start)
      self.axes.add_patch(polygon_draw)
      art3d.pathpatch_2d_to_3d(polygon_draw, z=zorder*0.02, zdir="z")

    def drawBeliefsHypothesis2D(self, behavior, others_ids, label_map):
      if not self.belief_draw_ranges:
        self.ExtractBeliefDrawParameters2D(behavior)
      for idx,other_id in enumerate(others_ids):
        plot_y = list(self.belief_draw_ranges.values())[0]
        plot_x = list(self.belief_draw_ranges.values())[1]
        xlabel =label_map[list(self.belief_draw_ranges.keys())[1]]
        ylabel = label_map[list(self.belief_draw_ranges.keys())[0]]
        beliefs = behavior.current_beliefs[other_id]
        color = self.agent_color_map[other_id]
        

        def hatch_map(belief, agent_id):
          hatch_dict = {0 : "-", 1 : "|"}
          hatch = hatch_dict[agent_id]
          resolution = 0.05
          return round(belief/resolution)*hatch
        for hyp_id, belief in enumerate(beliefs):
          found_x = list(filter(lambda x : x[1] == hyp_id, plot_x))
          assert(len(found_x) == 1)
          x_patch = found_x[0][0]
          found_y = list(filter(lambda x : x[1] == hyp_id, plot_y))
          assert(len(found_y) == 1)
          y_patch = found_y[0][0]
          x_width = x_patch[1] - x_patch[0]
          y_width = y_patch[1] - y_patch[0]
          matplotlib.rcParams['hatch.linewidth'] = 0.2
          patch = Rectangle((x_patch[0], y_patch[0]), x_width, y_width, alpha=0.7, linewidth=0.01, edgecolor=(0.1,0.1,0.1, 1.0), facecolor = 'none')
          self.axis_beliefs.add_patch(patch)
          patch = Rectangle((x_patch[0], y_patch[0]), x_width, y_width, alpha=0.7, linewidth=0.0, edgecolor=color, facecolor = 'none', hatch =hatch_map(belief, idx))
          self.axis_beliefs.add_patch(patch)
        self.axis_beliefs.autoscale()
        self.axis_beliefs.set_ylabel(ylabel)
        self.axis_beliefs.set_xlabel(xlabel)
        self.axis_beliefs.yaxis.set_label_coords(-0.02, 0.5)
        self.axis_beliefs.spines["top"].set_linewidth(0.0)
        self.axis_beliefs.spines["right"].set_linewidth(0.0)
        self.axis_beliefs.spines["left"].set_linewidth(0.5)
        self.axis_beliefs.spines["bottom"].set_linewidth(0.5)
        plt.setp(self.axis_beliefs.get_xticklabels(), visible=False)
        plt.setp(self.axis_beliefs.get_yticklabels(), visible=False)
        self.axis_beliefs.tick_params(axis='both', which='major', pad=2.5, length=0)

    def drawBeliefsIntent(self, behavior, others_ids, xticks=["yield", "no yield"]):
        num_bars = 2 # always two intents
        bar_width = 0.1
        agent_pos = 0
        for other_id in others_ids:
          color = self.agent_color_map[other_id]
          plot_y = np.array(behavior.current_beliefs[other_id])
          if np.all(plot_y == 0.0):
            plot_y += 0.03
          bar_pos = [x + agent_pos*bar_width for x in np.arange(num_bars)]
          self.axis_beliefs.bar(bar_pos, plot_y, width = bar_width, color=color, linewidth=0.01, fc=(*color, 0.2))
          self.axis_beliefs.tick_params(axis='both', which='major', pad=2.5, length=0)
          self.axis_beliefs.xaxis.set_label_coords(0.9, -0.07)
          self.axis_beliefs.set_ylabel("$\\text{Pr}(\hbst{}{}|\cdot)$")
          self.axis_beliefs.yaxis.set_label_coords(-0.02, 0.5)
          self.axis_beliefs.spines["top"].set_linewidth(0.0)
          self.axis_beliefs.spines["right"].set_linewidth(0.0)
          self.axis_beliefs.spines["left"].set_linewidth(0.5)
          self.axis_beliefs.spines["bottom"].set_linewidth(0.5)
          plt.setp(self.axis_beliefs.get_yticklabels(), visible=False)
          agent_pos += 1
        self.axis_beliefs.tick_params(axis='both', which='major', pad=2.5, length=0)
        self.axis_beliefs.set_xticks([x + bar_width*(agent_pos-1)/2 for x in range(num_bars)])
        self.axis_beliefs.set_xticklabels(xticks)
        plt.pause(0.0001)
        

    def drawPolicy(self, behavior, action_mapping, ylabel=False):
        stoch_policy = behavior.last_policy_sampled[1]
        optimal_action_idx = behavior.last_policy_sampled[0]
        cost_envelope_values = behavior.last_cost_values["envelope"]
        cost_collision_values = behavior.last_cost_values["collision"]
        return_values = behavior.last_return_values
        num_actions = len(action_mapping)
        action_ticks = np.arange(num_actions)
        bar_width = 0.15
        #risk_constraint_env = behavior.
        expected_env_risk = behavior.last_expected_risk[0]
        required_env_risk = behavior.last_scenario_risk[0]
        expected_col_risk = behavior.last_expected_risk[1]
        hatches = ["/", "-", "\\", "."]

        with matplotlib.rc_context(rc=GetMplStyle("policy")):
          bars_policy = self.policy_axis.bar(list(stoch_policy.keys()), list(stoch_policy.values()), \
                    edgecolor="k", color="black", fill=True, width=bar_width,label="$\p{i}(\\a{}{}{}|\oh{t})$")
          bars_cost_env = self.policy_axis.bar([x + bar_width for x in list(cost_envelope_values.keys())], list(cost_envelope_values.values()), \
                    edgecolor="k", color="lightgrey", fill=False, width=bar_width, label="$\\rho_\\text{env}(\\langle\ost{t}{}\\rangle,\\a{}{}{})$")
          bars_cost_col = self.policy_axis.bar([x + 2*bar_width for x in list(cost_collision_values.keys())], list(cost_collision_values.values()), \
                    edgecolor="k", color="lightgrey", fill=False, width=bar_width, label="$\\rho_\\text{col}(\\langle\ost{t}{}\\rangle,\\a{}{}{})$")
          bars_returns = self.policy_axis.bar([x + 3*bar_width for x in list(return_values.keys())], list(return_values.values()), \
                    edgecolor="k", color="lightgrey", fill=True, width=bar_width, label="$Q_R(\\langle\ost{t}{}\\rangle,\\a{}{}{})$")
              
          hatch_density = 5
          for bar in bars_policy:
            bar.set_hatch("/"*hatch_density)
          for bar in bars_cost_env:
            bar.set_hatch("-"*hatch_density)
          for bar in bars_cost_col:
            bar.set_hatch("."*hatch_density)
          for bar in bars_returns:
            bar.set_hatch("\\"*hatch_density)
        
        self.policy_axis.tick_params(axis='x', which='major', pad=2.5, length=0.0)
        self.policy_axis.tick_params(axis='y', which='major', pad=2.5, length=3)
        self.policy_axis.set_xticklabels(["{:1.1f}".format(label) for label  \
            in self.policy_axis.get_xticks()], {"family" : "sans-serif"})
       # self.policy_axis.set(xlim=(-bar_width/2.0, num_actions+1))
        self.policy_axis.axhline(expected_env_risk, -0.25, color="r", linestyle = "dashed", linewidth=1.0, alpha=0.5, label="$\\rho_\\text{env}^\\text{exp.}$")
        self.policy_axis.axhline(required_env_risk, -0.25, color="b", linestyle = "solid", linewidth=1.0, alpha=0.5, label="$\\beta$")
        self.policy_axis.axhline(expected_col_risk, -0.25, color="r", linestyle = "dotted", linewidth=1.0, alpha=0.5, label="$\\rho_\\text{col}^\\text{exp.}$")

        # self.axis_beliefs.set_xlabel(xlabel, size=4)
        self.policy_axis.xaxis.set_label_coords(0.5, -0.1)
        self.policy_axis.yaxis.set_label_coords(-0.08, 0.5)
        self.policy_axis.spines["top"].set_linewidth(0.0)
        self.policy_axis.spines["right"].set_linewidth(0.0)
        self.policy_axis.spines["left"].set_linewidth(0.5)
        self.policy_axis.spines["bottom"].set_linewidth(0.5)
       # plt.setp(self.policy_axis.get_xticklabels(), visible=False)
        self.policy_axis.set_xticks([x + bar_width for x in range(num_actions)], )
        self.policy_axis.set_xticklabels(action_mapping)
        if not ylabel:
          plt.setp(self.policy_axis.get_yticklabels(), visible=False)
        else:
          self.policy_axis.set_yticks([0.0, 1.0])
          self.policy_axis.set_ylim([0.0, 1.0])

    def getColorFromMap(self, double_color):
      hsv = (np.random.uniform(0,1,1)[0], double_color, 0.5)
      rgb = colorsys.hsv_to_rgb(*hsv)
      return rgb

    def ExtractBeliefDrawParameters1D(self, behavior):
      hypotheses = behavior.hypotheses
      assert(len(hypotheses) > 0)
      self.belief_draw_region = None
      belief_draw_ranges_list = []
      for hypothesis in hypotheses:
        for param_name, param_range in hypothesis.parameter_regions.items():
          if param_range[0] != param_range[1]:
            if self.belief_draw_region and \
                self.belief_draw_region != param_name:
              raise ValueError("Belief drawing only for 1D hypothesis supported")
            else:
              self.belief_draw_region = param_name
              belief_draw_ranges_list.append(param_range)
      belief_draw_ranges_list = sorted(belief_draw_ranges_list, key=lambda x: x[1])
      self.belief_draw_ranges = [x[0] for x in belief_draw_ranges_list]
      self.belief_draw_ranges.append(belief_draw_ranges_list[-1][1])

    def ExtractBeliefDrawParameters2D(self, behavior):
      hypotheses = behavior.hypotheses
      assert(len(hypotheses) > 0)
      belief_draw_ranges_list = defaultdict(list)
      for hyp_id, hypothesis in enumerate(hypotheses):
        for param_name, param_range in hypothesis.parameter_regions.items():
          if param_range[0] != param_range[1]:
            belief_draw_ranges_list[param_name].append((param_range, hyp_id))
      self.belief_draw_ranges = belief_draw_ranges_list
      return belief_draw_ranges_list

    def drawSearchTree(self, behavior, ego_id, **kwargs):
      mcts_extracted_edges = behavior.edge_infos
      lowest_plot_time = None
      for edge in mcts_extracted_edges:
        if edge[1] == 0:
          lowest_plot_time = edge[4][0, int(StateDefinition.TIME_POSITION)]
          print(lowest_plot_time)
          break
      for edge in mcts_extracted_edges:
        self.drawMctsEdgeTrajectory(edge, ego_id, lowest_plot_time, **kwargs)
    def drawMctsEdgeTrajectory(self, edge, ego_id, lowest_plot_time, **kwargs):
      # idx 0: agent id
      # idx 1: depth
      # idx 2: action id
      # idx 3: action weight
      traj = edge[4]
      width = edge[3]
      maxw = kwargs.pop("maxw", 2.0)
      minw = kwargs.pop("minw", 0.5)
      others_ids = kwargs.pop("others_ids", None)
      norm_range = kwargs.pop("norm_range", [0, 10.0])
      def normalize_width(width):
        maxp = 0.5
        if math.isnan(width):
          return minw
        return min(max(width/maxp*(maxw-minw) + minw, maxw), minw)
      agent_id = edge[0]
      if agent_id != ego_id and others_ids and agent_id not in others_ids:
        return
      lw = normalize_width(width)
      if(len(traj) > 0):
        x = traj[:, int(StateDefinition.X_POSITION)]
        y = traj[:, int(StateDefinition.Y_POSITION)]
        t = traj[:, int(StateDefinition.TIME_POSITION)]-lowest_plot_time
        cmap = LinearSegmentedColormap.from_list("", [(1, 0, 0), (0, 0, 1)])
        points = np.array([x, y]).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-1],points[1:]], axis=1)
        lc = LineCollection(segments, cmap=plt.cm.rainbow, linewidth=lw, norm= plt.Normalize(*norm_range), alpha=0.2, zorder=20)
        lc.set_array(t)
        self.axes.add_collection(lc)
        state = traj[-1]
     # self.axes.plot(
     #             state[int(StateDefinition.X_POSITION)],
      #            state[int(StateDefinition.Y_POSITION)],
     #            state[int(StateDefinition.TIME_POSITION)],
     #             marker='o', markersize=0.2 , alpha=0.3,
     #             color="k")
    
    def GetBehaviorInfoText(self, behavior):
      if isinstance(agent.behavior_model, BehaviorUCTRiskConstraint):
        return "$t={:2.1f}$ [s], $\rho_\text{safe}={:2.1f}$, $\rho_\text{coll}={:2.1f}$".format(world_idx*0.2, ego_behavior.last_expected_risk[0],
                            ego_behavior.last_expected_risk[1])
      else:
        return ""

    def OnMainAxisClick(self, event):
      self.waiting_for_click = False

    def WaitForClick(self):
      self.waiting_for_click = True

    def WaitingForClick(self):
      return self.waiting_for_click

    def InitWaitForClick(self):
      self.use_wait_for_click = True
      self.axes.get_figure().canvas.mpl_connect("button_press_event", self.OnMainAxisClick)
