# example_benchmark_2
Repository implements benchmarks of several Multi-Agent Monte Carlo Tree Search interactive planners for autonomous vehicles. 
The implementation and description of the implemented interactive planners is given in the repository [planner-mcts](https://github.com/bark-simulator/planner-mcts). 

## Main Contribution
A fully functional experiment setup to start researching new Multi-Agent MCTS planners. The repository provides
- A basic scenario database consisting of different scenario parameterizations of a highway entering scenario. Change the scenario parameters in the json parameter files in src/evaluation/bark/database_configuration/database/scenario_sets and run `bazel run //src/evaluation/bark/database_configuration:visualize_scenarios` to see the effects.
- Configuration and easy use of different MCTS planning variations by encapsulating parameters settings within BARK's `BehaviorConfig` class. See available configurations and examples in src/evaluation/bark/behavior_configuration/behavior_configs.py
- 



## Details on the available benchmarks
The folder src/paper/benchmark/ contains five python scripts with the following functionalities:
- `check_scenarios.py`: Compare different scenario parameterizations of the scenarios set provided in 



## Code Structure



##  Setup & Usage
