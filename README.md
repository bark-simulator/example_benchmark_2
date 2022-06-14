# Benchmarking Risk-Constrained, Probabilistic and Cooperative Multi-Agent Monte Carlo Tree Search Planners for Autonomous Driving.
The code implements configurable benchmarks of several Multi-Agent Monte Carlo Tree Search planners in applications of autonomous driving. 
The implementation and description of the implemented planners is given in the repository [planner-mcts](https://github.com/bark-simulator/planner-mcts). 

## Contribution & Structure
A fully functional experiment setup to start researching the Multi-Agent MCTS planners implemented in [planner-mcts](https://github.com/bark-simulator/planner-mcts). The repository provides
- A _scenario database_ consisting of different scenario parameterizations of a highway entering scenario. Change the scenario parameters in the json parameter files in `src/evaluation/bark/database_configuration/database/scenario_sets` and run `bazel run //src/evaluation/bark/database_configuration:visualize_scenarios` to see the effects. For instance, configure different microscopic behavior variations of other traffic participnts and intention models under the setting "ConfigBehaviorModels".
- _Easy use of Multi-Agent MCTS planners_ in benchmarks by encapsulating parameters settings within BARK's `BehaviorConfig` class and configuring the parameters based on planner and scenario type. See available configurations and examples in `src/evaluation/bark/behavior_configuration/behavior_configs.py`, the planner-type-specific parameter files in `src/evaluation/bark/behavior_configuration/mcts_params/algorithm_dependent` and the scenario-specific parameter files in `src/evaluation/bark/behavior_configuration/mcts_params/scenario_dependent`.
- _Easy configuration of behavior spaces_ to build hypothesis sets for probablistic microscopic prediction of other traffic participants. The provided configurations use a 1D behavior space build over the desired distance parameter of the Intelligent Driver Model (IDM). Tune or create new behavior space definitions in the folder `src/common/evaluation/bark/behavior_configuration/behavior_spaces/`   
- _Several benchmark scripts_ using the above functionalities to compare and test the performance of the Multi-Agent MCTS planners in a distributed processing manner using BARK's multiprocessing benchmark runner. A detailed description of the benchmarks is given in the subsequent section. These scripts can serve as a starting point to develop own benchmarks and planning algorithms.

## Details on Available Benchmarks
The folder `src/paper/benchmark/` contains five python scripts with the following functionalities:
- **Online tuning of the planning algorithms**: With `bazel run //src/paper/benchmark:check_scenarios`, you can tune the planning algorithms online with enabled visualization of the scenario. This helps to set meaningful parameters of the planning algorithms before starting a full benchmark.
- **Comparison of intent- and behavior-space-based planning algorithms**: The benchmark run with `bazel run //src/paper/benchmark:run_benchmark_intent_compare` compares two MCTS planning variants: 1) One variant uses two behavior hypothesis parameterized to model yielding and no yielding intents 2) The other variant employs only microscopic prediction models obtained by partitioning a behavior space. The algorithms are benchmarked in two scenario variants, with and without _simulated intents_, respectively.
- **Comparison of non-belief- and belief-based MCTS planning algorithms**: The benchmark run with `bazel run //src/paper/benchmark:run_benchmark_planner_comparison` compares different Markov- and Belief-based planning approaches.
- **Testing parallelized MCTS planning**: The benchmark run with `bazel run //src/paper/benchmark:run_benchmark_parallel_mcts` tests how parallelized variants of MCTS planning behave compared to single MCTS variants. 
- **Testing statistical interpretability of risk-constrained MCTS planning**: The benchmark run with `bazel run //src/paper/benchmark:run_benchmark_parallel_mcts` records the amount of safety envelope violations in each scenario obtained when planning with a risk-constrained MCTS planner to evaluated if the observed safety envelope violations correspond to the specified risk level used during planning. This benchmark should be run on a system with many cores, e.g. `num_cpus > 30`, to get meaningful results.

In all benchmark scripts, you can tune various parameters, e.g., the number of benchmark scenarios, the number of planner search iterations or more specific parameters of each algorithm. Take the provided setup in the scripts as starting point, to develop your own benchmark. To obtain statistically more significant results, the number of scenarios must be increased in all benchmarks, eventually requiring running the benchmarks on a system with distributed computational resources. The benchmark scripts provide ways to adjust the number of CPU cores and memory available for benchmarking.

##  Installation

1. Install [Bazel](https://docs.bazel.build/versions/main/install.html).
2. [Clone the repository](https://git.fortiss.org/bark-simulator/example_benchmark_2) and change to the base repository directory
3. `bash tools/python/setup_test_venv.sh`: This will create a virtual python environment (located in ./bark_mcts/python_wrapper/venv)
4. `source tools/python/into_test_venv.sh`: This will activate the virtual environment and set environment variables(keep this in mind for the future: each time you use Bazel, even beyond this installation, be sure to have run this command beforehand)
5. `bazel run [command from above]`: Run one of the benchmark commands or the scenario tuning script mentioned above.


## Citation 
If you use this code for your own research, please cite one of the papers mentioned in [planner-mcts](https://github.com/bark-simulator/planner-mcts).
