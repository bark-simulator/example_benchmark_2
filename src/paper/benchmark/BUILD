py_test(
  name = "run_benchmark_intent_compare",
  srcs = ["run_benchmark_intent_compare.py"],
   data = [
          "@bark_project//bark:generate_core",
          "//src/evaluation/bark/behavior_configuration/mcts_params:mcts_params",
          "//src/evaluation/bark/database_configuration/database:database",
          "//src/evaluation/bark/database_configuration/visualization_params:visualization_params",
          "//src/evaluation/bark/behavior_configuration/behavior_spaces:behavior_spaces"
          ],
  imports = [
            '../python/'
            ],
  deps = [
        "@benchmark_database//serialization:database_serializer",
          "@bark_project//bark/benchmark:benchmark_runner_mp",
          "@benchmark_database//load:benchmark_database",
          "//src/evaluation/bark/behavior_configuration:behavior_configuration",
          "@planner_uct//bark_mcts/runtime/scenario:config_readers",
          "//src/common:common"
          ],
)


py_test(
  name = "run_benchmark_rcrsbg_risk_increase",
  srcs = ["run_benchmark_rcrsbg_risk_increase.py"],
   data = [
          "@bark_project//bark:generate_core",
          "//src/evaluation/bark/behavior_configuration/mcts_params:mcts_params",
          "//src/evaluation/bark/database_configuration/database:database",
          "//src/evaluation/bark/database_configuration/visualization_params:visualization_params",
          "//src/evaluation/bark/behavior_configuration/behavior_spaces:behavior_spaces",
          ],
  imports = [
            '../python/'
            ],
  deps = [
        "@benchmark_database//serialization:database_serializer",
          "@bark_project//bark/benchmark:benchmark_runner_mp",
          "@benchmark_database//load:benchmark_database",
          "//src/evaluation/bark/behavior_configuration:behavior_configuration",
          "@planner_uct//bark_mcts/runtime/scenario:config_readers",
          "@bark_project//bark/benchmark:benchmark_analyzer",
          "//src/common:common"
          ],
)



py_test(
  name = "histories_rcrsbg",
  srcs = ["histories_rcrsbg.py"],
   data = [
          "@bark_project//bark:generate_core",
          "//src/evaluation/bark/behavior_configuration/mcts_params:mcts_params",
          "//src/evaluation/bark/database_configuration/database:database",
          "//src/evaluation/bark/database_configuration/visualization_params:visualization_params",
          "//src/evaluation/bark/behavior_configuration/behavior_spaces:behavior_spaces",
          "//results/benchmark:benchmark"
          ],
  imports = [
            '../python/'
            ],
  deps = [
        "@benchmark_database//serialization:database_serializer",
          "@bark_project//bark/benchmark:benchmark_runner_mp",
          "@benchmark_database//load:benchmark_database",
          "//src/evaluation/bark/behavior_configuration:behavior_configuration",
          "@planner_uct//bark_mcts/runtime/scenario:config_readers",
          "@bark_project//bark/benchmark:benchmark_analyzer",
          "//src/common:common"
          ],
)

py_test(
  name = "run_benchmark_planner_comparison",
  srcs = ["run_benchmark_planner_comparison.py"],
   data = [
          "@bark_project//bark:generate_core",
          "//src/evaluation/bark/behavior_configuration/mcts_params:mcts_params",
          "//src/evaluation/bark/database_configuration/database:database",
          "//src/evaluation/bark/database_configuration/visualization_params:visualization_params",
          "//src/evaluation/bark/behavior_configuration/behavior_spaces:behavior_spaces",
          ],
  imports = [
            '../python/'
            ],
  deps = [
        "@benchmark_database//serialization:database_serializer",
          "@bark_project//bark/benchmark:benchmark_runner_mp",
          "@benchmark_database//load:benchmark_database",
          "//src/evaluation/bark/behavior_configuration:behavior_configuration",
          "@planner_uct//bark_mcts/runtime/scenario:config_readers",
          "@bark_project//bark/benchmark:benchmark_analyzer",
          "//src/common:common"
          ],
)


py_test(
  name = "run_benchmark_parallel_mcts",
  srcs = ["run_benchmark_parallel_mcts.py"],
   data = [
          "@bark_project//bark:generate_core",
          "//src/evaluation/bark/behavior_configuration/mcts_params:mcts_params",
          "//src/evaluation/bark/database_configuration/database:database",
          "//src/evaluation/bark/database_configuration/visualization_params:visualization_params",
          "//src/evaluation/bark/behavior_configuration/behavior_spaces:behavior_spaces"
          ],
  imports = [
            '../python/'
            ],
  deps = [
        "@benchmark_database//serialization:database_serializer",
          "@bark_project//bark/benchmark:benchmark_runner_mp",
          "@benchmark_database//load:benchmark_database",
          "//src/evaluation/bark/behavior_configuration:behavior_configuration",
          "@planner_uct//bark_mcts/runtime/scenario:config_readers",
          "//src/common:common"
          ],
)


py_test(
  name = "check_scenarios",
  srcs = ["check_scenarios.py"],
   data = [
          "@bark_project//bark:generate_core",
          "//src/evaluation/bark/behavior_configuration/mcts_params:mcts_params",
          "//src/evaluation/bark/database_configuration/database:database",
          "//src/evaluation/bark/database_configuration/visualization_params:visualization_params",
          "//src/evaluation/bark/behavior_configuration/behavior_spaces:behavior_spaces"
          ],
  imports = [
            '../python/'
            ],
  deps = [
        "@benchmark_database//serialization:database_serializer",
          "@bark_project//bark/benchmark:benchmark_runner_mp",
          "@benchmark_database//load:benchmark_database",
          "//src/evaluation/bark/analysis:custom_viewer",
          "//src/evaluation/bark/behavior_configuration:behavior_configuration",
          "@planner_uct//bark_mcts/runtime/scenario:config_readers",
          ],
)

