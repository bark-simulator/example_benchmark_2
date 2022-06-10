py_test(
  name = "behavior_config_test",
  srcs = ["behavior_config_test.py"],
  data = [
          "@bark_project//bark:generate_core",
          "//src/evaluation/bark/behavior_configuration/mcts_params:mcts_params",
          "//src/evaluation/bark/behavior_configuration/behavior_spaces:behavior_spaces",
          "//src/evaluation/bark/database_configuration/database:database",
          ],
  imports = ['../python/'],
  deps = [
      "//src/evaluation/bark/behavior_configuration:behavior_configuration",
      "@benchmark_database//load:benchmark_database",
      "@benchmark_database//serialization:database_serializer",
    "@planner_uct//bark_mcts/runtime/scenario:config_readers",
      ]
)


py_test(
  name = "integration_nheuristic_test",
  srcs = ["integration_nheuristic_test.py"],
  data = [
        "@bark_project//bark:generate_core",
        "@bark_ml_project//bark_ml:generate_core",
        "//src/evaluation/bark/behavior_configuration/mcts_params:mcts_params",
        "//src/evaluation/bark/behavior_configuration/behavior_spaces:behavior_spaces",
        "//src/evaluation/bark/database_configuration/visualization_params:visualization_params",
        "//src/evaluation/bark/behavior_configuration/training_params:training_params",
        "//src/evaluation/bark/database_configuration/database:database",
          ],
  imports = ['../python/'],
  deps = [
        "//src/evaluation/bark/behavior_configuration:behavior_configuration",
        "@benchmark_database//load:benchmark_database",
        "@benchmark_database//serialization:database_serializer",
        "@planner_uct//bark_mcts/runtime/scenario:config_readers",
        "@bark_ml_project//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent/demonstrations:demonstrations",
        "@bark_ml_project//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:agents",
        "@bark_project//bark/benchmark:benchmark_runner_mp",
        "//src/evaluation/bark/analysis:custom_viewer",
        ]
)
