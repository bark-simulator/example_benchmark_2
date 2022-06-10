
py_test(
  name = "serialize_database",
  srcs = ["serialize_database.py"],
  data = ["@bark_project//bark:generate_core",
          "//src/evaluation/bark/database_configuration/database:database"],
  imports = ['../python/'],
  deps = ["@benchmark_database//serialization:database_serializer",
          "@benchmark_database//load:benchmark_database",
          "@planner_uct//bark_mcts/runtime/scenario:config_readers",
          ],
)

py_test(
  name = "visualize_scenarios",
  srcs = ["visualize_scenarios.py"],
  data = ["@bark_project//bark:generate_core",
          "//src/evaluation/bark/database_configuration/database:database"],
  imports = ['../python/'],
  deps = ["@benchmark_database//serialization:database_serializer",
          "@benchmark_database//load:benchmark_database",
          "@planner_uct//bark_mcts/runtime/scenario:config_readers",
          "@bark_project//bark/runtime/viewer:video_renderer"],
)


py_test(
  name = "scenario_tuning",
  srcs = ["scenario_tuning.py"],
  data = [
        "@bark_project//bark:generate_core",
        "//src/evaluation/bark/database_configuration/test_scenario_params:test_scenario_params",
        "//src/evaluation/bark/database_configuration/database:database",
        "//src/evaluation/bark/database_configuration/visualization_params:visualization_params"
          ],
  imports = ['../python/'],
  deps = ["@benchmark_database//serialization:database_serializer",
          "@benchmark_database//load:benchmark_database",
          "@bark_project//bark/runtime:runtime",
          "@planner_uct//bark_mcts/runtime/scenario:config_readers",
          "//src/common:common"
          ],
)
