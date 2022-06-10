py_test(
  name = "videos_rcrsbg_single",
  srcs = ["videos_rcrsbg_single.py"],
  data = ["@bark_project//bark:generate_core",
           "//src/evaluation/bark/database_configuration/database:database",
          "//src/evaluation/bark/database_configuration/visualization_params:visualization_params",
          "//results/benchmark:benchmark",
          ],
  deps=[
      "//src/common:common",
      "@bark_project//bark/benchmark:benchmark_runner",
      "@benchmark_database//serialization:database_serializer",
      "@bark_project//bark/benchmark:benchmark_runner_mp",
      "@bark_project//bark/benchmark:benchmark_analyzer",
      "//src/evaluation/bark/behavior_configuration:behavior_configuration",
      "//src/evaluation/bark/analysis:custom_video_renderer",
      "//src/evaluation/bark/analysis:custom_viewer",
        "@benchmark_database//load:benchmark_database",
        "@planner_uct//bark_mcts/runtime/scenario:config_readers",
     ],
  imports = ['../python/']
)

py_test(
  name = "videos_rcrsbg_compare",
  srcs = ["videos_rcrsbg_compare.py"],
  data = ["@bark_project//bark:generate_core",
           "//src/evaluation/bark/database_configuration/database:database",
          "//src/evaluation/bark/database_configuration/visualization_params:visualization_params",
          "//results/benchmark:benchmark",
          ],
  deps=[
      "//src/common:common",
      "@bark_project//bark/benchmark:benchmark_runner",
      "@benchmark_database//serialization:database_serializer",
      "@bark_project//bark/benchmark:benchmark_runner_mp",
      "@bark_project//bark/benchmark:benchmark_analyzer",
      "//src/evaluation/bark/behavior_configuration:behavior_configuration",
      "//src/evaluation/bark/analysis:custom_video_renderer",
      "//src/evaluation/bark/analysis:custom_viewer",
        "@benchmark_database//load:benchmark_database",
        "@planner_uct//bark_mcts/runtime/scenario:config_readers",
     ],
  imports = ['../python/']
)

py_test(
  name = "videos_planners",
  srcs = ["videos_planners.py"],
  data = ["@bark_project//bark:generate_core",
           "//src/evaluation/bark/database_configuration/database:database",
          "//src/evaluation/bark/database_configuration/visualization_params:visualization_params",
          "//results/benchmark:benchmark",
          ],
  deps=[
      "//src/common:common",
      "@bark_project//bark/benchmark:benchmark_runner",
      "@benchmark_database//serialization:database_serializer",
      "@bark_project//bark/benchmark:benchmark_runner_mp",
      "//src/evaluation/bark/behavior_configuration:behavior_configuration",
      "//src/evaluation/bark/analysis:custom_viewer",
    "@benchmark_database//load:benchmark_database",
    "@bark_project//bark/benchmark:benchmark_analyzer",
    "@planner_uct//bark_mcts/runtime/scenario:config_readers",
     ],
  imports = ['../python/']
)