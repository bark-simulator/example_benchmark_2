py_library(
  name = "behavior_configuration",
  srcs = [
        "behavior_configurate.py",
        "behavior_configs.py"
        ],
  data = [
      "@bark_project//bark:generate_core"
        ],
  imports = ['../python/'],
  deps = [
          "@bark_project//bark/runtime:runtime",
          "@bark_project//bark/benchmark:benchmark_runner",
          '@planner_uct//bark_mcts/models/behavior/hypothesis/behavior_space:behavior_space',
          ],
   visibility = ["//visibility:public"],
)
