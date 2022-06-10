workspace(name="phd")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")


load("//tools:deps.bzl", "example_benchmark_dependencies")
example_benchmark_dependencies()

# -------- BARK Dependency -------------
git_repository(
    name = "bark_project",
    commit="7f7461c982010e33561ca5428f3f2519a951ac1c",
    remote = "https://github.com/bark-simulator/bark",
)


load("@bark_project//tools:deps.bzl", "bark_dependencies")
bark_dependencies()

load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()


# -------- Benchmark Database -----------------------
git_repository(
  name = "bark_ml_project",
  commit = "b2feabb4bd78f9d86052ca9ef5ae1f32695a34cc",
  remote="https://github.com/juloberno/bark-ml"
)


# ------ Planner UCT ------------------------------
git_repository(
  name = "planner_uct",
  commit="03c8e9d24a5d813b1391a9948157fa7e4204076e",
  remote = "https://github.com/juloberno/bark_hypothesis_uct"
)
load("@planner_uct//util:deps.bzl", "planner_uct_rules_dependencies")
planner_uct_rules_dependencies()
# --------------------------------------------------


# -------- Benchmark Database -----------------------
git_repository(
  name = "benchmark_database",
  commit="f861eae5fb0fb5f8c67a724f3233f3ed4aa0e2a3",
  remote = "https://github.com/bark-simulator/benchmark-database"
)

load("@benchmark_database//util:deps.bzl", "benchmark_database_dependencies")
load("@benchmark_database//load:load.bzl", "benchmark_database_release")
benchmark_database_dependencies()
benchmark_database_release()
# --------------------------------------------------

# Interaction dataset ------------------------------

git_repository(
 name = "interaction_dataset_fortiss_internal",
 commit = "9ace5fde9260c20736b0463026e0f407b7d395ba",
 remote = "https://git.fortiss.org/autosim/interaction_dataset"
)

# Google or tools for mamcts -----------------------------
load("@mamcts_project//util:deps_or.bzl", "google_or_dependencies")
google_or_dependencies()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
# Load common dependencies.
protobuf_deps()

# ---------------------- RSS -----------------------
load("@com_github_rules_rss//rss:rss.bzl", "rss_dependencies")
rss_dependencies()
# --------------------------------------------------