workspace(name="phd")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")


load("//tools:deps.bzl", "example_benchmark_dependencies")
example_benchmark_dependencies()

# -------- BARK Dependency -------------
load("@bark_project//tools:deps.bzl", "bark_dependencies")
bark_dependencies()

load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")

# -------- BOOST Dependency -------------
load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()



# ------ Planner UCT ------------------------------
load("@planner_uct//util:deps.bzl", "planner_uct_rules_dependencies")
planner_uct_rules_dependencies()
# --------------------------------------------------

# -------- Benchmark Database -----------------------
load("@benchmark_database//util:deps.bzl", "benchmark_database_dependencies")
load("@benchmark_database//load:load.bzl", "benchmark_database_release")
benchmark_database_dependencies()
benchmark_database_release()
# --------------------------------------------------


# Google or tools for mamcts -----------------------------
load("@mamcts_project//util:deps_or.bzl", "google_or_dependencies")
google_or_dependencies()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()
