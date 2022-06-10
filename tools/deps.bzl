load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")


def example_benchmark_dependencies():
    _maybe(
    native.new_local_repository,
    name = "python_linux",
    path = "./tools/python/venv/",
    build_file_content = """
cc_library(
    name = "python-lib",
    srcs = glob(["lib/libpython3.*", "libs/python3.lib", "libs/python36.lib"]),
    hdrs = glob(["include/**/*.h", "include/*.h"]),
    includes = ["include/python3.6m", "include", "include/python3.7m", "include/python3.5m"], 
    visibility = ["//visibility:public"],
)
    """
    )

    _maybe(
    native.new_local_repository,
    name = "torchcpp",
    path = "./tools/python/venv/lib/python3.7/site-packages/",
    build_file_content = """
cc_library(
    name = "lib",
    srcs = ["torch/lib/libtorch.so",
                 "torch/lib/libc10.so", "torch/lib/libtorch_cpu.so"],
    hdrs = glob(["torch/include/**/*.h", "torch/include/*.h"]),
    strip_include_prefix="torch/include/",
    visibility = ["//visibility:public"],
    linkopts = [
        "-ltorch",
        "-ltorch_cpu",
        "-lc10",
    ],
)
    """)

    _maybe(
        git_repository,
        name = "mamcts_project",
        commit="e6bc1fc84bc97710940252e2ef64c4a6e571fa47",
        remote = "https://github.com/juloberno/mamcts"
    #  path = "/home/bernhard/development/mamcts"
    )

    # -------- BARK Dependency -------------
    _maybe(
        git_repository,
        name = "bark_project",
        commit="7f7461c982010e33561ca5428f3f2519a951ac1c",
        remote = "https://github.com/bark-simulator/bark",
        # path="/home/bernhard/development/bark"
    )

    # -------- Benchmark Database -----------------------
    _maybe(
    git_repository,
        name = "bark_ml_project",
        commit = "b2feabb4bd78f9d86052ca9ef5ae1f32695a34cc",
        remote="https://github.com/juloberno/bark-ml"
        #path = "/home/bernhard/development/bark-ml"
    )

    # ------ Planner UCT ------------------------------
    _maybe(
    git_repository,
        name = "planner_uct",
        commit="03c8e9d24a5d813b1391a9948157fa7e4204076e",
        remote = "https://github.com/juloberno/bark_hypothesis_uct"
        # path="/home/bernhard/development/bark_hypothesis_uct"
    )
    # --------------------------------------------------

    # -------- Benchmark Database -----------------------
    _maybe(
    git_repository,
        name = "benchmark_database",
        commit="f861eae5fb0fb5f8c67a724f3233f3ed4aa0e2a3",
        remote = "https://github.com/bark-simulator/benchmark-database"
        #path = "/home/julo/development/benchmark-database"
    )

    _maybe(
        git_repository,
        name = "interaction_dataset_fortiss_internal",
        commit = "9ace5fde9260c20736b0463026e0f407b7d395ba",
        remote = "https://git.fortiss.org/autosim/interaction_dataset"
    )

def _maybe(repo_rule, name, **kwargs):
    if name not in native.existing_rules():
        repo_rule(name = name, **kwargs)