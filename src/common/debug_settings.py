import sys
import os

cwd = os.getcwd()
workspace_folder = cwd
repo_paths = ["benchmark_database", "com_github_interaction_dataset_interaction_dataset", \
        "com_github_interaction_dataset_interaction_dataset/python", "planner_uct", "phd", "phd/external/bark_project", "bark_ml_project"]

executed_file = sys.argv[0]
executed_file = executed_file.replace("development/thesis", "development/thesis/bazel-bin")
tmp = executed_file.rsplit(".runfiles/phd", 1)[0]
tmp = tmp.replace(".py", "")
runfiles_dir = f"{tmp}.runfiles"

sys.path.append(runfiles_dir)
for repo in repo_paths:
    full_path = os.path.join(runfiles_dir, repo)
    print("adding python path: {}".format(full_path))
    sys.path.append(full_path)