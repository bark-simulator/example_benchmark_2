import subprocess
import os
import shutil

BAZEL_BUILD_SUB_FOLDER = "bazel-bin"
CLUSTER_EXPERIMENT_MAIN_DIR = "mnt/gluster/home/bernhard/experiments/"


class ClusterResources:
    """
    Defines a resource request on the cluster.

    Parameters
    ----------
    cpus : int, optional
        Number of CPUs to reserve; default is 2.
    memory : str, optional
        Memory representation (same as slurm) to execute. Default is 10GB
    gpus : int, optional
        Number of GPUs to reserve. Default is None

    """
    def __init__(self, cpus=2, memory="10GB", gpus=None):
        self.cpus, self.memory, self.gpus = cpus, memory, gpus

    def options_string(self):
        options = ["-c {}".format(self.cpus),
                   "--mem {}".format(self.memory)]
        if self.gpus:
            options.append("--gres=gpu:{}".format(self.gpus))
        return " ".join(options)


class ClusterExecutor:
    """
    Executor to run something on our cluster.

    It takes a server and the path to an ssh key (or the "LABPAGE_CLUSTER_KEY" environment variable),
    connects, and afterwards can execute experiments.

    Parameters
    ----------
    server : str
        Servername of the server to connect to. Probably "4gpu"
    cluster_key : str, optional
        Path to the key. If not set, reads it from the LABPAGE_CLUSTER_KEY environment variable. If
        not available either, raises a ValueError.

    Attributes
    ----------
    client : paramiko.SSHClient
        The connection
    """

    def __init__(self, server: str, cluster_key: str=None):
        self.client = utils.create_connection(server, cluster_key)

    def execute_experiment(self,
                           executed_file_relative_path,
                           workspace_dir,
                           cluster_exp_main_dir,
                           experiment_name,
                           resources,
                           prebuilt_image,
                           account,
                           qos: str=None):
        """
        Executes an (existing) experiment on the cluster.

        Parameters
        ----------
        target_path : Path
            The already-exported experiment run to execute
        resources : ClusterResources, optional
            The resources requested.
        prebuilt_image : Path, optional
            If set, will use the specified image instead of building a new one.
        extra_data_folder : Path, optional
            If set, will use the specified data folder.
        account : str, optional
            The account to use. If not set, will use default slurm account.
        qos : str, optional
            QOS to use. If not set, will use default slurm qos.

        Returns
        -------
        job_id : str
            ID of the job.
        """
        batch_file_raw = self._batch_file(prebuilt_image=prebuilt_image)
        # flip such that it's name_date instead of date_name
        job_name = target_path.name[27:] + "_" + target_path.name[:26]
        options_txt = self._options_str(job_name, resources, account=account, qos=qos)
        command_to_execute = "cd {exp_dir}; mkdir -p results; sbatch {options_txt} << {batch_file_txt}".format(
            exp_dir=str(target_path),
            options_txt=options_txt,
            batch_file_txt=batch_file_raw
        )
        _, stdout, _ = self.client.exec_command(command_to_execute)
        job_id = self._job_id_from_stdout(stdout)
        return job_id

    def _job_id_from_stdout(self, stdout):
        result = stdout.read().strip().decode('ascii')
        # this relies on the last word of the result being the job id.
        job_id = result.rsplit(None, 1)[-1]
        return job_id

    def _batch_file(self, image_path, experiment_dir, python_cmd):
        singularity_cmd = "srun singularity exec " \ 
            "--nv -B {exp_dir}:./:ro {image} {pycmd}",

        batch_file_lines = [
            "EOF",
            "#!/bin/sh",
            "#SBATCH -c 4",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --mem=10GB",

            "EOF"
        ]
        batch_file_raw = "\n".join(batch_file_lines)
        return batch_file_raw

    def _options_str(self, job_name, resources, account=None, qos=None):
        options = ["--output=./results/slurm-output.out",
                   "--job-name={}".format(job_name),
                   resources.options_string()]
        if account:
            options.append("--account={}".format(account))
        if qos:
            options.append("--qos={}".format(qos))

        options_txt = " ".join(options)
        return options_txt

        
    def _get_build_path(self, workspace_dir, executed_file_relative_path):
        return os.path.join(workspace_dir, BAZEL_BUILD_SUB_FOLDER, os.path.dirname(executed_file_relative_path))


    def _build_command(self, executed_file_relative_path, workspace_dir):
        os.chdir(workspace_dir)
        command = "bazel build //{}".format(executed_file_relative_path)
        subprocess.check_call(command, shell=True, capture_output=True)
        return get_build_path(workspace_dir, executed_file_relative_path)

    def _make_cleaned_up_experiment_dir(self, dir, experiment_name):
        tmp_dir = os.path.join("/tmp",experiment_name)
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        shutil.copytree(dir, tmp_dir, symlinks=False) # copy to resolve symlinks
        return tmp_dir

    def _copy_to_cluster(self, experiment_local_dir, cluster_exp_main_dir):
        experiment_name = os.path.basename(experiment_local_dir)
        cluster_exp_dir = os.path.join(cluster_exp_main_dir, experiment_name)
        if not os.makedirs(cluster_exp_dir):
            raise FileNotFoundError()
        shutil.copytree(experiment_local_dir, cluster_exp_dir, symlinks=False)



    def deploy_to_cluster(self, executed_file_relative_path, workspace_dir, cluster_exp_main_dir, experiment_name):
        build_path = build_command(executed_file_relative_path, workspace_dir)
        experiment_tmp_dir = make_cleaned_up_experiment_dir(build_path, experiment_name)
        copy_to_cluster(experiment_tmp_dir, cluster_exp_main_dir)

