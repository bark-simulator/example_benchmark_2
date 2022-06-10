from pathlib import Path
import subprocess
from ruamel.yaml import YAML
import hashlib
import os
import paramiko
import time


def load_from_file(path: Path):
    yaml = YAML(typ="rt")
    with path.open(mode="r", encoding="utf-8") as file:
        return yaml.load(file)


def store_to_path(path: Path, data):
    yaml = YAML(typ="rt")
    with path.open(mode="w", encoding="utf-8") as file:
        yaml.dump(data, file)


def term(c, strip=True):
    """
    Helper function wrapping subprocess.check_output. Call it to execute terminal commands.
    """
    result = subprocess.check_output(c, shell=True).decode("utf-8")
    if strip:
        result = result.strip()
    return result


def call(c):
    subprocess.check_call(c, shell=True)


def make_hash(inpt):
    hasher = hashlib.sha1()
    if isinstance(inpt, (set, tuple, list)):
        for entry in inpt:
            hasher.update(make_hash(entry))

    elif isinstance(inpt, dict):
        for entry in sorted(inpt.keys()):
            hasher.update(make_hash(entry) + make_hash(inpt[entry]))
    elif isinstance(inpt, str):
        hasher.update(inpt.encode("utf-8"))
    elif inpt is None:
        hasher.update(b"")
    else:
        hasher.update(str(inpt).encode("utf-8"))

    return hasher.hexdigest().encode("utf-8")


def ensure_empty(path: Path, force: bool= False):
    """
    Ensures that the directory dir exists, creating it if necessary. If not forced, files existing in that
    directory will result in an exception.

    Raises
    ------
    IOError : iff the directory exists, is not empty, and force is not specified.
    """
    if path.exists():
        sub_files = [x for x in path.iterdir()]
        if sub_files:
            if not force:
                raise IOError("Specified output path, but path exists. Call it with -f or with another path.")
            else:
                subprocess.check_output("rm -rf %s" % path, shell=True)
    if not path.exists():
        path.mkdir(parents=True)

#
# k = paramiko.RSAKey.from_private_key_file("/home/julo/.ssh/id_rsa")
#>>> c = paramiko.SSHClient()
#>>> c = paramiko.SSHClient()
#c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#>>> c.connect( hostname = "8gpu", username = "bernhard", pkey = k )
#>>> c.exec_command("ls")
#
#
#

class CachedClient:
    def __init__(self, server, cluster_key=None):
        self.client = paramiko.SSHClient()
        self.client.load_system_host_keys()
        if cluster_key is None:
            try:
                cluster_key = os.path.expanduser(os.environ["LABPAGE_CLUSTER_KEY"])
            except KeyError:
                raise ValueError("No cluster key specified and environment variable LABPAGE_CLUSTER_KEY not set."
                                 " One of these should be set to your cluster ssh key."
                                 " Recommend adding 'export LABPAGE_CLUSTER_KEY=<yourkeypath>' to bashrc "
                                 "(don't forget to source)")
        self.client.connect(server, key_filename=cluster_key)
        self.cache = {}

    def exec_command(self, command):
        return self.client.exec_command(command)

    def get_stdout(self, command, cache_age=None):
        if cache_age is None or command not in self.cache or time.time() - self.cache[command]["time"] > cache_age:
            _, stdout, _ = self.client.exec_command(command)
            result = stdout.read().strip().decode("ascii")
            # cache miss
            self.cache[command] = {
                "result": result,
                "time": time.time()
            }
        return self.cache[command]["result"]


def create_new_connection(server, cluster_key=None):
    return CachedClient(server, cluster_key)


known_connections = {}


def create_connection(server, cluster_key=None):
    if server not in known_connections:
        known_connections[server] = create_new_connection(server, cluster_key)
    return known_connections[server]

