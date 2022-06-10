import ray
import os
import logging

@ray.remote
class Actor:
    def run(self):
        return 42

if "slurm_num_cpus" in os.environ:
    num_cpus = int(os.environ["slurm_num_cpus"])
    memory = int(os.environ["slurm_memory"])*1000*1024*1024
    logging.info("Cpus={} and memory={} based on environment variables.".format(num_cpus, memory))
else:
    memory = 16*1000*1024*1024 # 32gb
    num_cpus=12


ray.init(num_cpus=num_cpus, memory=memory*0.3, object_store_memory=memory*0.7, _internal_config='{"initial_reconstruction_timeout_milliseconds": 100000}') # we split memory between workers (30%) and objects (70%)
actors = [Actor.remote() for _ in range(20)]
ray.get([a.run.remote() for a in actors])