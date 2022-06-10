
import collections
import jsonpickle
import json
import os

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def picklable_object_to_dict(picklable_object):
    dcts = jsonpickle.encode(picklable_object, unpicklable=False, make_refs=False)
    dct = json.loads(dcts)
    return dct


def get_ray_init_config():
  if "slurm_num_cpus" in os.environ:
    num_cpus = int(os.environ["slurm_num_cpus"])
    memory = int(os.environ["slurm_memory"])*1000*1024*1024
    object_story_memory = int(os.environ["slurm_object_memory"])*1000*1024*1024
    ray_session_dir = "/mnt/glusterdata/home/bernhard/ray_sessions"
  else:
    memory = 32*1000*1024*1024 # 32gb
    num_cpus=6
    object_story_memory = 10*1000*1024*1024 # 30 gb
    home_dir = os.path.expanduser("~")
    ray_session_dir = os.path.join(home_dir, "ray_sessions")

  ray_init_args = {
  "num_cpus" : num_cpus, 
  "memory" : memory,
  "object_store_memory" : object_story_memory,
  "_internal_config" : '{"initial_reconstruction_timeout_milliseconds": 10000000}',
  "num_redis_shards" : 4}
  return ray_init_args