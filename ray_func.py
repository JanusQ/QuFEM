import ray
import threading
import inspect
import uuid
from random import random
import concurrent.futures
from concurrent.futures._base import Future
from collections.abc import Iterable
from inspect import isgeneratorfunction
from collections import defaultdict
from sklearn.utils import shuffle
import numpy as np

# 是不是需要远程执行的函数
def is_ray_func(func):
    for name, f in inspect.getmembers(func, lambda f: hasattr(f, '_remote')):
        return True
    return False

def is_ray_future(obj):
    return isinstance(obj, ray._raylet.ObjectRef)

def wait(future, show_progress = False):
    # TODO: 可能会导致循环递归
    if isinstance(future, (list, set)):
        futures = future
        
        if not show_progress: 
            return [wait(future) for future in futures]
        else:
            from tqdm import tqdm
            results = []
            for future in tqdm(futures):
                results.append(wait(future) )
            return results
    elif is_ray_future(future):
        return ray.get(future)
    elif isinstance(future, Future):
        return future.result()
    elif isinstance(future, (dict, defaultdict)):
        return {
            key: wait(item)
            for key, item in future.items()
        }
    else:
        # raise Exception(future, 'is not future type')
        return future

def batch(X, batch_size = 100):
    # print(X.shape)
    for start in range(0, len(X), batch_size):
        yield X[start: start+batch_size]
