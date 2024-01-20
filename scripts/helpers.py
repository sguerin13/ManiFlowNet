import json
from types import SimpleNamespace

def load_config(path):
    with open(path) as f:
        config = json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d))
    return config