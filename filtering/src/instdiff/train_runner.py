from typing import Dict, Any

from .utils import set_seed


def run_train(config: Dict[str, Any]) -> None:
    from instdiff.train_v2 import train as train_fn

    seed = config.get("seed", 42)
    set_seed(seed)

    kwargs = dict(config.get("train", {}))
    train_fn(**kwargs)
