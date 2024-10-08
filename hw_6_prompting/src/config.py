from functools import lru_cache
from typing import Dict

import yaml


def parse_config(config_path: str) -> Dict:
    """
    :param config_path:
    :return:
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


@lru_cache
def get_config(config_path: str):
    config = parse_config(config_path)
    return config
