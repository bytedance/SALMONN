from .salmonn import SALMONN


def load_model(config):
    return SALMONN.from_config(config)