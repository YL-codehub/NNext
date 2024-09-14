from box import Box
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

CFG_DIRECTORY = Path(__file__).parent



def load_yaml(file_path):
    """Load a YAML file and return its content as a dictionary."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def get_cfg(filename):
    return Box(load_yaml(CFG_DIRECTORY / filename))
