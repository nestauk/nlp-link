import yaml
from pathlib import Path
from typing import Optional
import logging


def get_yaml_config(file_path: Path) -> Optional[dict]:
    """Fetch yaml config and return as dict if it exists."""
    if file_path.exists():
        with open(file_path, "rt") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)


# Define project base directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

logger = logging.getLogger(__name__)

soc_mapper_config = get_yaml_config(PROJECT_DIR / "nlp_link/soc_mapper/config.yaml")
