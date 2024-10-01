import yaml
from pathlib import Path
from typing import Optional
import logging
import boto3
import json
from fnmatch import fnmatch
from decimal import Decimal
import numpy


def get_yaml_config(file_path: Path) -> Optional[dict]:
    """Fetch yaml config and return as dict if it exists."""
    if file_path.exists():
        with open(file_path, "rt") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)


# Define project base directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

logger = logging.getLogger(__name__)

soc_mapper_config = get_yaml_config(PROJECT_DIR / "nlp_link/soc_mapper/config.yaml")


def get_s3_resource():
    s3 = boto3.resource("s3")
    return s3


def load_s3_json(bucket_name, file_name):
    """
    Load data from S3 location.

    bucket_name: The S3 bucket name
    file_name: S3 key to load
    """
    s3 = get_s3_resource()

    obj = s3.Object(bucket_name, file_name)

    if fnmatch(file_name, "*.json"):
        file = obj.get()["Body"].read().decode()
        return json.loads(file)
    else:
        logger.error(f'{file_name} has wrong file extension! Only supports "*.json"')


def load_local_json(file_name: str) -> dict:
    """Loads a dict stored in a json file from path.
    Args:
            file_name (str): Local path to json.
    Returns:
            file (dict): Loaded dict
    """
    if fnmatch(file_name, "*.json"):
        with open(file_name, "r") as file:
            return json.load(file)
    else:
        logger.error(f'{file_name} has wrong file extension! Only supports "*.json"')


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        return super(CustomJsonEncoder, self).default(obj)


def save_to_s3(bucket_name, output_var, output_file_dir):
    s3 = get_s3_resource()

    obj = s3.Object(bucket_name, output_file_dir)

    if fnmatch(output_file_dir, "*.csv"):
        output_var.to_csv("s3://" + bucket_name + "/" + output_file_dir, index=False)
    elif fnmatch(output_file_dir, "*.parquet"):
        output_var.to_parquet(
            "s3://" + bucket_name + "/" + output_file_dir, index=False
        )
    elif fnmatch(output_file_dir, "*.txt"):
        obj.put(Body=output_var)
    else:
        obj.put(Body=json.dumps(output_var, cls=CustomJsonEncoder))

    logger.info(f"Saved to s3://{bucket_name} + {output_file_dir} ...")


def save_json_dict(dictionary: dict, file_name: str):
    """Saves a dict to a json file.

    Args:
            dictionary (dict): The dictionary to be saved
            file_name (str): Local path to json.
    """
    if fnmatch(file_name, "*.json"):
        with open(file_name, "w") as file:
            json.dump(dictionary, file, cls=CustomJsonEncoder)
        logger.info(f"Saved to {file_name} ...")
    else:
        logger.error(f'{file_name} has wrong file extension! Only supports "*.json"')
