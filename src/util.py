import json
import os
from datetime import datetime
from os.path import dirname, abspath
from pathlib import Path

import jsonlines
import boto3

ROOT_DIR: str = dirname(dirname(abspath(__file__)))
DATA_DIR: str = ROOT_DIR + "/data"
REF_DIR: str = DATA_DIR + "/ref"


def on_laptop() -> bool:
    return "didduran" in os.environ["HOME"]


def ts_suffix(short: bool = False) -> str:
    if short:
        return datetime.now().strftime("%m-%d-%H-%M-%S")
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def aws_account() -> str:
    return boto3.client('sts').get_caller_identity().get('Account')


def check_response(resp: dict, http_status_code=200) -> bool:
    assert resp["ResponseMetadata"]["HTTPStatusCode"] == http_status_code
    return True


def to_json(data: dict | list = None) -> str:
    return json.dumps(data, indent=4, default=str)


def to_jsonl(items: list = None, path: Path = None) :
    with jsonlines.open(path, 'w') as writer:
        writer.write_all(items)


def from_json(data: str = "") -> dict | None:
    if data != "":
        return json.loads(data)
    return None


def read_from_file(file_path: Path = None) -> str | Exception:
    assert file_path is not None
    assert file_path.is_file(), "path: " + str(file_path)
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def write_to_file(file_path: Path = None, content: str = "") -> int:
    assert file_path is not None
    with open(file_path, "w", encoding="utf-8") as file:
        rc = file.write(content)
    if rc != len(content):
        raise SystemError("incorrect length for written bytes: ",
                          str(len(content)), " <> ", str(rc))
    return rc
