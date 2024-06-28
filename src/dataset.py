from pathlib import Path

from datasets import load_dataset, Dataset
from sagemaker.s3 import S3Uploader

from genai_model import GenAiModel
from embed import embed_dataset
from training_sm import AWS_LLM_TRAINING
from util import DATA_DIR, write_to_file, to_json, on_laptop, to_jsonl

POLITE = "polite_rewrite"
DS_POLITE = "jdustinwind/Polite"
DS_POLITE_SIZE = 100_000
DS_POLITE_TEST_SIZE = 1000
DS_POLITE_TRAIN_SIZE = 60_000
DS_POLITE_SM_TEST_PATH = Path(DATA_DIR, POLITE, "sm_test.jsonl")
DS_POLITE_SM_TRAIN_PATH = Path(DATA_DIR, POLITE, "sm_train.jsonl")
DS_POLITE_MISTRAL_TRAIN_PATH = Path(DATA_DIR, POLITE, "mistral_train.jsonl")
DS_POLITE_MISTRAL_TEST_PATH = Path(DATA_DIR, POLITE, "mistral_test.jsonl")
DS_POLITE_AUTOML_TRAIN_CSV_PATH = Path(DATA_DIR, POLITE, "automl_train.csv")
DS_POLITE_REF_PATH = Path(DATA_DIR, POLITE, POLITE + ".json")
DS_POLITE_EMBED_PATH = Path(DATA_DIR, POLITE, POLITE + "_embed.json")
TEMPLATE_POLITE_PATH = Path(DATA_DIR, POLITE, "template.json")

S3_POLITE_FOLDER = f"s3://{AWS_LLM_TRAINING}/{POLITE}"

SIMILARITY_SCORE = "similarity_score"
DISTANCE = "distance"
COSINE = "cosine"

STSB = "stsb_multi_mt"
DS_STSB = STSB
DS_STSB_SIZE = 5_749
DS_STSB_TEST_SIZE = 5_000
DS_STSB_TRAIN_SIZE = 749
DS_STSB_REF_PATH = Path(DATA_DIR, STSB, STSB + ".json")
DS_STSB_EMBED_PATH = Path(DATA_DIR, STSB, STSB + "_embed.json")
DS_STSB_MULTI_SCORE: Path = Path(DATA_DIR, STSB + "_score.json")


def load_hf_datasets(ds_name: str = "", ds_len: int = 0,
                     test_size: int = 0, train_size: int = 0,
                     config: str = "default"):
    polite_dataset = load_dataset(ds_name, config, split="train")
    print("dataset:", polite_dataset)
    assert ds_len == polite_dataset.num_rows, "num rows:" + str(polite_dataset.num_rows)
    assert ds_len == len(polite_dataset), "num rows:" + str(polite_dataset.num_rows)
    return polite_dataset.train_test_split(test_size=test_size, train_size=train_size)


def prepare_polite_dataset_for_training() -> tuple[Dataset, Dataset, str]:
    polite_datasets = load_hf_datasets(ds_name=DS_POLITE,
                                       ds_len=DS_POLITE_SIZE,
                                       test_size=DS_POLITE_TEST_SIZE,
                                       train_size=DS_POLITE_TRAIN_SIZE)
    # prepare data for Mistral Plateforme
    items: list = []
    for item in polite_datasets["train"]:
        mistral_dict = {"messages":
            [
                {"role": "system",
                 "content": "the user will provide you with impolite, disrespectful or offensive input."
                            "Please, reformulate in a polite and respectful form"},
                {"role": "user",
                 "content": item["src"]},
                {"role": "assistant",
                 "content": item["tgt"]},
            ]
        }
        items.append(mistral_dict)
    to_jsonl(items, DS_POLITE_MISTRAL_TRAIN_PATH)
    items: list = []
    for item in polite_datasets["test"]:
        mistral_dict = {"messages":
            [
                {"role": "system",
                 "content": "the user will provide you with impolite, disrespectful or offensive input."
                            "Please, reformulate in a polite and respectful form"},
                {"role": "user",
                 "content": item["src"]},
                {"role": "assistant",
                 "content": item["tgt"]},
            ]
        }
        items.append(mistral_dict)
    to_jsonl(items, DS_POLITE_MISTRAL_TEST_PATH)
    # prepare data for SM
    polite_datasets["test"].to_json(DS_POLITE_SM_TEST_PATH)
    polite_datasets["train"].to_json(DS_POLITE_SM_TRAIN_PATH)
    # prepare template for SM
    template = {
        "prompt": "Below is an input sentence in a form that is not polite.\n"
                  "Write a response that appropriately changes this sentence to a polite form.\n"
                  "Change all impolite, aggressive or offending words to polite words showing respect.\n"
                  "Do NOT add any other word than the input sentence translated to polite form.\n"
                  "### Input Sentence:\n{src}\n\n",
        "completion": "{tgt}",
    }
    write_to_file(TEMPLATE_POLITE_PATH, to_json(template))
    # csv for SM AutoML
    csv: str = "source,target\n"
    for item in polite_datasets["test"]:
        csv += item["src"].replace(",", " ") + "," + item["tgt"].replace(",", " ") + "\n"
    write_to_file(DS_POLITE_AUTOML_TRAIN_CSV_PATH, csv)
    # upload to S3 for training
    if on_laptop():
        S3Uploader.upload(str(TEMPLATE_POLITE_PATH), S3_POLITE_FOLDER)
        S3Uploader.upload(str(DS_POLITE_SM_TRAIN_PATH), S3_POLITE_FOLDER)
        S3Uploader.upload(str(DS_POLITE_AUTOML_TRAIN_CSV_PATH), S3_POLITE_FOLDER)
    return polite_datasets["test"], polite_datasets["train"], csv


def prepare_polite_dataset_for_metrics() -> dict:
    polite_datasets = load_hf_datasets(ds_name=DS_POLITE,
                                       ds_len=DS_POLITE_SIZE,
                                       test_size=DS_POLITE_TEST_SIZE,
                                       train_size=DS_POLITE_TRAIN_SIZE)
    ds_dict = {}
    count = 0
    for item in polite_datasets["test"]:
        count += 1
        ds_dict[str(count)] = item
    write_to_file(DS_POLITE_REF_PATH, to_json(ds_dict))
    return ds_dict


def prepare_stsb_dataset_for_metrics() -> dict:
    polite_datasets = load_hf_datasets(ds_name=DS_STSB,
                                       config="en",
                                       ds_len=DS_STSB_SIZE,
                                       test_size=DS_STSB_TEST_SIZE,
                                       train_size=DS_STSB_TRAIN_SIZE)
    ds_dict = {}
    count = 0
    for item in polite_datasets["test"]:
        count += 1
        ds_dict[str(count)] = item
    write_to_file(DS_STSB_REF_PATH, to_json(ds_dict))
    return ds_dict


def embed_polite_dataset() -> bool:
    embed_dataset(dataset=DS_POLITE_EMBED_PATH,
                  embedding_engines=[
                      GenAiModel.TITAN_EMBED_TEXT_V2_0,
                      GenAiModel.COHERE_EMBED_ENGLISH_V3,
                      GenAiModel.MISTRAL_EMBED,
                  ],
                  max_count=DS_POLITE_TEST_SIZE
                  )
    return True


def embed_stsb_dataset() -> bool:
    embed_dataset(dataset=DS_STSB_EMBED_PATH,
                  embedding_engines=[
                      GenAiModel.TITAN_EMBED_TEXT_V2_0,
                      GenAiModel.COHERE_EMBED_ENGLISH_V3,
                      GenAiModel.MISTRAL_EMBED,
                  ],
                  max_count=DS_STSB_TEST_SIZE
                  )
    return True
