import os
from pathlib import Path

from mistralai.client import MistralClient

from bedrock import embed
from genai_model import GenAiModel
from util import read_from_file, from_json, write_to_file, to_json

CHECKPOINT = 100

EMBEDDING = "embedding"


def embed_dataset(dataset: Path, embedding_engines: list[GenAiModel] = None, max_count=1_000_000) -> dict:
    assert dataset.is_file(), str(dataset) + " does not exist"
    assert len(embedding_engines) > 0, "no embedding engines provided"
    dict_stsb = from_json(read_from_file(dataset))
    changed: bool = False
    for embedding_engine in embedding_engines:
        count = 0
        for item in dict_stsb.values():
            count += 1
            if count > max_count:
                break
            if EMBEDDING not in item:
                item[EMBEDDING] = {}
            if embedding_engine.value not in item[EMBEDDING]:
                item[EMBEDDING][embedding_engine.value] = {}
            item: dict = item
            for key in item.keys():
                if isinstance(item[key], (float, int)):  # for already existing metrics in the dataset
                    if key not in item[EMBEDDING]:
                        item[EMBEDDING][key] = item[key]
                        changed = True
                if isinstance(item[key], str):
                    if key not in item[EMBEDDING][embedding_engine.value]:
                        if embedding_engine == GenAiModel.MISTRAL_EMBED:
                            client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
                            embeddings_response = client.embeddings(
                                model=GenAiModel.MISTRAL_EMBED.value,
                                input=item[key]
                            )
                            item[EMBEDDING][embedding_engine.value][key] = embeddings_response.data[0].embedding
                        else:
                            item[EMBEDDING][embedding_engine.value][key] = (
                                embed(embedding_engine, item[key]))
                        changed = True
            if changed and count % CHECKPOINT == 0:
                print("checkpoint: " + str(count) + " - embedding engine: " + str(embedding_engine.value))
                write_to_file(dataset, to_json(dict_stsb))
                changed = False
    write_to_file(dataset, to_json(dict_stsb))
    return dict_stsb
