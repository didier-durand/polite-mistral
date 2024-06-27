import json
from enum import Enum

import boto3

from util import check_response

EMBEDDING_LENGTH: int = 1024


class GenAiModel(Enum):
    # Mistral Platform
    PLATFORM_MISTRAL = "platform.mistral-embed"
    # Hugging Face
    HF_MISTRAL_7B = "huggingface-llm-mistral-7b"
    HF_GEMMA_7B = "huggingface-llm-gemma-7b-instruct"
    # Bedrock
    TITAN_EMBED_TEXT_V2_0 = "amazon.titan-embed-text-v2:0"
    COHERE_EMBED_ENGLISH_V3 = "cohere.embed-english-v3"

def embed(model_id: GenAiModel, input_text: str = "") -> list[float] | Exception:
    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime',
        # region_name='us-west-2',
    )
    if model_id == GenAiModel.TITAN_EMBED_TEXT_V2_0:
        body = json.dumps({
            "inputText": input_text,
        })
    elif model_id == GenAiModel.COHERE_EMBED_ENGLISH_V3:
        body = json.dumps({
            "texts": [input_text],
            "input_type": "clustering",
            "truncate": "NONE"
        })
    else:
        raise ValueError("model unknown for embedding: " + model_id.value)
    accept = 'application/json'
    content_type = 'application/json'
    try:
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId=model_id.value,
            accept=accept,
            contentType=content_type
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        print("EXCEPTION:", type(e), " - model:", model_id.value, " - msg:", str(e))
        return e
    check_response(response)
    response_body = json.loads(response['body'].read())
    if model_id == GenAiModel.TITAN_EMBED_TEXT_V2_0:
        embedding = response_body.get('embedding')
    elif model_id == GenAiModel.COHERE_EMBED_ENGLISH_V3:
        embedding = response_body.get('embeddings')[0]
    else:
        raise ValueError("model unknown for embedding: " + model_id.value)
    assert isinstance(embedding, list)
    assert EMBEDDING_LENGTH == len(embedding), (
            "len(embedding) = " + str(len(embedding)) + " :" + str(embedding))
    return embedding
