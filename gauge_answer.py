import os
from math import sqrt
from pathlib import Path

from mistralai.client import MistralClient

from bedrock import embed
from genai_model import GenAiModel
from metric import cosine_similarity
from mistral_chat import mistral_chat
from openai_chat import openai_chat
from util import DATA_DIR, from_json, read_from_file

POLITE_SAMPLES: Path = Path(DATA_DIR, "polite_samples.json")

GAUGE_SYS_PROMPT: str = ("turn the content provided by the user to a polite and respectful form. "
                         "Don't add any other word.")


def gauge_answer(src: str, sys_prompt: str) -> float:
    print(" --- src: ", src)
    mistral_client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
    openai_answer = openai_chat(model=GenAiModel.GPT_4_TURBO, prompt=src, sys_prompt=sys_prompt)
    print(" --- openai: ", openai_answer)
    mistral_answer = mistral_chat(model=GenAiModel.MISTRAL_FINE_TUNED, prompt=src, sys_prompt=sys_prompt)
    print(" --- mistral: ", mistral_answer)
    #
    cohere_openai = embed(GenAiModel.COHERE_EMBED_ENGLISH_V3, openai_answer)
    titan_openai = embed(GenAiModel.TITAN_EMBED_TEXT_V2_0, openai_answer)
    mistral_openai = mistral_client.embeddings(
        model=GenAiModel.MISTRAL_EMBED.value,
        input=openai_answer,
    ).data[0].embedding
    #
    cohere_mistral = embed(GenAiModel.COHERE_EMBED_ENGLISH_V3, mistral_answer)
    titan_mistral = embed(GenAiModel.TITAN_EMBED_TEXT_V2_0, mistral_answer)
    mistral_mistral = mistral_client.embeddings(
        model=GenAiModel.MISTRAL_EMBED.value,
        input=openai_answer,
    ).data[0].embedding
    #
    mistral_cs = cosine_similarity(mistral_mistral, mistral_openai)
    print("--- mistral cs:", mistral_cs)
    cohere_cs = cosine_similarity(cohere_mistral, cohere_openai)
    print("--- cohere cs:", cohere_cs)
    titan_cs = cosine_similarity(titan_mistral, titan_openai)
    print("--- titan cs:", titan_cs)
    #
    distance = sqrt(pow(1 - mistral_cs, 2) + pow(1 - cohere_cs, 2) + pow(1 - titan_cs, 2)) / 2 * sqrt(3)
    confidence = 1 - distance
    print("confidence:", confidence)
    return confidence


def gauge_answers(samples: Path = POLITE_SAMPLES, sys_prompt: str = GAUGE_SYS_PROMPT):
    samples = from_json(read_from_file(samples))
    for _, item in samples.items():
        gauge_answer(src=item["src"], sys_prompt=sys_prompt)


if __name__ == "__main__":
    gauge_answers(samples=POLITE_SAMPLES, sys_prompt=GAUGE_SYS_PROMPT)
