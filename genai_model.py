from enum import Enum


class GenAiModel(Enum):
    # Mistral Platform
    MISTRAL_7B = "open-mistral-7b"
    MISTRAL_TINY = "mistral-tiny"
    MISTRAL_EMBED = "mistral-embed"
    # fine-tuned model with 60K samples over 60 steps (June 28th)
    MISTRAL_FINE_TUNED = "ft:open-mistral-7b:48d56ecd:20240628:e31fa6b7"
    # Hugging Face
    HF_MISTRAL_7B = "huggingface-llm-mistral-7b"
    HF_GEMMA_7B = "huggingface-llm-gemma-7b-instruct"
    # OpenAI
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4_TURBO = "gpt-4-turbo"
    # Bedrock
    TITAN_EMBED_TEXT_V2_0 = "amazon.titan-embed-text-v2:0"
    COHERE_EMBED_ENGLISH_V3 = "cohere.embed-english-v3"
