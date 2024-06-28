import os

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from genai_model import GenAiModel


def mistral_chat(model: GenAiModel, prompt: str = "", sys_prompt: str = "") -> str:
    api_key = os.environ["MISTRAL_API_KEY"]
    client = MistralClient(api_key=api_key)
    chat_response = client.chat(
        model=model.value,
        messages=[
            ChatMessage(role="system", content=sys_prompt),
            ChatMessage(role="user", content=prompt)
        ],
    )
    content = chat_response.choices[0].message.content
    assert isinstance(content, str)
    assert len(content) > 0
    return content
