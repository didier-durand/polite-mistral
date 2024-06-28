import os

from openai import OpenAI
from openai.types.chat.chat_completion import Choice

from genai_model import GenAiModel


def openai_chat(model: GenAiModel, prompt: str = "", sys_prompt: str = "") -> str:
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    match model:
        case GenAiModel.GPT_3_5_TURBO | GenAiModel.GPT_4_TURBO:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": sys_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model.value,
            )
            assert chat_completion.model.startswith(model.value)
            assert "chat.completion" == chat_completion.object
            assert 0 < chat_completion.usage.completion_tokens
            assert 0 < chat_completion.usage.prompt_tokens
            assert (chat_completion.usage.prompt_tokens
                    + chat_completion.usage.completion_tokens == chat_completion.usage.total_tokens)
            #
            assert 1 == len(chat_completion.choices)
            choice: Choice = chat_completion.choices[0]
            assert "stop" == choice.finish_reason
            assert 0 == choice.index
            assert "assistant" == choice.message.role
            return choice.message.content
        case _:
            raise ValueError("unknown openai model:", model.value)
