import os
import unittest

from openai import OpenAI
from openai.types.chat.chat_completion import Choice

from gauge_answer import POLITE_SAMPLES, GAUGE_SYS_PROMPT
from genai_model import GenAiModel
from openai_chat import openai_chat
from util import read_from_file, from_json


class TestOpenAI(unittest.TestCase):

    def setUp(self):
        print("executing test: " + self.__class__.__name__ + "." + self._testMethodName + "() ...")
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def test_list_openai_models(self):
        models = self.client.models.list()
        count: int = 0
        for model in models:
            count += 1
            print("model: " + model.id)
        print("model count: " + str(count))
        self.assertEqual(27, count)

    def test_openai_connect(self):
        model: str = GenAiModel.GPT_3_5_TURBO.value
        message: str = "what is a Large Langauge Model?"
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Say " + message,
                }
            ],
            model=model,
        )
        print(chat_completion)
        self.assertTrue(chat_completion.model.startswith(model))
        self.assertEqual("chat.completion", chat_completion.object)
        self.assertLess(0, chat_completion.usage.completion_tokens)
        self.assertLess(0, chat_completion.usage.prompt_tokens)
        self.assertEqual(chat_completion.usage.prompt_tokens + chat_completion.usage.completion_tokens,
                         chat_completion.usage.total_tokens)
        #
        self.assertEqual(1, len(chat_completion.choices))
        choice: Choice = chat_completion.choices[0]
        self.assertEqual("stop", choice.finish_reason)
        self.assertEqual(0, choice.index)
        self.assertEqual("assistant", choice.message.role)

    def test_invoke_openai_gpt(self):
        for model in [GenAiModel.GPT_3_5_TURBO, GenAiModel.GPT_4_TURBO]:
            response: str = openai_chat(model, prompt="what is a Large Langauge Model?")
            print(" --- model:", model.value, "--- response: ", response)
            self.assertGreater(len(response), 0)
            self.assertTrue("Large" in response)
            self.assertTrue("Language" in response)
            self.assertTrue("Model" in response)

    def test_invoke_openai_chat_polite_samples(self):
        polite_samples = from_json(read_from_file(POLITE_SAMPLES))
        sys_prompt = GAUGE_SYS_PROMPT
        #
        for _,item in polite_samples.items():
            prompt = item["src"]
            print("\n--- prompt: ", prompt)
            completion = openai_chat(model=GenAiModel.GPT_4_TURBO, prompt=prompt, sys_prompt=sys_prompt)
            print("--- completion: ", completion)
