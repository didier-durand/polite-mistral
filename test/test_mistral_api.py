import os
import unittest
from datetime import datetime, timedelta
from time import sleep

import requests
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from mistralai.models.embeddings import EmbeddingResponse
from mistralai.models.files import Files
from mistralai.models.jobs import TrainingParameters
from mistralai.models.models import ModelCard

from dataset import prepare_polite_dataset_for_training, DS_POLITE_TEST_SIZE, DS_POLITE_TRAIN_SIZE, \
    DS_POLITE_MISTRAL_TRAIN_PATH, DS_POLITE_MISTRAL_TEST_PATH
from gauge_answer import GAUGE_SYS_PROMPT, POLITE_SAMPLES
from genai_model import GenAiModel
from mistral_chat import mistral_chat
from util import DATA_DIR, ts_suffix, on_laptop, from_json, read_from_file


class TestMistralApi(unittest.TestCase):
    def setUp(self):
        print("executing test: " + self.__class__.__name__ + "." + self._testMethodName + "() ...")
        self.api_key = os.environ["MISTRAL_API_KEY"]
        self.assertGreater(len(self.api_key), 0, "api key is not set")
        self.client = MistralClient(api_key=self.api_key)

    @unittest.skipIf(not on_laptop(), "skipped when not on laptop")
    def test_fine_tune_mistral_on_plateforme(self):
        dry_run = False
        training_steps = 60
        learning_rate = 0.0001
        #
        test_dataset, train_dataset, csv = prepare_polite_dataset_for_training()
        self.assertEqual(DS_POLITE_TEST_SIZE, test_dataset.num_rows)
        self.assertEqual(DS_POLITE_TRAIN_SIZE, train_dataset.num_rows)
        self.assertEqual(DS_POLITE_TEST_SIZE + 2, len(csv.split("\n")))
        with open(DS_POLITE_MISTRAL_TRAIN_PATH, "rb") as train_file:
            mistral_train_file = self.client.files.create(
                purpose="fine-tune",
                file=("polite_training.jsonl", train_file.read())
            )
        print(mistral_train_file)
        with open(DS_POLITE_MISTRAL_TEST_PATH, "rb") as test_file:
            mistral_test_file = self.client.files.create(
                purpose="fine-tune",
                file=("polite_testing.jsonl", test_file.read())
            )
        print(mistral_test_file)
        created_job = self.client.jobs.create(
            model="open-mistral-7b",
            training_files=[mistral_train_file.id],
            validation_files=[mistral_test_file.id],
            integrations=[
                {
                    "type": "wandb",
                    "project": "test-fine-tune-mistral",
                    "name": "polite-ft-job-" + ts_suffix(),
                    "api_key": os.environ["WANDB_API_KEY"]
                }
            ],
            hyperparameters=TrainingParameters(
                training_steps=training_steps,
                learning_rate=learning_rate,
            ),
            dry_run=dry_run
        )
        print(created_job)
        if not dry_run:
            while created_job.status not in ["SUCCESS", "FAILED"]:
                sleep(2)
                created_job = self.client.jobs.retrieve(created_job.id)
                print("status: ", created_job.status, "created job:", created_job.id)
        # cleanup
        self.client.files.delete(mistral_train_file.id)
        self.client.files.delete(mistral_test_file.id)

    def test_invoke_mistral_chat_polite_samples(self):
        polite_samples = from_json(read_from_file(POLITE_SAMPLES))
        sys_prompt = GAUGE_SYS_PROMPT
        #
        for _, item in polite_samples.items():
            prompt = item["src"]
            print("\n--- prompt: ", prompt)
            completion = mistral_chat(model=GenAiModel.MISTRAL_7B, prompt=prompt, sys_prompt=sys_prompt)
            print("--- completion: ", completion)

    def test_list_models(self):
        delete_ft: bool = False
        model_count = 0
        list_models_response = self.client.list_models()
        self.assertEqual(type(list_models_response.data), list)
        for model_card in list_models_response.data:
            print(model_card)
            self.assertIsInstance(model_card, ModelCard)
            if not model_card.id.startswith("ft:"):
                model_count += 1
            else:
                if delete_ft:
                    url = "https://api.mistral.ai/v1/models/" + model_card.id
                    print("deleting model:", url)
                    response = requests.delete(url,
                                               headers={"x-api-key": self.api_key},
                                               timeout=10)
                    self.assertEqual(response.status_code, 200)
        self.assertEqual(18, model_count, "standard models: " + str(model_count))

    def test_embedding(self):
        embeddings_response: EmbeddingResponse = self.client.embeddings(
            model="mistral-embed",
            input="Mistral creates Open Source Large Language Models"
        )
        print(embeddings_response)
        self.assertEqual(embeddings_response.model, "mistral-embed")
        self.assertGreater(len(embeddings_response.id), 0)
        self.assertGreater(embeddings_response.usage.prompt_tokens, 0)
        self.assertEqual(0, embeddings_response.usage.completion_tokens)
        self.assertEqual(type(embeddings_response.data), list)
        self.assertEqual(0, embeddings_response.data[0].index)
        self.assertEqual("embedding", embeddings_response.data[0].object)
        self.assertEqual(1024, len(embeddings_response.data[0].embedding))

    def test_chat(self):
        model = "mistral-tiny"
        chat_response = self.client.chat(
            model=model,
            messages=[ChatMessage(role="user", content="What is a Large Language Model?")],
        )
        print(chat_response.choices[0].message.content)
        content = chat_response.choices[0].message.content
        self.assertIs(type(content), str)
        self.assertGreater(len(content), 0)
        self.assertTrue("Large" in content)
        self.assertTrue("Language" in content)
        self.assertTrue("Model" in content)

    def test_file(self):
        full_delete: bool = True
        files = self.client.files.list()
        print(files)
        self.assertIs(type(files), Files)
        #
        if len(files.data) > 0:
            print("existing files:", len(files.data))
            for file in files.data:
                if not full_delete:
                    if file.filename.startswith("test_"):
                        print("deleting file:", file.id, " - ", file.filename)
                        self.client.files.delete(file.id)
                else:
                    print("deleting file:", file.id, " - ", file.filename)
                    self.client.files.delete(file.id)
        else:
            print("no existing files")
        #
        with open(DATA_DIR + "/test_training_file.jsonl", "rb") as test_file:
            created_file = self.client.files.create(file=("test_training_file.jsonl",
                                                          test_file.read()))
        print(created_file)
        self.client.files.delete(created_file.id)

    def test_job(self):
        dry_run = True
        started = datetime.now() - timedelta(seconds=10)
        #
        with open(DATA_DIR + "/test_training_file.jsonl", "rb") as f:
            training_file = self.client.files.create(file=f)
        with open(DATA_DIR + "/test_validation_file.jsonl", "rb") as f:
            validation_file = self.client.files.create(file=f)
        # Create a new job (dry run)
        created_job = self.client.jobs.create(
            model="open-mistral-7b",
            training_files=[training_file.id],
            validation_files=[validation_file.id],
            integrations=[{
                "type": "wandb",
                "project": "test-fine-tune-mistral",
                "name": "tuning-job-" + ts_suffix(),
                "api_key": "2dde446ae5865f42bccf132a1fbe9031f986942e"
            }
            ],
            hyperparameters=TrainingParameters(
                training_steps=1,
                learning_rate=0.0001,
            ),
            dry_run=dry_run
        )
        print(created_job)
        if dry_run:
            self.assertEqual('job.metadata', created_job.object)
            self.assertEqual(1, created_job.training_steps)
            self.assertGreaterEqual(created_job.expected_duration_seconds, 0)
        else:
            self.assertEqual('job', created_job.object)
        #
        jobs = self.client.jobs.list(created_after=started)
        print(jobs)

    def test_list_jobs(self):
        jobs = self.client.jobs.list()
        print(jobs)
