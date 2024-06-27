import unittest

from bedrock import embed, EMBEDDING_LENGTH, GenAiModel
from util import on_laptop


class TestBedrock(unittest.TestCase):

    def setUp(self):
        print("\n\nexecuting test: "
              + self.__class__.__name__ + "." + self._testMethodName + "() ...")

    @unittest.skipIf(not on_laptop(), "skipped when not on laptop")
    def test_titan_embedding(self):
        embedding = embed(GenAiModel.TITAN_EMBED_TEXT_V2_0, "Mistral is an Open Source LLM")
        print(embedding)
        self.assertIsInstance(embedding, list)
        self.assertEqual(EMBEDDING_LENGTH, len(embedding))
        for dim in embedding:
            self.assertIsInstance(dim, float)

    @unittest.skipIf(not on_laptop(), "skipped when not on laptop")
    def test_cohere_embedding(self):
        embedding = embed(GenAiModel.COHERE_EMBED_ENGLISH_V3, "Mistral is an Open Source LLM")
        print(embedding)
        self.assertIsInstance(embedding, list)
        self.assertEqual(EMBEDDING_LENGTH, len(embedding))
        for dim in embedding:
            self.assertIsInstance(dim, float)
