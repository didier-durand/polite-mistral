import unittest
import gdown

from dataset import DS_POLITE_EMBED_PATH, DS_STSB_EMBED_PATH

from metric import distance, cosine_similarity, correlation, correlate_embeddings
from util import to_json


class TestUtils(unittest.TestCase):

    def setUp(self):
        print("executing test: " + self.__class__.__name__ + "." + self._testMethodName + "() ...")

    def test_distance(self):
        a = [1, 2, 3]
        b = [1, 2, 3]
        self.assertEqual(0, distance(a, b))
        a = [0, 0]
        b = [1, 1]
        self.assertAlmostEqual(1.41421, distance(a, b), places=5)
        a = [0, 0, 0]
        b = [1, 1, 1]
        self.assertAlmostEqual(1.73205, distance(a, b), places=5)
        a = [0, 0, 0, 0]
        b = [1, 1, 1, 1]
        self.assertEqual(2, distance(a, b))

    def test_cosine_similarity(self):
        a = [1, 1, 1]
        b = [1, 1, 1]
        self.assertAlmostEqual(1.0, cosine_similarity(a, b), places=8)
        a = [1, 1, 1]
        b = [0.5, 0.5, 0.5]
        self.assertAlmostEqual(1.0, cosine_similarity(a, b), places=8)
        a = [0, 2, 0]
        b = [1, 0, 3]
        self.assertAlmostEqual(0.0, cosine_similarity(a, b), places=8)

    def test_correlation(self):
        a = [1.0, 0.8, 1.0]
        b = [1.0, 0.9, 1.0]
        self.assertEqual(1, correlation(a, b))
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        self.assertAlmostEqual(-1.0, correlation(a, b))

    def test_polite_correlation(self):
        gdrive_url = "https://drive.google.com/file/d/1ldKhELChU1_X18Gg-VbPL24PKImm9GVy"
        gdrive_id = gdrive_url.rsplit("/", maxsplit=1)[-1]
        dataset = DS_POLITE_EMBED_PATH
        if not dataset.is_file():
            print("transferring dataset: " + str(dataset), " - from: ", gdrive_url)
            gdown.download(id=gdrive_id, output=str(dataset), quiet=False)
        eng_correlation: dict = correlate_embeddings(dataset)
        print(to_json(eng_correlation))

    def test_stsb_correlation(self):
        gdrive_url = "https://drive.google.com/file/d/1WpdWpuZjVF0XodqWYajxi8ra2un2vJy8"
        gdrive_id = gdrive_url.rsplit("/", maxsplit=1)[-1]
        dataset = DS_STSB_EMBED_PATH
        if not dataset.is_file():
            print("transferring dataset: " + str(dataset), " - from: ", gdrive_url)
            gdown.download(id=gdrive_id, output=str(dataset), quiet=False)
        eng_correlation: dict = correlate_embeddings(dataset)
        print(to_json(eng_correlation))
