import os
import unittest
from pathlib import Path


from dataset import prepare_polite_dataset_for_training, DS_POLITE_TEST_SIZE, DS_POLITE_TRAIN_SIZE, \
    prepare_polite_dataset_for_metrics, POLITE, embed_polite_dataset, STSB, \
    prepare_stsb_dataset_for_metrics, DS_STSB_TEST_SIZE, embed_stsb_dataset
from util import DATA_DIR, on_laptop


class TestDataset(unittest.TestCase):

    def setUp(self):
        print("executing test: " + self.__class__.__name__ + "." + self._testMethodName + "() ...")
        data_dir = Path(DATA_DIR)
        if not data_dir.is_dir():
            print("creating data dir: ", data_dir)
            os.mkdir(str(data_dir))
        polite_dir = Path(data_dir, POLITE)
        if not polite_dir.is_dir():
            print("creating polite dir: ", polite_dir)
            os.mkdir(str(polite_dir))
        stsb_dir = Path(data_dir, STSB)
        if not stsb_dir.is_dir():
            print("creating polite dir: ", stsb_dir)
            os.mkdir(str(stsb_dir))

    def test_prepare_polite_dataset_for_training(self):
        test_dataset, train_dataset, csv = prepare_polite_dataset_for_training()
        self.assertEqual(DS_POLITE_TEST_SIZE, test_dataset.num_rows)
        self.assertEqual(DS_POLITE_TRAIN_SIZE, train_dataset.num_rows)
        self.assertEqual(DS_POLITE_TEST_SIZE + 2, len(csv.split("\n")))

    def test_prepare_polite_dataset_for_metrics(self):
        ds_dict = prepare_polite_dataset_for_metrics()
        self.assertEqual(DS_POLITE_TEST_SIZE, len(ds_dict))

    def test_prepare_stsb_dataset_for_metrics(self):
        ds_dict = prepare_stsb_dataset_for_metrics()
        self.assertEqual(DS_STSB_TEST_SIZE, len(ds_dict))

    @unittest.skipIf(not on_laptop(), "skipped when not on laptop")
    def test_embed_polite_dataset(self):
        self.assertTrue(embed_polite_dataset())

    @unittest.skipIf(not on_laptop(), "skipped when not on laptop")
    def test_embed_stsb_dataset(self):
        self.assertTrue(embed_stsb_dataset())
