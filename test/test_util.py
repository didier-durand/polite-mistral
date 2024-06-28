import unittest

from util import check_response


class TestUtil(unittest.TestCase):

    def setUp(self):
        print("executing test: " + self.__class__.__name__ + "." + self._testMethodName + "() ...")

    def test_check_response(self):
        response = {"ResponseMetadata": {"HTTPStatusCode": 200}}
        self.assertTrue(check_response(response))
        response = {"ResponseMetadata": {"HTTPStatusCode": 404}}
        self.assertTrue(check_response(response, 404))
        #
        response = {}
        with self.assertRaises(KeyError) as context:
            check_response(response)
        self.assertEqual("'ResponseMetadata'", str(context.exception))
        response = {"ResponseMetadata": {}}
        with self.assertRaises(KeyError) as context:
            check_response(response)
        self.assertEqual("'HTTPStatusCode'", str(context.exception))
