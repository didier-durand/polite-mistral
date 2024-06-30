import unittest

from gauge_answer import gauge_answer, GAUGE_SYS_PROMPT
from util import on_laptop


class TestGaugeAnswer(unittest.TestCase):
    def setUp(self):
        print("executing test: " + self.__class__.__name__ + "." + self._testMethodName + "() ...")

    @unittest.skipIf(not on_laptop(), "skipped when not on laptop")
    def test_gauge_answer(self):
        confidence = gauge_answer(src="WTF was Rob thinking?", sys_prompt=GAUGE_SYS_PROMPT)
        print("confidence: ", confidence)
