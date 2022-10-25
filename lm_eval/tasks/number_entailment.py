"""
"""
from lm_eval.api.task import PromptSourceTask


_CITATION = """

"""


class NumEntailmentBase(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "kkawamu1/number_entailment"
    DATASET_NAME = None
    SPLIT = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def max_generation_length(self):
        return 64


class NumEntailment(NumEntailmentBase):
    """this is for train/validation/test"""

    SPLIT = ""


class NumEntailmentNegative(NumEntailmentBase):
    """this is for negative"""

    SPLIT = "negative"

    def test_docs(self):
        return self.dataset["negative"]

class NumEntailmentAffirmative(NumEntailmentBase):
    """this is for affirmative"""

    SPLIT = "affirmative"

    def test_docs(self):
        return self.dataset["affirmative"]