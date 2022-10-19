"""
"""
from lm_eval.api.task import PromptSourceTask


_CITATION = """

"""


class MetalinguisticNegationBase(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "kkawamu1/num_meta"
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


class MetalinguisticNegation(MetalinguisticNegationBase):
    """this is for train/validation/test"""

    SPLIT = ""


class MetalinguisticNegationControl_1(MetalinguisticNegationBase):
    """this is for control_1"""

    SPLIT = "control_1"

    def test_docs(self):
        return self.dataset["control_1"]

class MetalinguisticNegationControl_2(MetalinguisticNegationBase):
    """this is for control_2"""

    SPLIT = "control_2"

    def test_docs(self):
        return self.dataset["control_2"]

class MetalinguisticNegationControl_3(MetalinguisticNegationBase):
    """this is for control_3"""

    SPLIT = "control_3"

    def test_docs(self):
        return self.dataset["control_3"]


class MetalinguisticNegationControl_4(MetalinguisticNegationBase):
    """this is for control_4"""

    SPLIT = "control_4"

    def test_docs(self):
        return self.dataset["control_4"]
