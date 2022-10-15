"""
"""
from lm_eval.api.task import PromptSourceTask


_CITATION = """

"""


class MetalinguisticNegationBase(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "kkawamu1/test_2"
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


class MetalinguisticNegationControl_1_Negative(MetalinguisticNegationBase):
    """this is for control_1_negative"""

    SPLIT = "control_1_negative"

    def test_docs(self):
        return self.dataset["control_1_negative"]

class MetalinguisticNegationControl_1_Affirmative(MetalinguisticNegationBase):
    """this is for control_1_affirmative"""

    SPLIT = "control_1_affirmative"

    def test_docs(self):
        return self.dataset["control_1_affirmative"]

class MetalinguisticNegationControl_2_Negative(MetalinguisticNegationBase):
    """this is for control_2_adjective_modification_negative"""

    SPLIT = "control_2_adjective_modification_negative"

    def test_docs(self):
        return self.dataset["control_2_adjective_modification_negative"]

class MetalinguisticNegationControl_2_Affirmative(MetalinguisticNegationBase):
    """this is for control_2_adjective_modification_affirmative"""

    SPLIT = "control_2_adjective_modification_affirmative"

    def test_docs(self):
        return self.dataset["control_2_adjective_modification_affirmative"]

class MetalinguisticNegationExp(MetalinguisticNegationBase):
    """this is for experimental_condition"""

    SPLIT = "experimental_condition"

    def test_docs(self):
        return self.dataset["experimental_condition"]