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
        
class NumEntailmentNegative2To999(NumEntailmentBase):
    """this is for negative_2_999"""

    SPLIT = "negative_2_999"

    def test_docs(self):
        return self.dataset["negative_2_999"]

class NumEntailmentAffirmative2To999(NumEntailmentBase):
    """this is for affirmative_2_999"""

    SPLIT = "affirmative_2_999"

    def test_docs(self):
        return self.dataset["affirmative_2_999"]

## Spelled out ###

class NumEntailmentNegativeSpelledOut(NumEntailmentBase):
    """this is for negative_spelled_out"""

    SPLIT = "negative_spelled_out"

    def test_docs(self):
        return self.dataset["negative_spelled_out"]


class NumEntailmentAffirmativeSpelledOut(NumEntailmentBase):
    """this is for affirmative_spelled_out"""

    SPLIT = "affirmative_spelled_out"

    def test_docs(self):
        return self.dataset["affirmative_spelled_out"]
        
class NumEntailmentNegative2To999SpelledOut(NumEntailmentBase):
    """this is for negative_2_999_spelled_out"""

    SPLIT = "negative_2_999_spelled_out"

    def test_docs(self):
        return self.dataset["negative_2_999_spelled_out"]

class NumEntailmentAffirmative2To999SpelledOut(NumEntailmentBase):
    """this is for affirmative_2_999_spelled_out"""

    SPLIT = "affirmative_2_999_spelled_out"

    def test_docs(self):
        return self.dataset["affirmative_2_999_spelled_out"]

####################

class NumEntailmentNegativeExplicit(NumEntailmentBase):
    """this is for negative_explicit"""

    SPLIT = "negative_explicit"

    def test_docs(self):
        return self.dataset["negative_explicit"]

class NumEntailmentAffirmativeExplicit(NumEntailmentBase):
    """this is for affirmative_explicit"""

    SPLIT = "affirmative_explicit"

    def test_docs(self):
        return self.dataset["affirmative_explicit"]

class NumEntailmentNegativeSpelledOutExplicit(NumEntailmentBase):
    """this is for negative_spelled_out_explicit"""

    SPLIT = "negative_spelled_out_explicit"

    def test_docs(self):
        return self.dataset["negative_spelled_out_explicit"]


class NumEntailmentAffirmativeSpelledOutExplicit(NumEntailmentBase):
    """this is for affirmative_spelled_out_explicit"""

    SPLIT = "affirmative_spelled_out_explicit"

    def test_docs(self):
        return self.dataset["affirmative_spelled_out_explicit"]


###########################################################



class NumEntailmentLessThanExplicit(NumEntailmentBase):
    """this is for less_than_explicit"""

    SPLIT = "less_than_explicit"

    def test_docs(self):
        return self.dataset["less_than_explicit"]

class NumEntailmentMoreThanExplicit(NumEntailmentBase):
    """this is for more_than_explicit"""

    SPLIT = "more_than_explicit"

    def test_docs(self):
        return self.dataset["more_than_explicit"]

class NumEntailmentLessThanSpelledOutExplicit(NumEntailmentBase):
    """this is for less_than_spelled_out_explicit"""

    SPLIT = "less_than_spelled_out_explicit"

    def test_docs(self):
        return self.dataset["less_than_spelled_out_explicit"]


class NumEntailmentMoreThanSpelledOutExplicit(NumEntailmentBase):
    """this is for more_than_spelled_out_explicit"""

    SPLIT = "more_than_spelled_out_explicit"

    def test_docs(self):
        return self.dataset["more_than_spelled_out_explicit"]

####################at_least##############################

class NumEntailmentNegativeAtLesast(NumEntailmentBase):
    """this is for negative_at_least"""

    SPLIT = "negative_at_least"

    def test_docs(self):
        return self.dataset["negative_at_least"]

class NumEntailmentAffirmativeAtLesast(NumEntailmentBase):
    """this is for affirmative_at_least"""

    SPLIT = "affirmative_at_least"

    def test_docs(self):
        return self.dataset["affirmative_at_least"]


class NumEntailmentNegativeSpelledOutAtLesast(NumEntailmentBase):
    """this is for negative_spelled_out_at_least"""

    SPLIT = "negative_spelled_out_at_least"

    def test_docs(self):
        return self.dataset["negative_spelled_out_at_least"]


class NumEntailmentAffirmativeSpelledOutAtLesast(NumEntailmentBase):
    """this is for affirmative_spelled_out_at_least"""

    SPLIT = "affirmative_spelled_out_at_least"

    def test_docs(self):
        return self.dataset["affirmative_spelled_out_at_least"]
        

###################################################
class NumEntailmentMoreThan(NumEntailmentBase):
    """this is for more than"""

    SPLIT = "more_than"

    def test_docs(self):
        return self.dataset["more_than"]

class NumEntailmentLessThan(NumEntailmentBase):
    """this is for less than"""

    SPLIT = "less_than"

    def test_docs(self):
        return self.dataset["less_than"]
        
class NumEntailmentAHas(NumEntailmentBase):
    """this is for A_has"""

    SPLIT = "A_has"

    def test_docs(self):
        return self.dataset["A_has"]

class NumEntailmentBHas(NumEntailmentBase):
    """this is for B_has"""

    SPLIT = "B_has"

    def test_docs(self):
        return self.dataset["B_has"]