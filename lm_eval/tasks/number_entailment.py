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
    def test_docs(self):
        return self.dataset[self.SPLIT]
types = ['fine_grained', 'coarse_grained']
int_ranges = ['1_100', '1_1000', '1_10000']
conditions = ['affirmative', 'affirmative_spelled_out', 'negative', 'negative_spelled_out', 'more_than', 'more_than_spelled_out', 'less_than', 'less_than_spelled_out']

_URLS = {}
for type in types:
    for int_range in int_ranges:
        for condition in conditions:
            _URLS[f'{condition}_{int_range}_{type}'] = f'{condition}_{int_range}_{type}.jsonl'
def construct_tasks():
    tasks = {}
    for task_name in _URLS:
        a = NumEntailment
        a.SPLIT = task_name
        tasks[f"number_entailment_{task_name}"] = a
    return tasks


        
class NumEntailmentNegative2To100(NumEntailmentBase):
    """this is for negative_2_100"""

    SPLIT = "negative_2_100"

    def test_docs(self):
        return self.dataset["negative_2_100"]

class NumEntailmentAffirmative2To100(NumEntailmentBase):
    """this is for affirmative_2_100"""

    SPLIT = "affirmative_2_100"

    def test_docs(self):
        return self.dataset["affirmative_2_100"]

class NumEntailmentNegative2To100SpelledOut(NumEntailmentBase):
    """this is for negative_spelled_out_2_100"""

    SPLIT = "negative_spelled_out_2_100"

    def test_docs(self):
        return self.dataset["negative_spelled_out_2_100"]

class NumEntailmentAffirmative2To100SpelledOut(NumEntailmentBase):
    """this is for affirmative_2_100_spelled_out"""

    SPLIT = "affirmative_2_100_spelled_out"

    def test_docs(self):
        return self.dataset["affirmative_spelled_out_2_100"]



class NumEntailmentLessThan2To100(NumEntailmentBase):
    """this is for less_than_2_100"""

    SPLIT = "less_than_2_100"

    def test_docs(self):
        return self.dataset["less_than_2_100"]

class NumEntailmentMoreThan2To100(NumEntailmentBase):
    """this is for more_than_2_100"""

    SPLIT = "more_than_2_100"

    def test_docs(self):
        return self.dataset["more_than_2_100"]

class NumEntailmentLessThan2To100SpelledOut(NumEntailmentBase):
    """this is for less_than_spelled_2_100"""

    SPLIT = "less_than_spelled_out_2_100"

    def test_docs(self):
        return self.dataset["less_than_spelled_out_2_100"]


class NumEntailmentMoreThan2To100SpelledOut(NumEntailmentBase):
    """this is for more_than_spelled_out_2_100"""

    SPLIT = "more_than_spelled_out_2_100"

    def test_docs(self):
        return self.dataset["more_than_spelled_out_2_100"]

#########################################################################

class NumEntailmentNegative100To999(NumEntailmentBase):
    """this is for negative_100_999"""

    SPLIT = "negative_100_999"

    def test_docs(self):
        return self.dataset["negative_100_999"]

class NumEntailmentAffirmative100To999(NumEntailmentBase):
    """this is for affirmative_100_999"""

    SPLIT = "affirmative_100_999"

    def test_docs(self):
        return self.dataset["affirmative_100_999"]

class NumEntailmentNegative100To999SpelledOut(NumEntailmentBase):
    """this is for negative_spelled_out_100_999"""

    SPLIT = "negative_spelled_out_100_999"

    def test_docs(self):
        return self.dataset["negative_spelled_out_100_999"]

class NumEntailmentAffirmative100To999SpelledOut(NumEntailmentBase):
    """this is for affirmative_100_999_spelled_out"""

    SPLIT = "affirmative_100_999_spelled_out"

    def test_docs(self):
        return self.dataset["affirmative_spelled_out_100_999"]



class NumEntailmentLessThan100To999(NumEntailmentBase):
    """this is for less_than_100_999"""

    SPLIT = "less_than_100_999"

    def test_docs(self):
        return self.dataset["less_than_100_999"]

class NumEntailmentMoreThan100To999(NumEntailmentBase):
    """this is for more_than_100_999"""

    SPLIT = "more_than_100_999"

    def test_docs(self):
        return self.dataset["more_than_100_999"]

class NumEntailmentLessThan100To999SpelledOut(NumEntailmentBase):
    """this is for less_than_spelled_100_999"""

    SPLIT = "less_than_spelled_out_100_999"

    def test_docs(self):
        return self.dataset["less_than_spelled_out_100_999"]


class NumEntailmentMoreThan100To999SpelledOut(NumEntailmentBase):
    """this is for more_than_spelled_out_100_999"""

    SPLIT = "more_than_spelled_out_100_999"

    def test_docs(self):
        return self.dataset["more_than_spelled_out_100_999"]



#########################################################################

class NumEntailmentNegative1000To10000(NumEntailmentBase):
    """this is for negative_1000_10000"""

    SPLIT = "negative_1000_10000"

    def test_docs(self):
        return self.dataset["negative_1000_10000"]

class NumEntailmentAffirmative1000To10000(NumEntailmentBase):
    """this is for affirmative_1000_10000"""

    SPLIT = "affirmative_1000_10000"

    def test_docs(self):
        return self.dataset["affirmative_1000_10000"]

class NumEntailmentNegative1000To10000SpelledOut(NumEntailmentBase):
    """this is for negative_spelled_out_1000_10000"""

    SPLIT = "negative_spelled_out_1000_10000"

    def test_docs(self):
        return self.dataset["negative_spelled_out_1000_10000"]

class NumEntailmentAffirmative1000To10000SpelledOut(NumEntailmentBase):
    """this is for affirmative_1000_10000_spelled_out"""

    SPLIT = "affirmative_1000_10000_spelled_out"

    def test_docs(self):
        return self.dataset["affirmative_spelled_out_1000_10000"]



class NumEntailmentLessThan1000To10000(NumEntailmentBase):
    """this is for less_than_1000_10000"""

    SPLIT = "less_than_1000_10000"

    def test_docs(self):
        return self.dataset["less_than_1000_10000"]

class NumEntailmentMoreThan1000To10000(NumEntailmentBase):
    """this is for more_than_1000_10000"""

    SPLIT = "more_than_1000_10000"

    def test_docs(self):
        return self.dataset["more_than_1000_10000"]

class NumEntailmentLessThan1000To10000SpelledOut(NumEntailmentBase):
    """this is for less_than_spelled_1000_10000"""

    SPLIT = "less_than_spelled_out_1000_10000"

    def test_docs(self):
        return self.dataset["less_than_spelled_out_1000_10000"]


class NumEntailmentMoreThan1000To10000SpelledOut(NumEntailmentBase):
    """this is for more_than_spelled_out_1000_10000"""

    SPLIT = "more_than_spelled_out_1000_10000"

    def test_docs(self):
        return self.dataset["more_than_spelled_out_1000_10000"]