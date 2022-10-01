"""
"""
from lm_eval.api.task import PromptSourceTask


_CITATION = """

"""


class MetalinguisticNegationBase(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "kkawamu1/test"
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


# class GEMXSUMChallgeSample(GEMXSUMBase):
#     """this is for challenge_train_sample/challenge_validation_sample"""

#     SPLIT = "challenge_sample"

#     def has_test_docs(self):
#         return False

#     def training_docs(self):
#         if self.has_training_docs():
#             return self.dataset["challenge_train_sample"]

#     def validation_docs(self):
#         if self.has_validation_docs():
#             return self.dataset["challenge_validation_sample"]


# class GEMXSUMChallgeTestBacktranslation(GEMXSUMBase):
#     """this is for challenge_test_backtranslation"""

#     SPLIT = "challenge_test_backtranslation"

#     def has_training_docs(self):
#         return False

#     def has_validation_docs(self):
#         return False

#     def test_docs(self):
#         if self.has_test_docs():
#             return self.dataset[self.SPLIT]


# class GEMXSUMChallgeTestBFP02(GEMXSUMBase):
#     """this is for challenge_test_bfp_02"""

#     SPLIT = "challenge_test_bfp_02"

#     def has_training_docs(self):
#         return False

#     def has_validation_docs(self):
#         return False

#     def test_docs(self):
#         if self.has_test_docs():
#             return self.dataset[self.SPLIT]


# class GEMXSUMChallgeTestBFP05(GEMXSUMBase):
#     """this is for challenge_test_bfp_05"""

#     SPLIT = "challenge_test_bfp_05"

#     def has_training_docs(self):
#         return False

#     def has_validation_docs(self):
#         return False

#     def test_docs(self):
#         if self.has_test_docs():
#             return self.dataset[self.SPLIT]


# class GEMXSUMChallgeTestNopunc(GEMXSUMBase):
#     """this is for challenge_test_nopunc"""

#     SPLIT = "challenge_test_nopunc"

#     def has_training_docs(self):
#         return False

#     def has_validation_docs(self):
#         return False

#     def test_docs(self):
#         if self.has_test_docs():
#             return self.dataset[self.SPLIT]


# class GEMXSUMChallgeTestCovid(GEMXSUMBase):
#     """this is for challenge_test_covid"""

#     SPLIT = "challenge_test_covid"

#     def has_training_docs(self):
#         return False

#     def has_validation_docs(self):
#         return False

#     def test_docs(self):
#         if self.has_test_docs():
#             return self.dataset[self.SPLIT]
