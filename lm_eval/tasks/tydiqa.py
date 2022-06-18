"""
TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages

TyDi QA is a question answering dataset covering 11 typologically diverse languages with 200K question-answer pairs.

Paper: https://arxiv.org/abs/2003.05002
Homepage: https://ai.google.com/research/tydiqa
"""

from lm_eval.base import PromptSourceTask

_CITATION = """
@article{tydiqa,
    title   = {TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
    author  = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki}
    year    = {2020},
    journal = {Transactions of the Association for Computational Linguistics}
}
"""

class TyDiQAPrimary(PromptSourceTask):
    VERSION = 1
    DATASET_PATH = "tydiqa"
    DATASET_NAME = "primary_task"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def invalid_doc_for_prompt(self, doc) -> bool:
        # HACK: Some templates have conditionals that ignore documents
        # when the condition is not met, like `{if doc['question'] != \"cause\"}`.
        # This means the prompt will never produce an input and target.
        # TODO: Remove this when fixed in `promptsource`
        try:
            self.prompt.apply(doc)
            return False
        except:
            return True
        
class TyDiQASecondary(PromptSourceTask):
    VERSION = 1
    DATASET_PATH = "tydiqa"
    DATASET_NAME = "secondary_task"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def max_generation_length(self):
        return 128

    def compute_score(self, targets, pred, metric=compute_exact, agg=max):
        return agg([metric(t, pred) for t in targets])

    def process_results(self, doc, results):
        # Based on PIAF's implementation
        targets = self.doc_to_target(doc)
        pred = results[0].strip()
        agg_exact_match = self.compute_score(targets, pred, compute_exact)
        agg_f1 = self.compute_score(targets, pred, compute_f1)
        
        out = {"f1": agg_f1, 
                "exact_match": agg_exact_match}

        if self.save_examples:
            example = {"target": targets, "pred": pred}
            return out, example
        return out

    def invalid_doc_for_prompt(self, doc) -> bool:
        # HACK: Some templates have conditionals that ignore documents
        # when the condition is not met, like `{if doc['question'] != \"cause\"}`.
        # This means the prompt will never produce an input and target.
        # TODO: Remove this when fixed in `promptsource`
        try:
            self.prompt.apply(doc)
            return False
        except:
            return True
    
    def higher_is_better(self):
        return {
            "f1": True,
            "exact_match": True,
        }

    def aggregation(self):
        return {
            "f1": mean,
            "exact_match": mean,
        }
