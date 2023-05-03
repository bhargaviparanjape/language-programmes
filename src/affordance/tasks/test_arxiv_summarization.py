import unittest

from arxiv_summarization import *


class TestSentClean(unittest.TestCase):
    cases = {
        "basic": ("This is an [example] article.", "This is an  article."),
        "nested": ("This is an [example [nested] brackets] article.", "This is an  article."),
    }

    def test_all(self):
        for case in self.cases.values():
            out = sent_clean(case[0])

            self.assertEqual(case[1], out)


if __name__ == "__main__":
    unittest.main(module="test_arxiv_summarization")
