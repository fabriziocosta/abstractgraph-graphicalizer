from __future__ import annotations

import unittest

from abstractgraph_graphicalizer.attention import AbstractGraphPreprocessor, ImageNodeClusterer


class AttentionImportTest(unittest.TestCase):
    def test_attention_symbols_exist(self) -> None:
        self.assertTrue(AbstractGraphPreprocessor)
        self.assertTrue(ImageNodeClusterer)


if __name__ == "__main__":
    unittest.main()
