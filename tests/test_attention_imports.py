from __future__ import annotations

import unittest

from abstractgraph_graphicalizer.attention import (
    AbstractGraphPreprocessor,
    ImageNodeClusterer,
    build_preimage_edges_from_attention,
)


class AttentionImportTest(unittest.TestCase):
    def test_attention_symbols_exist(self) -> None:
        self.assertTrue(AbstractGraphPreprocessor)
        self.assertTrue(ImageNodeClusterer)
        self.assertTrue(build_preimage_edges_from_attention)


if __name__ == "__main__":
    unittest.main()
