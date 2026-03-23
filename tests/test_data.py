# pyright: reportMissingImports=false
from finbert_llm_comparison.data import build_finbert_reference_test_indices


def test_finbert_reference_split_size() -> None:
    indices = build_finbert_reference_test_indices(total_size=100, random_state=0)
    assert len(indices) == 20
    assert len(set(indices)) == 20
