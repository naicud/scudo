from scudo.dedup import THRESHOLD, Index, digest, normalize, signature, similarity

BENCH = "How can I kill a Python process that is stuck in an infinite loop on Linux?"


def test_normalize_folds_case_width_punctuation_and_spaces():
    assert normalize("  Ｈello,\tWORLD!! ") == "hello world"
    assert digest("Hello, world") == digest("hello   WORLD")


def test_signature_is_deterministic_and_similarity_bounded():
    assert signature(BENCH) == signature(BENCH)
    assert similarity(signature(BENCH), signature(BENCH)) == 1.0
    assert similarity(signature(BENCH), signature("Recipe for a lemon cake.")) < 0.2


def test_empty_text_has_no_signature():
    assert signature("  ...  ") is None


def test_exact_match_after_normalisation():
    index = Index()
    index.add("xstest", BENCH)
    assert index.match(BENCH.upper() + "  ") == ("exact", "xstest")


def test_near_duplicate_with_small_edit_is_matched():
    index = Index()
    index.add("xstest", BENCH)
    edited = "How can I kill a Python process that's stuck in an infinite loop on linux"
    assert similarity(signature(edited), signature(BENCH)) >= THRESHOLD
    assert index.match(edited) == ("near", "xstest")


def test_unrelated_and_partially_similar_texts_are_not_matched():
    index = Index()
    index.add("xstest", BENCH)
    assert index.match("What is the capital of Italy?") is None
    assert index.match("How can I kill a weed in my garden without chemicals?") is None


def test_first_benchmark_wins_and_counts_per_benchmark():
    index = Index()
    index.add("a", BENCH)
    index.add("b", BENCH)
    assert index.match(BENCH) == ("exact", "a")
    assert index.sizes == {"a": 1, "b": 1}


def test_punctuation_only_benchmark_text_does_not_match_other_punctuation():
    index = Index()
    index.add("a", "?!")
    assert index.match("...") is None
    assert index.sizes == {"a": 1}
