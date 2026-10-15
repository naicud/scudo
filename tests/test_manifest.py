from dataclasses import replace

import pytest

from scudo import aegis, nemotron
from scudo.ingest import CONVERTERS, UNCONVERTED
from scudo.manifest import (
    Decision,
    Entry,
    ManifestError,
    benchmarks,
    gate,
    load,
    problems,
    training,
)

SHA = "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c"
TRAIN = Entry(
    name="demo-train", id="org/demo", role="train", revision=SHA, read_on="2026-10-15",
    config="default", splits=("train",), languages=("en",), surfaces=("prompt",),
    license="cc-by-4.0", license_source="API e README", gated=False, texts_by="umani",
    labels_by="umani", generators=("nessun modello",), generator_terms="SÌ",
    generator_terms_note="testi e label umani", filters=("nessuno",), title="Demo",
    authors="Autori", url="https://huggingface.co/datasets/org/demo",
    changes="mappatura delle label", risks=("contenuti dannosi",),
)
BENCH = replace(TRAIN, name="demo-test", role="benchmark", splits=("test",),
                text_fields=("prompt",))
DECISION = Decision(by="Mario Rossi", on="2026-10-20", rationale="parere Legal n. 12")


def blocked(*entries):
    return problems(list(entries))


def test_admissible_train_and_benchmark_have_no_problems():
    assert blocked(TRAIN, BENCH) == []
    gate([TRAIN, BENCH])


@pytest.mark.parametrize("license", ["cc-by-nc-4.0", "cc-by-sa-4.0", "other", "NON VERIFICATO"])
def test_non_admitted_licence_blocks_training(license):
    [problem] = blocked(replace(TRAIN, license=license))
    assert "licenza" in problem and "demo-train" in problem


def test_nc_licence_is_accepted_only_for_an_eval_only_benchmark():
    assert blocked(replace(BENCH, license="cc-by-nc-4.0")) == []
    assert blocked(replace(BENCH, license="cc-by-sa-4.0"))


@pytest.mark.parametrize("revision", ["main", "NON VERIFICATO", SHA[:12]])
def test_unpinned_revision_blocks(revision):
    assert blocked(replace(TRAIN, revision=revision))
    assert blocked(replace(BENCH, revision=revision))


def test_uncertain_generator_terms_block_without_a_human_decision():
    uncertain = replace(TRAIN, generator_terms="INCERTO")
    assert "INCERTO" in blocked(uncertain)[0]
    assert blocked(replace(uncertain, decision=DECISION)) == []
    assert blocked(replace(uncertain, decision=replace(DECISION, rationale=" ")))


def test_restrictive_generator_terms_block_even_with_a_decision():
    assert blocked(replace(TRAIN, generator_terms="NO", decision=DECISION))


def test_generator_terms_do_not_gate_an_eval_only_benchmark():
    assert blocked(replace(BENCH, generator_terms="INCERTO")) == []


def test_gated_training_source_is_blocked():
    assert blocked(replace(TRAIN, gated=True))
    assert blocked(replace(BENCH, gated=True)) == []


def test_benchmark_without_verified_text_fields_is_blocked():
    assert "campi di testo" in blocked(replace(BENCH, text_fields=()))[0]


def test_incomplete_data_card_fields_block():
    assert blocked(replace(TRAIN, title=""))
    assert blocked(replace(TRAIN, changes="  "))
    assert blocked(replace(TRAIN, surfaces=("tool_call",)))


def test_held_out_split_used_for_training_is_blocked():
    leaked = replace(BENCH, splits=("train",))
    assert any("held-out" in problem for problem in blocked(TRAIN, leaked))


def test_duplicate_names_are_blocked():
    assert any("duplicato" in problem for problem in blocked(TRAIN, TRAIN))


def test_gate_raises_with_every_problem():
    with pytest.raises(ManifestError) as error:
        gate([replace(TRAIN, license="cc-by-nc-4.0"), replace(BENCH, text_fields=())])
    assert len(error.value.problems) == 2


def test_load_rejects_unknown_fields(tmp_path):
    path = tmp_path / "manifest.toml"
    path.write_text('version = "x"\n[[dataset]]\nname = "a"\nlicence = "mit"\n')
    with pytest.raises(ManifestError):
        load(path)


def test_shipped_manifest_loads_with_unique_names_and_pinned_or_blocked_revisions():
    entries = load()
    names = [entry.name for entry in entries]
    assert len(names) == len(set(names))
    for entry in entries:
        if len(entry.revision) != 40:
            assert any(entry.name in problem for problem in problems(entries))


def test_converter_revisions_match_the_manifest():
    by_name = {entry.name: entry for entry in load()}
    assert by_name["aegis2-train"].revision == aegis.REVISION
    assert by_name["nemotron3-train"].revision == nemotron.REVISION
    for entry in load():
        converter = CONVERTERS.get(entry.id)
        if converter is None:
            continue
        assert entry.revision == converter.REVISION, entry.name
        assert entry.config == getattr(converter, "CONFIG", entry.config), entry.name


def test_every_converted_training_source_is_in_the_manifest():
    trained = {entry.id for entry in training()}
    assert set(CONVERTERS) <= trained
    assert set(UNCONVERTED) <= trained


def test_every_registered_teacher_has_a_training_entry_with_its_revision_and_card():
    from scudo.teacher import CANDIDATES

    for model_id, candidate in CANDIDATES.items():
        (entry,) = [e for e in training() if e.id == model_id]
        assert entry.revision == candidate.revision and entry.license == candidate.license
        assert UNCONVERTED[model_id]
        # Data card completa: l'unico blocco ammesso è la decisione umana sui termini.
        assert [p for p in problems([entry]) if "INCERTO" not in p] == []


def test_open_jev_passes_the_gate_once_its_procedural_teacher_is_verified():
    assert not [p for p in problems(load()) if p.startswith("open-jev-train:")]


def test_registry_lists_the_held_out_benchmarks():
    names = {entry.name for entry in benchmarks()}
    assert {"aegis2-test", "nemotron3-test", "xstest", "toxicchat-test",
            "jbb-behaviors", "orbench-hard", "canttalk-test", "open-jev-heldout"} <= names
    excluded = {entry.name for entry in load() if entry.role == "excluded"}
    assert {"wildguardtest", "nemotron-sft-safety-v2"} <= excluded
    assert all(entry.role == "benchmark" for entry in benchmarks())
    assert all(entry.role == "train" for entry in training())


def test_training_entry_is_found_by_dataset_id_and_split():
    from scudo.manifest import training_entry

    entries = [BENCH, TRAIN]
    assert training_entry(entries, "org/demo", "train") == TRAIN
    assert training_entry(entries, "org/demo", "test") is None  # solo benchmark
    assert training_entry(entries, "org/other", "train") is None


def test_excluded_source_with_a_human_decision_does_not_block_and_is_never_used():
    excluded = replace(TRAIN, name="demo-excluded", role="excluded", generator_terms="NO",
                       gated=True, text_fields=(), decision=DECISION)
    assert blocked(excluded, TRAIN, BENCH) == []
    assert training([excluded, TRAIN, BENCH]) == [TRAIN]
    assert benchmarks([excluded, TRAIN, BENCH]) == [BENCH]


def test_excluded_source_needs_a_complete_human_decision():
    excluded = replace(TRAIN, role="excluded")
    assert "decisione" in blocked(excluded)[0]
    assert blocked(replace(excluded, decision=replace(DECISION, by=" ")))
