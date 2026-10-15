from dataclasses import replace

from scudo.datacard import attributions, render
from scudo.manifest import load
from test_manifest import BENCH, DECISION, SHA, TRAIN


def test_card_follows_the_template_and_carries_manifest_fields():
    card = render(TRAIN)
    for heading in ("## Fonte", "## Licenza", "## Generatore e teacher",
                    "## Trasformazioni e filtri", "## Deduplicazione contro i benchmark",
                    "## Attribuzioni", "## Rischi noti"):
        assert heading in card
    assert SHA in card and "cc-by-4.0" in card and "SÌ" in card
    assert "pendente" in card  # nessuna statistica di build


def test_card_reports_build_counts_and_human_decision():
    stats = {"rows": 10, "dropped": {"prompt REDACTED": 2}, "cases": 12, "kept": 9,
             "needs_relabel": 3, "removed_exact": {"xstest": 2}, "removed_near": {"xstest": 1}}
    card = render(replace(TRAIN, generator_terms="INCERTO", decision=DECISION), stats)
    assert "prompt REDACTED: 2" in card and "xstest: 2" in card and "Mario Rossi" in card


def test_attributions_cover_cc_by_and_odc_by_training_sources_only():
    text = attributions([TRAIN, replace(TRAIN, name="odc", license="odc-by", title="Odc"),
                         replace(TRAIN, name="mit", license="mit", title="Mitset"),
                         replace(BENCH, title="Bench")])
    assert "Demo" in text and "Odc" in text
    assert "Mitset" not in text and "Bench" not in text
    assert "https://creativecommons.org/licenses/by/4.0/" in text
    assert SHA in text


def test_every_shipped_entry_renders():
    for entry in load():
        assert entry.id in render(entry)


def test_risk_text_containing_a_dash_list_marker_is_kept_verbatim():
    from dataclasses import replace

    from scudo.datacard import render
    from scudo.manifest import load

    entry = replace(load()[0], risks=("vedi  - nota interna",))
    assert "- vedi  - nota interna" in render(entry, None)
