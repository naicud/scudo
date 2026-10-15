"""Riepilogo markdown del report del pilot, da incollare nell'issue #5."""

import json


def _pct(value) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _usd(value, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _num(value) -> str:
    return "—" if value is None else f"{value:.0f}"


def markdown(report: dict) -> str:
    """Riepilogo da incollare nell'issue #5; il dettaglio resta in report.json."""
    meta, teachers = report["sample"]["meta"], report["teachers"]
    lines = [
        "## Pilot teacher (#5)", "",
        f"Campione `{report['sample']['sha256'][:16]}`: {report['sample']['gold']} casi gold "
        f"(label umane), {report['sample']['relabel']} righe `llm_jury`; seed {meta['seed']}; "
        f"prompt `{report['prompt_version']}`.",
        "Fonti: " + ", ".join(f"`{k}` ({'/'.join(v)})" for k, v in meta["sources"].items()) + ". "
        "I casi dei benchmark sono solo valutazione: le uscite del teacher su di essi non "
        "entrano nel training.",
        f"Lingue senza casi gold: {', '.join(meta['languages_missing']) or 'nessuna'}.", "",
        "Voci del manifest bloccate dal cancello nel loro ruolo (non usate):",
        *[f"- `{name}`: {'; '.join(found)}" for name, found in meta["gate"].items()],
        "", "### Teacher", "",
        "| teacher | revisione | licenza | provider | revisione servita | termini verificati |",
        "|---|---|---|---|---|---|",
    ]
    for model, t in teachers.items():
        p = t["provider"]
        lines.append(f"| `{model}` | `{t['registry']['revision'][:12]}` | {t['registry']['license']} "
                     f"| {p['provider']} (`{p['provider_model']}`) | {p['served_revision']} "
                     f"| {p['terms_checked_by']}, {p['terms_checked_on']}: {p['terms_url']} |")
    lines += ["", "### Gold", "",
              "| teacher | accordo | accordo su ok | rifiuti | malformati | troncati | "
              "timeout/errori | token in | token out | USD per item |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for model, t in teachers.items():
        g = t["gold"]
        lines.append(f"| `{model}` | {_pct(g['agreement'])} | {_pct(g['agreement_on_ok'])} "
                     f"| {_pct(g['refusal_rate'])} | {_pct(g['malformed_rate'])} "
                     f"| {_pct(g['truncated_rate'])} | {_pct(g['error_rate'])} "
                     f"| {_num(g['mean_prompt_tokens'])} | {_num(g['mean_completion_tokens'])} "
                     f"| {_usd(g['cost_per_item_usd'], 6)} |")
    lines += ["", "Accordo per strato:", "", "| teacher | " + " | ".join(
        f"{dim}={value}" for dim in ("language", "surface", "verdict")
        for value in next(iter(teachers.values()))["gold"][f"by_{dim}"]) + " |"]
    lines.append("|---|" + "---|" * (lines[-1].count("|") - 2))
    for model, t in teachers.items():
        lines.append(f"| `{model}` | " + " | ".join(
            _pct(cell["agreement"]) for dim in ("language", "surface", "verdict")
            for cell in t["gold"][f"by_{dim}"].values()) + " |")
    lines += ["", "Proxy dei ragionamenti (meccanico; la qualità la giudica la revisione umana "
              "di `reasoning-review-*.csv`):", "",
              "| teacher | ok | coerenti col verdetto | al più 3 frasi | lessico dell'evidenza | "
              "caratteri medi |", "|---|---|---|---|---|---|"]
    for model, t in teachers.items():
        r = t["gold"]["reasoning_proxy"]
        lines.append(f"| `{model}` | {r['n_ok']} | {_pct(r['consistent_rate'])} "
                     f"| {_pct(r['within_limit_rate'])} | {_pct(r['evidence_overlap_rate'])} "
                     f"| {_num(r['mean_chars'])} |")
    v = report["relabel_validation"]
    spot = v["spot_check"]
    lines += ["", "### Rietichettatura `llm_jury`", "",
              f"{v['n']} righe; verità: {v['truth']}.",
              "Accordo con la giuria originale: " + ", ".join(
                  f"`{t}` {_pct(a)}" for t, a in v["agreement_with_jury"].items()) + ".",
              f"Giuria di teacher (tutti ok e concordi): copertura {_pct(v['teacher_jury_coverage'])}, "
              f"accordo con la giuria originale {_pct(v['teacher_jury_agrees_with_jury'])}.",
              "Ancora, accordo sulle risposte gold: " + ", ".join(
                  f"`{t}` {_pct(a)}" for t, a in v["anchor_gold_response_agreement"].items()) + ".",
              ("Spot-check umano: " + spot["status"]) if "status" in spot else (
                  f"Spot-check umano su {spot['n']} righe ({spot['invalid']} non valide): giuria "
                  f"originale {_pct(spot['jury_accuracy'])}, giuria di teacher "
                  f"{_pct(spot['teacher_jury_accuracy'])} su {spot['teacher_jury_covered']} coperte."),
              "", "### Volume v0.1 (stima)", "",
              f"Parametri: {json.dumps(report['volume']['parameters'])}. "
              f"Metodo: {report['volume']['method']}.", "",
              "| teacher | #8 difficoltà | #9 ragionamenti | #7 traduzione | totale USD | "
              "nel residuo |", "|---|---|---|---|---|---|"]
    for model, est in report["volume"]["teachers"].items():
        lines.append(f"| `{model}` | {_usd(est['fast_difficulty_usd'])} | {_usd(est['reasoning_usd'])} "
                     f"| {_usd(est['translation_usd'])} | {_usd(est['total_usd'])} "
                     f"| {'sì' if est['fits_remaining'] else 'no'} |")
    s = report["spend"]
    lines += ["", "### Spesa", "",
              f"Pilot: {_usd(s['pilot_usd'], 4)} USD (autorizzati: " + ", ".join(
                  f"{a['amount_usd']:.2f}" for a in s["authorisations"]) + f"). Registro: "
              f"{_usd(s['ledger_total_usd'], 4)} USD su {s['cap_usd']:.0f}; residuo "
              f"{_usd(s['remaining_usd'])} USD.",
              "", "Stop dopo il pilot: nessuna generazione in volume senza la conferma umana "
              "(`python -m scudo.pilot confirm-volume`).", ""]
    return "\n".join(lines)
