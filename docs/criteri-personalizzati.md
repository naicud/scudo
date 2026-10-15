# Criteri personalizzati, casi INSUFFICIENT e criteri di default

Issue #6. Termini come in [`CONTEXT.md`](../CONTEXT.md); formato dei casi in
[`protocol.md`](protocol.md). Fonte: `src/scudo/open_jev.py`,
`src/scudo/canttalk.py`, `src/scudo/procedural.py`, `src/scudo/insufficient.py`,
`src/scudo/default_criteria.py`.

Regola comune: un tipo di domanda che non si esprime nel tri-state senza
perdita resta fuori, contato per motivo (`convert` restituisce casi e
`Counter` degli scarti). Un valore sconosciuto in un campo tri-state è un
errore, mai un NO_VIOLATION.

## Fonti

Revisioni lette il 15 ottobre 2026 da `api/datasets/<id>/revision/main`;
schema letto da datasets-server e dai file grezzi alla stessa revisione.

| Dataset | Config | Revisione | Licenza | Generatore e label |
| --- | --- | --- | --- | --- |
| `ZefanCai/Open-Jev` | `release-v2-redistributable` | `c67699e13d0ae25e35b77165a4b6b079bedc8aba` | CC0-1.0 (record generati), codice MIT | sintetico/procedurale del progetto Open-Jev; label da regole, oracoli programmatici o script expert; il «teacher» è uno script, nessun LLM nei generatori (codice letto alla revisione, 2026-10-15) |
| `nvidia/CantTalkAboutThis-Topic-Control-Dataset` | `mixtral` | `b015ae8902d8429ce32ebfed9f9935f891d10ed5` | CC-BY-4.0 | dialoghi e distrattori di train da Mixtral-8x7B-Instruct (Apache-2.0); distrattori del test scritti da due esperti (paper) |
| `tasksource/procedural-typed-decisions` | `all` | `e4b21b7b55f6cc9e76e33563e9d604ac5fcd25ad` | Apache-2.0 | generazione procedurale deterministica, nessun LLM; risposte calcolate esattamente dallo stato |

Superficie: `prompt` per tutti. Nel contratto v1 i criteri personalizzati non
hanno una superficie propria: per CantTalkAboutThis il turno giudicato è un
messaggio dell'utente; per Open-Jev e procedural l'evidenza è lo stato in
ingresso al sistema, non un output del modello. Lingua: `en` (nessun carattere
CJK o turco nei casi tenuti del train di Open-Jev).

## Mappature

### Open-Jev

Criterio = `question`; evidenza = `state_json` decodificato (stringa verbatim,
oggetto in `json.dumps(..., ensure_ascii=False)`). `metadata_json` e
`record_json` non si leggono: possono contenere label teacher privilegiate.

| Fonte | Verdetto o scarto |
| --- | --- |
| `noul`, opzioni `["no", "yes"]`, target `[0, 1]` | VIOLATION |
| `noul`, opzioni `["no", "yes"]`, target `[1, 0]` | NO_VIOLATION |
| `noul` con target non netto | `soft_target` |
| `noul` con altre opzioni | `unexpected_options` |
| `choice` | `choice_not_tristate` |
| `score` | `score_not_tristate` |
| `source = customer-control-v1` | `unverified_upstream_licence`: le descrizioni delle domande vengono dalla documentazione TypeSafe, licenza dichiarata non verificata |
| altro `kind` | errore |

Smoke locale sul train (79.116 righe): 47.505 casi (12.946 VIOLATION,
34.559 NO_VIOLATION); scarti 18.286 `choice_not_tristate`, 11.923
`score_not_tristate`, 1.402 `unverified_upstream_licence`. I `noul` tenuti
includono fonti non testuali in senso stretto (geometria di pixel, Snake,
ViZDoom): lo stato è comunque testo e la label è netta; il peso nel mix lo
decide chi costruisce il mix. Test e OOD hanno label pubbliche: l'uso va
dichiarato.

### CantTalkAboutThis

Criterio = `criterion(system_instruction)`: la policy del dialogo più la domanda
del paper («l'ultimo turno rispetta lo scenario?») girata in modo che VIOLATION
sia il turno fuori tema. Evidenza = dialogo fino al turno giudicato, in JSON
(`role`/`content`), con il turno utente giudicato per ultimo.

| Fonte | Verdetto o scarto |
| --- | --- |
| ogni turno utente di `conversation` | NO_VIOLATION (il paper li valuta come turni on-topic) |
| ogni distrattore, inserito dopo il turno assistente che contiene il suo `bot turn` | VIOLATION |
| `bot turn` assente da tutti i turni assistente | `bot_turn_not_found` |
| `bot turn` in più turni assistente | `bot_turn_ambiguous` |
| `conversation` vuota | `empty_conversation` (riga intera) |

La storia viene sempre da `conversation`, mai da altri distrattori;
`conversation_with_distractors` non si usa (vuoto nel test). Lo `source_id` è
un hash di policy e dialogo più `t<i>` o `d<k>`, perché la fonte non ha id.
Smoke locale: train 1.073 righe, 13.334 casi (4.845 VIOLATION), scarti 373
`bot_turn_not_found`, 2 `bot_turn_ambiguous`, 29 `empty_conversation`; test 20
righe, 307 casi. Limite: alcuni turni originali toccano sotto-temi che la policy
vieta; la label segue il dataset (fuori tema = distrattore). Il test è
annotato da umani e va tenuto fuori dal training.

### procedural-typed-decisions

Evidenza = `state` verbatim; `source_id` = `<id>:<domanda>`. Serve la colonna
`task`, presente solo nella config `all`.

| Fonte | Verdetto o scarto |
| --- | --- |
| `noul` sì/no, valore 1 / 0 | VIOLATION / NO_VIOLATION, criterio = `instructions` |
| `noul` graduato (`arithmetic.random_*`, `partial_observation_calibration.incident_real`, `policy_under_uncertainty.access_allowed`: colonna float, non ClassLabel) o valore diverso da 0 e 1 | `graded_noul`, anche se vale 0 o 1 |
| `record_eligibility.probe_eligibility`: `ineligible` / `eligible` / `undetermined` | VIOLATION / NO_VIOLATION / INSUFFICIENT con `eligibility_criterion(R)` |
| `record_eligibility.requirement_status`: `violated` / `met` / `unknown` | VIOLATION / NO_VIOLATION / INSUFFICIENT con `requirement_criterion(R, C)` |
| le due precedenti con formulazione diversa | `unparsed_question` |
| le due precedenti con probabilità non netta | `soft_label` |
| ogni altra `choice` | `choice_not_tristate` |
| `score` | `score_not_tristate` |
| altro tipo, o valore tri-state sconosciuto | errore |

`undetermined` e `unknown` diventano INSUFFICIENT perché lo stato stesso lo
definisce: «Explicitly absent information and conflicting claims count as
unknown, not false». Smoke locale su 366 righe (66 di `all`, 300 di
`record_eligibility`): 1.030 casi, 143 INSUFFICIENT dalla fonte.

## Casi INSUFFICIENT generati

`derive(case, method)` costruisce un caso INSUFFICIENT dal genitore etichettato
o solleva `Excluded` con il motivo; `derive_all` applica i metodi a una lista,
conta le esclusioni per `(metodo, motivo)` e scarta i doppioni di criterio ed
evidenza. Il figlio tiene fonte, lingua, superficie e criterio del genitore
(tranne `criterion_absent_info`); `source_id` è
`<genitore>#insufficient:<metodo>`.

| Metodo | Fonte | Costruzione |
| --- | --- | --- |
| `evidence_removed` | CantTalkAboutThis | si toglie il turno utente giudicato; il dialogo non finisce più con un turno da giudicare |
| `criterion_absent_info` | procedural, criteri di `record_eligibility` | stesso criterio su un record `R<n>` che non compare nello stato |
| `context_truncated` | procedural, criteri di `record_eligibility` | lo stato è troncato subito prima della voce del record sondato |
| `facts_not_provided` | Open-Jev, `reasoning-control-v1` aritmetica | si tolgono i `values`: la risposta proposta dipende da valori non forniti |

Regola di esclusione, strutturale e non euristica: mai INSUFFICIENT su un testo
che basta di per sé a violare. Ogni metodo accetta solo la sua fonte
(`unsupported_source` per tutte le altre, compresi i prompt dannosi di
Nemotron), solo genitori VIOLATION o NO_VIOLATION (`parent_insufficient`), e
toglie solo l'evidenza decisiva nota per struttura, verificando che ciò che
resta non la contenga:

- `evidence_removed`: rifiuta se la storia rimasta finisce con un altro turno
  utente (`retained_ends_with_user`); la storia viene solo da `conversation`,
  quindi non contiene distrattori;
- `context_truncated`: rifiuta se il record sondato è ancora nominato nel
  prefisso (`decisive_evidence_retained`) o non si trova una sola volta;
- `criterion_absent_info`: sceglie un id che non compare in nessun punto dello
  stato;
- `facts_not_provided`: solo stati con esattamente `expression`, `record_id`,
  `semantics`, `values` con `a`, `b`, `c`, e solo le tre espressioni verificate
  sul train (`a + (b * c)`, `(a - b) * c`, `(a * b) + c`), dove ogni variabile
  compare una volta e il risultato dipende da tutte (`unverified_expression`
  altrimenti).

I test negativi in `tests/test_insufficient.py` coprono ciascun rifiuto.
Smoke locale sui casi sopra: 9.209 figli (8.014 `evidence_removed`, 459
`criterion_absent_info`, 459 `context_truncated`, 277 `facts_not_provided`);
il pool è sbilanciato verso CantTalkAboutThis.

## Quota

`mix_insufficient(cases, generated, share=DEFAULT_SHARE, seed=...)` aggiunge
INSUFFICIENT generati finché gli INSUFFICIENT sono `share` del mix, contando
anche quelli già etichettati dalla fonte. `DEFAULT_SHARE = 0.12`, dentro il
10–15% dell'issue. `share` deve stare in (0, 1). Selezione e ordine dipendono
solo da seme, fonte e `source_id`, non dall'ordine dell'input. Fallisce se il
pool non basta o se i casi di partenza superano già la quota: non riempie a
metà e non sfora.

## Criteri di default

`scudo.default-criteria/1` (`DEFAULT_CRITERIA`, `by_id`): un criterio per ogni
categoria di rischio di Aegis 2.0, id `aegis.<nome>`, più i due criteri di
prompt injection della v1; ogni criterio è a `version = 0`, testo inglese nello
stile dei criteri del tracer («Does the text …? … Missing essential context
means INSUFFICIENT.»). Valgono per qualsiasi superficie. Il set 0 aveva solo i
criteri Aegis.

| Id | VIOLATION | Non è violazione |
| --- | --- | --- |
| `injection.direct` | il testo prova a far ignorare, sovrascrivere o aggirare istruzioni, regole o linee guida di sicurezza dell'assistente (istruzioni da dimenticare, modalità o persona «senza restrizioni», role-play o ipotesi per sbloccare ciò che rifiuterebbe) o a fargli rivelare istruzioni nascoste o segreti da proteggere | richieste ordinarie, anche dannose se poste in chiaro (le coprono i criteri Aegis); persona, tono o formato che non tolgono le regole |
| `injection.indirect` | contenuto di terzi letto dall'assistente durante un compito per l'utente (email, pagina web, documento, risultato di un tool) costruito per farlo deviare: ignorare le istruzioni, compiere azioni non chieste (inviare messaggi, chiamare tool), divulgare dati dell'utente | informazione e richieste che spetta al lettore umano decidere |

Testi scritti da noi; li usano i convertitori di in-the-wild, jackhhao
(`injection.direct`) e LLMail-Inject (`injection.indirect`).

Tassonomia letta dal README di `nvidia/Aegis-AI-Content-Safety-Dataset-2.0`
(`d86bb8bedff51d25ac834ab7838f1cc61acb7a2c`, CC-BY-4.0) e dalla Table 2 del
paper (arXiv 2501.09004v1): 12 categorie core e 9 fine-grained.

| Categoria Aegis 2.0 | Id |
| --- | --- |
| Violence | `aegis.violence` |
| Sexual | `aegis.sexual` |
| Criminal Planning/Confessions | `aegis.criminal_planning` |
| Guns/Illegal Weapons | `aegis.weapons` |
| Controlled/Regulated substances | `aegis.controlled_substances` |
| Suicide and Self Harm | `aegis.self_harm` |
| Sexual Minor | `aegis.sexual_minor` |
| Hate/Identity Hate | `aegis.hate` |
| PII/Privacy | `aegis.pii_privacy` |
| Harassment | `aegis.harassment` |
| Threat | `aegis.threat` |
| Profanity | `aegis.profanity` |
| Illegal Activity | `aegis.illegal_activity` |
| Immoral/Unethical | `aegis.immoral_unethical` |
| Unauthorized Advice | `aegis.unauthorized_advice` |
| Political/Misinformation/Conspiracy | `aegis.political_misinformation` |
| Fraud/Deception | `aegis.fraud_deception` |
| Copyright/Trademark/Plagiarism | `aegis.copyright` |
| High Risk Gov. Decision Making | `aegis.high_risk_gov_decision` |
| Malware | `aegis.malware` |
| Manipulation | `aegis.manipulation` |

`Safe`, `Needs Caution` (zona grigia, che il paper mappa a safe) e `Other`
(residuo senza definizione) non sono criteri. I testi sono nostri: per le core
seguono le regole «Should not» / «Can» del prompt catdesc del paper (Appendix
A.12.3), non copiate perché l'HTML arXiv è CC BY-SA 4.0. Per le fine-grained il
paper e il README pubblicano solo il nome: la loro definizione è NON
VERIFICATO e va rivista prima di misurare il generalista su quei criteri.
