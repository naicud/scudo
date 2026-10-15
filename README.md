# Scudo

Scudo è una coppia di modelli generalisti IT/EN che giudicano un testo rispetto
a un **criterio semantico** scritto in linguaggio naturale e restituiscono un
verdetto tri-state: `VIOLATION`, `NO_VIOLATION` o `INSUFFICIENT`. Pesi e codice
sono pubblicati con licenza Apache-2.0.

Questo repository possiede la costruzione dei dataset, le ricette di training
su Colab, le eval dei modelli e le model card. Stato: scheletro, nessun modello
ancora addestrato né pubblicato.

## Confini

Scudo è un **sensore semantico**, non un'autorità. Non prende decisioni di
policy, non concede permessi e non esegue tool: la policy engine e il gateway
decidono e applicano l'azione (ALLOW, BLOCK, REVIEW). Il gateway vive in un
progetto privato separato e consuma i GGUF pubblicati da Scudo tramite un
manifest con digest. Identità, permessi, tenant, DLP e autorizzazione dei tool
restano controlli deterministici fuori dal modello. Un timeout, un errore o un
output malformato del sensore non diventano mai `ALLOW` a valle.

Le categorie di rischio classiche sono pubblicate come **criteri di default**,
non cablate nel modello. La specializzazione per dominio o cliente passa da
**adapter di specializzazione** sopra il generalista, senza cambiare il
contratto tri-state.

## Modelli

| Tier | Base | Ruolo | Repository Hugging Face |
| --- | --- | --- | --- |
| Fast | `Qwen/Qwen3.5-4B` | verdetto tipizzato letto sugli score delle opzioni, senza generazione | `naicud/scudo-fast-4b`, `naicud/scudo-fast-4b-GGUF` |
| Accurate | `Qwen/Qwen3.5-9B` | ragionamento, poi score delle opzioni sul verdetto tri-state | `naicud/scudo-accurate-9b`, `naicud/scudo-accurate-9b-GGUF` |

Il training usa Unsloth CUDA su Colab A100, con checkpoint su un repository
Hugging Face privato e ripresa dopo l'interruzione della sessione. Ogni modello
si esporta in GGUF prima della pubblicazione. Le decisioni sono negli
[ADR](docs/adr/).

## Perimetro della v1

- prompt utente;
- risposta del modello;
- prompt injection diretta e indiretta;
- over-refusal.

Tool/agent e PII restano alla v2. La superficie è testuale IT/EN/misto.

## Regola sulle licenze dei dati

Entrano nel training solo dati con licenza che consente uso commerciale e pesi
derivati: Apache-2.0, MIT, CC0, CC-BY, ODC-BY. Restano fuori dataset NC,
CC-BY-SA, dataset gated con termini propri e dati generati o distillati da
modelli la cui licenza limita uso o training. Ogni dataset ha una
[data card](docs/templates/data-card.md) con fonte, licenza, generatore,
filtri, deduplicazione contro i benchmark e attribuzioni CC-BY.

Dati, pesi, checkpoint e token non entrano mai in questo repository: li
bloccano `.gitignore` e il controllo `scripts/check_no_artifacts.py`.

## Sviluppo

```sh
uv sync                      # pacchetto base, senza dipendenze runtime
uv run pytest
uv sync --extra train        # stack di training (Unsloth, torch, transformers, ...)
uvx pre-commit install       # attiva il controllo su dati, pesi e token
```

L'import di `scudo` non carica torch, Unsloth o transformers.

## Checkpoint di training

I checkpoint vanno in un repository Hugging Face privato. Va creato una volta,
con un token di scrittura dell'account `naicud`:

```sh
uvx --from huggingface_hub hf auth login
uvx --from huggingface_hub hf repos create naicud/scudo-checkpoints --type model --private --exist-ok
```

## Licenza

[Apache-2.0](LICENSE).
