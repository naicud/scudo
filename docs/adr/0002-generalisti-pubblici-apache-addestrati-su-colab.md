---
status: accepted
---

> Origine: ADR accettato nel repository privato `open-guard`, copiato qui senza
> modifiche al testo. Riferimenti a issue, tag, lane e percorsi sono di quel
> repository.

# Generalisti pubblici Apache-2.0, addestrati su Colab con dati a licenza commerciale

Fast e Accurate diventano modelli nostri, pubblicati come **modelli generalisti**
IT/EN su Hugging Face e GitHub con licenza Apache-2.0 per pesi e codice. I
generalisti giudicano criteri semantici liberi; le categorie di rischio classiche
sono pubblicate come **criteri di default**, non cablate nel modello. La
specializzazione per dominio o cliente avviene con **adapter di
specializzazione** sopra il generalista. I ruoli dei tier restano quelli
dell'ADR 0001.

Regole:

- i dati di training ammessi hanno licenza che consente uso commerciale e pesi
  derivati (Apache-2.0, MIT, CC0, CC-BY, ODC-BY). Restano fuori dataset NC,
  gated con termini propri e dati generati o distillati da modelli la cui licenza
  limita uso o training (es. famiglia `openjev/openjev`, CC-BY-NC-4.0). Ogni
  dataset entra nel manifest con licenza e fonte verificate;
- il training dei backbone reali usa Unsloth CUDA su Colab A100, con checkpoint
  su un repository Hugging Face privato e ripresa dopo l'interruzione della
  sessione. La lane MLX sul Mac resta per gli smoke meccanici. Ogni modello si
  esporta in GGUF e passa dal backend nativo prima della pubblicazione;
- la v1 copre prompt utente, risposta del modello, prompt injection diretta e
  indiretta, over-refusal. Tool/agent e PII restano alla v2: oggi sono coperti da
  controlli deterministici e dalla regola con Accurate obbligatorio;
- il progetto pubblico si chiama **Scudo**: repository GitHub `naicud/scudo` e
  modelli Hugging Face `naicud/scudo-fast-4b` e `naicud/scudo-accurate-9b` (più
  le varianti `-GGUF`), sull'account personale e non su un'organizzazione.
  Scudo possiede costruzione dei dataset, ricette Colab, eval dei modelli e
  model card; il gateway resta privato in `open-guard` e consuma i GGUF
  pubblicati tramite manifest con digest. I checkpoint pronti dello screen #25
  sono baseline di confronto, non punti di partenza scelti a priori;
- il Fast parte da `Qwen/Qwen3.5-4B` e l'Accurate da `Qwen/Qwen3.5-9B`. Un
  checkpoint di terzi entra come punto di partenza solo se dichiara i dati di
  training (APUS-Reflex oggi non lo fa); i teacher hanno licenza aperta e girano
  via API, l'A100 resta per il training.

## Considered Options

- **Tassonomia fissa** (stile Llama Guard): scartata, contraddice il criterio
  semantico versionato e non regge la specializzazione per cliente.
- **Ammettere CC-BY-SA o dati NC con pesi research-only**: scartato, un
  prodotto pubblico deve essere usabile da un'azienda senza parere legale.
- **Pubblicare l'intero repository**: scartato per ora, la storia contiene
  documenti interni e una pubblicazione non si ritira.
- **Partire da guard pronti** (Rizzo Flow, Granite policy-guardrails,
  APUS-Reflex): scartato come punto di partenza per provenienza dei dati non
  dichiarata o di altri produttori; restano baseline. La famiglia
  `openjev/openjev` è CC-BY-NC-4.0 ed è esclusa anche come teacher.
- **Training su Mac con MLX**: insufficiente per backbone 4–9B e per i dataset
  della v1.
