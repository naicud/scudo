# Provider dei teacher su Hugging Face Inference Providers (issue #5)

Letto il 2026-10-15. Domanda: per i due teacher del pilot
([teacher-pilot.md](../teacher-pilot.md)), `openai/gpt-oss-120b` e
`Qwen/Qwen3.5-397B-A17B`, quali provider li servono tramite il router di
Hugging Face, quanto costano e se i loro termini consentono di usare gli output
per addestrare e pubblicare un modello Apache-2.0. Prima i termini, poi il
prezzo.

## Metodo

- Elenco e prezzi: `GET https://router.huggingface.co/v1/models` con il token
  locale (letto da `~/.cache/huggingface/token`, mai stampato), campo
  `providers[].pricing` in USD per milione di token. I prezzi cambiano: vanno
  riletti prima di ogni run.
- Termini: pagine ufficiali dei provider e di Hugging Face, lette come testo.
  Dove una pagina non è leggibile senza JavaScript o login, il punto resta
  **NON VERIFICATO**.
- Licenze dei modelli: entrambi `apache-2.0`, non gated (vedi
  [teacher-pilot.md](../teacher-pilot.md)). La licenza Apache-2.0 non limita
  l'uso degli output.
- Stato dell'account: il router risponde con il token locale, ma l'account non
  ha credito prepagato (gli Hugging Face Jobs hanno risposto 402). Prima di una
  run serve credito, oppure una chiave del provider configurata su Hugging Face.

## Hugging Face Inference Providers

- [Terms of Service](https://huggingface.co/terms-of-service) (la pagina riporta
  «Effective Date: September 15, 2022» e cita già gli Inference Providers tra i
  Services): «You own the Content you create!». Nessuna clausola limita l'uso
  degli output di inferenza per addestrare modelli.
- [Pricing](https://huggingface.co/docs/inference-providers/pricing): con le
  richieste instradate «billing is managed directly by Hugging Face» e «No
  separate provider account is required»; prezzi «with no markup from Hugging
  Face». In alternativa si usa una chiave propria del provider.
- [Security](https://huggingface.co/docs/inference-providers/security): «Logs
  are kept for debugging purposes for up to 30 days, but no user data or tokens
  are stored» e «please refer to the Data Security Policies of each provider».
- **NON VERIFICATO**: la documentazione non dice se i termini del provider
  vincolino chi chiama tramite il router con fatturazione Hugging Face. Per
  prudenza vanno rispettati anche quelli del provider.
- Selezione del provider: senza suffisso il router sceglie «the fastest
  available provider» e, con `provider="auto"`, passa ad altri provider se il
  primo non è disponibile. Il provider va quindi fissato nel nome del modello
  («appending the provider name to the model id (e.g.
  "openai/gpt-oss-120b:groq")»): altrimenti una richiesta può finire su un
  provider con termini non verificati.

## Provider e prezzi

USD per milione di token, dal router il 2026-10-15.

| Provider | gpt-oss-120b in / out | Qwen3.5-397B-A17B in / out |
|---|---|---|
| deepinfra | 0,037 / 0,17 | 0,45 / 3,00 |
| novita | 0,05 / 0,25 | 0,60 / 3,60 |
| ovhcloud | 0,09 / 0,47 | 0,71 / 4,25 |
| nscale | 0,10 / 0,40 | — |
| baseten | 0,10 / 0,50 | — |
| together | 0,15 / 0,60 | — |
| fireworks-ai | 0,15 / 0,60 | — |
| groq | 0,15 / 0,75 | — |
| scaleway | 0,171 / 0,684 | 0,684 / 4,104 |
| cerebras | 0,35 / 0,75 | — |
| featherless-ai | prezzo non esposto | prezzo non esposto |

## Termini dei provider sugli output

| Provider | Fonte | Cosa dicono sugli output | Esito |
|---|---|---|---|
| DeepInfra | [Terms of Service](https://deepinfra.com/terms), «Last modified: August 17th, 2026» | Nessun divieto sull'uso degli output. «Customer's use of Third-Party Content is at Customer's own risk and is governed exclusively by the applicable third-party license ... including, for open-source models, their applicable open-source license terms»; «Provider ... will not use Customer Data to train, fine-tune, or otherwise improve any model». Unico limite pertinente: non usare i Services «in any manner that is competitive with any business of Provider» (Scudo non è un cloud di inferenza). | compatibile |
| Novita | [Terms of Service](https://novita.ai/legal/terms-of-service), aggiornati il 5 agosto 2026 | Nessun divieto sull'uso degli output; licenza a Novita sugli output solo «for the purpose of providing the Service to you»; «Novita AI will not use your Content to train our own models». Divieto assoluto (6): «Attempt to circumvent the safety filters ... including by means of jailbreaking, prompt injection», più «automated safety screening» dei contenuti. | compatibile; rischio operativo: il pilot invia jailbreak e injection da classificare |
| Scaleway | [Specific Conditions AI Services](https://www-uploads.scaleway.com/Specific_Conditions_AI_Services_275907180f.pdf), versione del 07/04/2026 | «Inputs and Outputs processed during inference are considered data belonging to the Client»; «The Client agrees to review the terms of the license governing the use of the selected Model»; dati del cliente «not used to train, retrain, or improve the base Models». Obblighi da «Provider» o «Deployer» secondo il Regolamento (UE) 2024/1689. | compatibile (UE) |
| OVHcloud | [AI Endpoints, condizioni della fase di test](https://storage.gra.cloud.ovh.net/v1/AUTH_325716a587c64897acbef9a4a4726e38/contracts/48743bf-AI_Endpoints-ALL-1.1.pdf), versione dell'11 luglio 2024 | Rimanda alle «Third-Party Product Conditions» (licenze dei modelli); nessun divieto sugli output. Il documento è della fase di test: le condizioni del servizio attuale sono **NON VERIFICATE**. | probabile, da riverificare |
| Together | [Terms of Service](https://www.together.ai/terms-of-service), aggiornati il 19 maggio 2026 | «As between you and Company, you exclusively own all right, title, and interest in Your Content and Output»; per i Third-Party Models «the model terms govern». | compatibile |
| Baseten | [Terms and Conditions](https://www.baseten.co/terms-and-conditions/), aggiornati il 15 settembre 2026 | I «Model Outputs» sono «Customer Content»; Baseten non li usa per addestrare; limite «use the Services or the Documentation to build a competing product or service». | compatibile |
| Cerebras | [Terms of Use](https://www.cerebras.ai/terms-of-service), in vigore dal 27 agosto 2024 | Proprietà dell'Output «governed by the Third-Party Model Terms»; vietato «violate any of the model provider’s license and usage restrictions» e usare il servizio «to develop competing products or services, for benchmarking or competitive analysis of the Service»; vieta anche di trasmettere materiale che «promotes hatred, violence, or harm». | compatibile sugli output; rischio operativo sui contenuti |
| Featherless | [Terms of Service](https://featherless.ai/terms), aggiornati il 10 giugno 2024 | Le risposte dei modelli («Model-Generated Content») sono «Your Content. You retain ownership of Your Content.» | compatibile; servizio «in beta», prezzo non esposto dal router |
| Groq | [Terms of Use e Service Specific Terms](https://groq.com/groq-service-specific-terms) | Il Services Agreement (console.groq.com/docs/legal/services-agreement) risponde 403: clausole su Output **NON VERIFICATE**. | non verificato |
| Fireworks | fireworks.ai/terms-of-service | Pagina resa solo via JavaScript: testo **NON VERIFICATO**. Un riassunto di terzi (ConductAtlas, fonte non primaria) segnala un divieto di usare contenuti e output della piattaforma per addestrare modelli: da escludere finché non si legge il testo. | non verificato, da evitare |
| Nscale | docs.nscale.com/docs/legal/terms-of-service | Pagina protetta da codice di accesso: **NON VERIFICATO**. | non verificato |

## Raccomandazione

Per entrambi i teacher: **DeepInfra**, fissato nel nome del modello
(`openai/gpt-oss-120b:deepinfra`, `Qwen/Qwen3.5-397B-A17B:deepinfra`), con
`terms_url = "https://deepinfra.com/terms"` nel TOML del pilot.

- Termini: nessun limite sugli output; per i modelli open-source rimanda alla
  loro licenza, Apache-2.0 per entrambi; non addestra sui dati del cliente.
- Prezzo: il più basso tra i provider con termini letti, per entrambi i modelli
  (gpt-oss-120b 0,037 / 0,17; Qwen3.5-397B 0,45 / 3,00 USD per Mtok).

Alternative, se serve un secondo provider o la residenza dei dati nell'UE:

- `gpt-oss-120b`: **Together** (0,15 / 0,60), che attribuisce espressamente al
  cliente la proprietà esclusiva dell'Output.
- `Qwen3.5-397B-A17B`: **Scaleway** (0,684 / 4,104), condizioni UE che
  attribuiscono gli Output al cliente.

Da evitare finché i termini non sono letti: Fireworks, Groq, Nscale. Novita e
Cerebras hanno termini compatibili sugli output, ma clausole d'uso che un
corpus di jailbreak e contenuti dannosi può far scattare.

## Rischi aperti

- Con fatturazione Hugging Face non è documentato quali termini del provider
  si applichino: vedi sopra.
- I prezzi del router cambiano: la stima del pilot va rifatta con i prezzi del
  giorno, mai con questa tabella.
- La revisione dei pesi servita da DeepInfra non è dichiarata dal router:
  `served_revision` resta NON VERIFICATO.
- Nessuna run è stata eseguita: l'account non ha credito prepagato.
