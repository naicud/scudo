# Label di triage

Le skill usano cinque ruoli canonici. Questa tabella indica le label GitHub da
usare in `naicud/scudo`. Verificare le label live prima di applicarle; se il
repository adotta nomi diversi, aggiornare qui la mappatura prima della triage,
evitando duplicati.

| Ruolo della skill | Label GitHub | Significato |
| --- | --- | --- |
| `needs-triage` | `needs-triage` | Richiede valutazione del maintainer |
| `needs-info` | `needs-info` | Attende informazioni dal segnalante |
| `ready-for-agent` | `ready-for-agent` | Specificata e pronta per un agente |
| `ready-for-human` | `ready-for-human` | Richiede implementazione umana |
| `wontfix` | `wontfix` | Non verrà svolta |

Prima di applicare `ready-for-agent`, verificare dipendenze, perimetro,
acceptance, dati, accessi (Hugging Face, Colab) e licenze richiesti dalla issue.
