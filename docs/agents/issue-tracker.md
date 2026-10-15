# Issue tracker: GitHub

L'unico tracker operativo è GitHub Issues in `naicud/scudo`. Le skill di
ingegneria usano solo GitHub CLI (`gh`), eseguita da shell con prefisso
`rtk proxy`, per leggere e aggiornare le issue. Non usare file Markdown locali,
Linear o altri tracker per rappresentare ticket o stato del lavoro. Indicare
sempre `--repo naicud/scudo` quando si lavora sulle issue.

## Prima di scegliere o pubblicare lavoro

Leggere issue, commenti, label, milestone, sub-issue e dipendenze live.
Verificare il commit di partenza e gli eventuali blocker. Riconciliare le issue
esistenti prima di crearne altre.

## Operazioni

- Leggere: `rtk proxy gh issue view <numero> --repo naicud/scudo --comments`.
- Elencare: `rtk proxy gh issue list --repo naicud/scudo --state all --limit 100`;
  usare filtri e paginazione quando servono.
- Creare: `rtk proxy gh issue create --repo naicud/scudo --title "..." --body-file <file>`.
- Commentare: `rtk proxy gh issue comment <numero> --repo naicud/scudo --body-file <file>`.
- Modificare label: `rtk proxy gh issue edit <numero> --repo naicud/scudo --add-label "..."`
  o `--remove-label "..."`.
- Chiudere: `rtk proxy gh issue close <numero> --repo naicud/scudo --comment "..."`.

Per campi, relazioni native e paginazione non esposti dal comando, usare
`rtk proxy gh api`. Una sub-issue raggruppa il lavoro; una dipendenza dichiara un
blocco. Per gli endpoint delle dipendenze, usare l'ID numerico del database
dell'issue bloccante, non il numero `#` né il `node_id`. Se l'API delle
dipendenze non è disponibile, registrare `Blocked by: #<numero>` nell'issue e
verificare che il blocker sia chiuso prima di considerarla pronta.

## Convenzioni per le skill

Quando una skill chiede di pubblicare nel tracker, creare o aggiornare una issue
GitHub in questo repository. Quando chiede il ticket pertinente, leggere la
issue live con i commenti. Per `wayfinder`, usare una issue mappa con label
`wayfinder:map`, le sub-issue come ticket, le dipendenze native come blocker e
l'assegnazione al responsabile che prende in carico il lavoro. In assenza di
sub-issue native, collegare i ticket nella task list della mappa e indicare
`Part of #<mappa>` nel corpo del ticket. Verificare i blocker aperti prima di
scegliere il prossimo ticket. Creare le label specifiche solo quando il flusso
viene effettivamente usato.

**PR come superficie di richieste: no.** La triage delle richieste avviene sulle
issue; modificare questa opzione qui solo se il repository adotta esplicitamente
anche le PR esterne come richieste.
