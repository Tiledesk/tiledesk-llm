# Temporal Digest

Modulo generico per l'aggregazione temporale di flussi documentali e il query routing
intelligente tra modalità **temporale** e **semantica**.

Casi d'uso principali:
- Monitoraggio attività amministrativa (determine, delibere ASL / PA italiana)
- Surveillance documentale legale (contratti, sentenze)
- Report periodici su flussi di documenti tecnici o finanziari

---

## Architettura

```
Documento ingestato
        │
        ▼
  [Situated Context] ← profilo pa_italiana (o custom)
        │
        ├── Frase contestuale preposta al chunk  →  vettore dense+sparse
        └── Metadati strutturati (act_type, topics, amount, date, …)  →  payload vettoriale

Digest batch (notturno o on-demand)
        │
        ▼
  Recupera tutti i chunk del periodo  (filtro metadata: date range + digest_type ≠ "digest")
        │
        ├── [Hybrid search]  dense + sparse (opzionale)
        └── LLM sintetizza  →  documento "digest" indicizzato come vettore
                             con metadata: digest_type="digest", digest_date_from/to

Query
        │
   ┌────┴──────────────────────────┐
   │  /api/digest/query            │  /api/digest/qa  (agentico)
   │  (parametri espliciti)        │  (LLM estrae date + mode)
   └────────────┬──────────────────┘
                │
         Query Router (rule-based IT+EN  +  LLM agent)
                │
   ┌────────────┴─────────────────┐
   ▼                              ▼
temporal                       semantic
(recupera digest               (vector search
 per date range)                raw chunks,
                                 digest esclusi)
   │                              │
   └──────[reranking opz.]────────┘
                │
           risposta LLM
           (con storico conversazione)
```

---

## Componenti

| File | Responsabilità |
|---|---|
| `models/schemas.py` | `DigestGenerationRequest/Response`, `DigestQueryRequest/Response`, `DigestAgentRequest/Response` |
| `services/digest_service.py` | `DigestService`: `generate()`, `query()`, `agent_query()` |
| `services/query_router.py` | `classify_query()`, `classify_query_debug()` — pattern IT+EN per rilevare query temporali |
| `services/domain_prompts.py` | Prompt pre-costruiti: `pa_italiana`, `legal`, `generic` |
| `logic.py` | Entry point con `@inject_llm_chat_async` + `@inject_repo_async` |
| `controllers.py` | FastAPI: `POST /api/digest/generate`, `POST /api/digest/query`, `POST /api/digest/qa`, `GET /api/digest/{namespace}/{date}` |

---

## Situated Context + Estrazione Metadati

Il profilo `pa_italiana` (in `tilellm/shared/profiles/situated_context/pa_italiana.yaml`)
abilita la modalità **JSON dual-output**: una singola chiamata LLM per chunk produce:

```json
{
  "context": "Questa determina n. 453 del 01/05/2026 riguarda l'acquisto di farmaci...",
  "metadata": {
    "act_type": "acquisto_farmaci",
    "topics": ["chemioterapia", "farmaci oncologici"],
    "amount": 45000.00,
    "personnel_role": null,
    "temporal_scope": "2026"
  }
}
```

Il campo `context` viene preposto al testo del chunk (migliorando il retrieval vettoriale).
I campi `metadata` vengono scritti direttamente nel payload del vettore → **filtrabili** in Qdrant/Pinecone/Milvus.

**Campi standard** scritti senza prefisso:

| Campo | Tipo | Descrizione |
|---|---|---|
| `act_type` | `string` | Categoria atto (vedi valori ammessi sotto) |
| `topics` | `list[string]` | Argomenti principali del chunk |
| `amount` | `float` | Importo in euro, se presente |
| `personnel_role` | `string\|null` | Ruolo personale (medico, infermiere, …) |
| `temporal_scope` | `string\|null` | Periodo di riferimento dell'atto |
| `date` | `string` | Data documento ISO `YYYY-MM-DD` — chiave primaria per il filtro temporale |

Valori `act_type` per PA italiana:
`acquisto_beni`, `acquisto_farmaci`, `acquisto_dispositivi_medici`, `acquisto_servizi`,
`appalto`, `concessione`, `assunzione_personale`, `incarico_professionale`,
`liquidazione`, `rimborso`, `autorizzazione`, `altro`

Campi non riconosciuti vengono scritti con prefisso `sc_` per evitare collisioni.

---

## Campo `date` nei metadati chunk

Il campo `date` (ISO `YYYY-MM-DD`) nel payload di ogni chunk è la chiave usata dal
filtro temporale nei path di generazione e query. Può essere impostato in due modi:

**A. Tramite Situated Context** (automatico, da campo `doc_date` di `PDFScrapingRequest`):
```json
{ "doc_date": "2026-05-01" }
```
Il valore viene scritto come `date` nel payload di ogni chunk.

**B. Tramite `additional_metadata`** (manuale, per tutti i tipi di ingestione):
```json
{
  "additional_metadata": {
    "date": "14/05/2026",
    "ente": "ASL Brindisi"
  }
}
```
Il valore `date` in formato `DD/MM/YYYY` viene auto-convertito in ISO `YYYY-MM-DD`.
Le chiavi di `additional_metadata` vengono fuse nel payload di **ogni chunk** del documento.

---

## Digest Vectors

I digest generati vengono indicizzati nello **stesso namespace** dei documenti sorgente,
marcati con `digest_type="digest"` nel payload. Le query semantiche li escludono
automaticamente con filtro `$ne`.

Metadata di un digest vector:

| Campo | Valore esempio |
|---|---|
| `digest_type` | `"digest"` |
| `digest_date_from` | `"2026-05-01"` |
| `digest_date_to` | `"2026-05-01"` |
| `digest_granularity` | `"daily"` |
| `chunk_count` | `47` |
| `act_types_json` | `'{"acquisto_farmaci": 5, "assunzione_personale": 2}'` |
| `total_amount` | `120000.0` |
| `domain` | `"pa_italiana"` |

**Deduplicazione**: ogni generazione cancella il digest precedente per lo stesso
`metadata_id` prima di indicizzarne uno nuovo — non si accumulano duplicati.

---

## Query Router

`classify_query(question)` (o `classify_query_debug`) restituisce `"temporal"` o `"semantic"`.

Pattern che attivano la modalità **temporal**:
- Riferimenti temporali espliciti: `oggi`, `ieri`, `questa settimana`, `questo mese`, nomi di mesi, `dal ... al ...`
- Intento aggregativo: `riassunto`, `riepilogo`, `cosa hanno fatto`, `tipo di attività`, `quante determine`
- Equivalenti inglesi: `today`, `this week`, `summary`, `how many`

Tutto il resto → **semantic** (vector search su raw chunk, digest esclusi).

Il routing viene loggato:
```
[query_router] auto → 'temporal' | pattern=r'\bcosa\s+hanno\s+fatto\b' | question='Cosa hanno fatto il 10 aprile?'
[query_router] auto → 'semantic' | pattern=None | question='Hanno acquistato antibiotici?'
```

---

## Endpoint

### `POST /api/digest/generate`

Genera digest per un range di date.

Parametri principali:

| Campo | Tipo | Default | Descrizione |
|---|---|---|---|
| `namespace` | `str` | — | Namespace vettoriale |
| `date_from` | `date` | — | Inizio range |
| `date_to` | `date` | `date_from` | Fine range |
| `granularity` | `"daily"\|"weekly"\|"monthly"` | `"daily"` | Granularità finestre |
| `domain` | `str\|null` | `null` | Prompt dominio (`pa_italiana`, `legal`, `generic`) |
| `top_k` | `int` | `1000` | Max chunk per finestra |
| `force_regenerate` | `bool` | `false` | Rigenera anche se già esistente |
| `date_metadata_field` | `str` | `"date"` | Campo payload che contiene la data documento |
| `webhook_url` | `str\|null` | `null` | URL notificato al termine (TaskIQ async) |

Con `ENABLE_TASKIQ=true`, l'endpoint ritorna immediatamente `{"task_id": "...", "status": "queued"}`.

---

### `POST /api/digest/query`

Query con parametri espliciti.

| Campo | Tipo | Default | Descrizione |
|---|---|---|---|
| `question` | `str` | — | Domanda in linguaggio naturale |
| `date_from` | `date\|null` | `null` | Inizio range (opzionale) |
| `date_to` | `date\|null` | `null` | Fine range |
| `query_mode` | `"auto"\|"temporal"\|"semantic"` | `"auto"` | Routing esplicito o automatico |
| `top_k` | `int` | `5` | Numero risultati |
| `search_type` | `"similarity"\|"hybrid"` | `"similarity"` | Dense-only o dense+sparse |
| `sparse_encoder` | `str\|TEIConfig\|null` | `null` | Encoder sparse per hybrid search |
| `reranking` | `bool\|TEIConfig\|PineconeRerankerConfig` | `false` | Abilita reranking |
| `reranker_model` | `str` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Modello locale (quando `reranking=true`) |
| `reranking_multiplier` | `int` | `3` | Candidati = `top_k × multiplier` per il reranking |
| `chat_history_dict` | `dict\|null` | `null` | Storico conversazione (vedi sotto) |
| `max_history_messages` | `int` | `10` | Turni di storia da includere nel prompt |
| `date_metadata_field` | `str` | `"date"` | Campo payload che contiene la data documento |

**Risposta:**
```json
{
  "answer": "...",
  "query_mode": "temporal",
  "sources": [...],
  "digests_used": ["2026-05-01", "2026-05-02"],
  "chunk_count": 2
}
```

---

### `POST /api/digest/qa` — Agentic Query

Nessun bisogno di specificare date o query_mode: il LLM li estrae dalla domanda e dallo storico.

| Campo | Tipo | Default | Descrizione |
|---|---|---|---|
| `question` | `str` | — | Domanda libera |
| `today` | `date\|null` | data server | Data di riferimento per espressioni relative |
| `chat_history_dict` | `dict\|null` | `null` | Storico conversazione |
| *(tutti i campi di retrieval di `/api/digest/query`)* | | | `search_type`, `sparse_encoder`, `reranking`, ecc. |

**Risposta:**
```json
{
  "answer": "La settimana scorsa l'ASL Roma 1 ha emesso...",
  "query_mode": "temporal",
  "sources": [...],
  "extracted_date_from": "2026-04-27",
  "extracted_date_to": "2026-05-02",
  "extracted_query_mode": "temporal",
  "agent_reasoning": "domanda aggregativa con riferimento temporale relativo"
}
```

Il campo `agent_reasoning` spiega la scelta del LLM — utile per debug e audit.

---

### `GET /api/digest/{namespace}/{digest_date}`

Recupera il digest per un namespace e una data specifica (richiede che il digest sia già stato generato).

---

## Storico Conversazione

Il formato `chat_history_dict` è un dizionario con chiavi intere (come stringhe) che mappa
ciascun turno a `{question, answer}`:

```json
{
  "chat_history_dict": {
    "0": {
      "question": "Cosa hanno fatto ad aprile?",
      "answer": "Ad aprile l'ASL Roma 1 ha emesso 312 determine..."
    },
    "1": {
      "question": "E a marzo?",
      "answer": "A marzo sono state emesse 287 determine..."
    }
  }
}
```

Lo storico viene incluso nel prompt LLM **prima dell'evidence block**, consentendo risposte
coerenti con il contesto della conversazione (es. follow-up su periodi già discussi).

---

## Hybrid Search + Reranking

### Hybrid (dense + sparse)

```json
{
  "search_type": "hybrid",
  "sparse_encoder": "splade"
}
```

Oppure con encoder TEI remoto:

```json
{
  "search_type": "hybrid",
  "sparse_encoder": {
    "provider": "tei",
    "name": "splade-v3",
    "url": "http://tei-host:8081"
  }
}
```

### Reranking

```json
{
  "reranking": true,
  "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "reranking_multiplier": 3
}
```

Il sistema recupera `top_k × reranking_multiplier` candidati, poi il reranker
seleziona i `top_k` più rilevanti prima di passarli all'LLM.

Con TEI remoto:
```json
{
  "reranking": {
    "provider": "tei",
    "name": "bge-reranker-large",
    "url": "http://tei-host:8080"
  }
}
```

Con Pinecone Inference:
```json
{
  "reranking": {
    "provider": "pinecone",
    "api_key": "pcsk-...",
    "name": "bge-reranker-v2-m3"
  }
}
```

---

## Backend e `_metadata_filter`

Il modulo usa `_metadata_filter` su `QuestionAnswer` per filtrare per `digest_type` e range di date.
Il filtro è supportato su tutti i backend:

| Backend | Filtro date | Filtro `digest_type` |
|---|---|---|
| Qdrant local/cloud | `DatetimeRange` (ISO string) | `FieldCondition` |
| Pinecone serverless | filtro nativo MongoDB-style | merge con `$and` |
| Pinecone pod | filtro nativo MongoDB-style | merge con `$and` |
| Milvus | espressione stringa `metadata["field"] >= "value"` | `metadata["digest_type"] == "digest"` |

---

## Configurazione

```bash
ENABLE_TEMPORAL_DIGEST=true   # default true — abilita il modulo
ENABLE_TASKIQ=true            # default false — generazione asincrona via TaskIQ/Redis
```

Con TaskIQ abilitato, `POST /api/digest/generate` ritorna immediatamente con `task_id`.
Polling: `GET /api/enqueue/status/{task_id}`.

---

## Estendere con nuovi Domini

1. Aggiungere voce in `services/domain_prompts.py`:
```python
"mio_dominio": {
    "system": "...",
    "user_template": "...",
}
```

2. (Opzionale) Creare profilo situated context in
   `tilellm/shared/profiles/situated_context/mio_dominio.yaml`
   con `json_mode: true` e i campi metadata rilevanti.

3. Passare `domain="mio_dominio"` nelle richieste di generazione e
   `situated_context.profile="mio_dominio"` nell'ingestione.

---

## Note Operative

**Se il digest non esiste per una data:**
Il servizio restituisce `"Nessun digest trovato per il periodo richiesto."`.
→ Chiamare `POST /api/digest/generate` per creare il digest mancante.

**Quante LLM call per giornata?**
- Ingestione con `pa_italiana`: 1 call per chunk (il SC produce contesto + metadati in un'unica call)
- Digest giornaliero: 1 call per ASL × giorno (indipendente dal numero di determine)
- Query: 1 call per risposta (+ 1 call LLM per estrazione parametri se `POST /api/digest/qa`)

**`top_k` nella generazione digest:**
Default `1000`. Il budget dell'evidence block è 60.000 chars (~800 char × 75 chunk nel prompt;
gli extra contribuiscono alle statistiche ma vengono troncati nell'evidence).

**Rigenerazione:**
`force_regenerate: false` (default) skippa le date già elaborate.
Ogni rigenerazione cancella il digest precedente (stesso `metadata_id`) prima di indicizzarne uno nuovo.
