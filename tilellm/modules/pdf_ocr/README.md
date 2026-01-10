# PDF OCR Module

Modulo per lo scraping di documenti PDF utilizzando Dolphin per l'estrazione di testo, tabelle, immagini e formule.

## Caratteristiche

- **Parsing completo**: Analisi del layout e ordine di lettura con Dolphin
- **Estrazione elementi**: Testo, tabelle, immagini, formule, codice
- **Markdown generation**: Conversione automatica in formato indicizzabile
- **Supporto multi-pagina**: Gestione completa di documenti multi-pagina
- **Base64/URL support**: Caricamento da file Base64 o URL
- **Elaborazione asincrona**: Pattern fire-and-forget per processi lunghi
- **Webhook support**: Notifiche automatiche al completamento

## API Endpoints

### POST /api/pdf/scrape

Sottomette un PDF per l'elaborazione asincrona.

**Request Body:**
```json
{
  "file_name": "document.pdf",
  "file_content": "base64_encoded_content_or_url",
  "include_images": true,
  "include_tables": true,
  "include_text": true,
  "max_batch_size": 16,
  "webhook_url": "https://example.com/webhook",
  "callback_token": "optional_auth_token"
}
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "accepted",
  "message": "PDF processing job submitted successfully",
  "estimated_time": 60
}
```

### GET /api/pdf/status/{job_id}

Verifica lo stato di un job di elaborazione.

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "pending|processing|completed|failed",
  "progress": 50,
  "message": "Status message",
  "result": {
    "file_name": "document.pdf",
    "total_pages": 10,
    "pages": [...],
    "markdown_content": "# Page 1\n\nContent...",
    "processing_time": 45.2,
    "metadata": {...}
  },
  "error_message": "Error description if failed"
}
```

### GET /api/pdf/health

Verifica lo stato del servizio PDF OCR.

**Response:**
```json
{
  "status": "healthy|degraded|unavailable",
  "dolphin_available": true,
  "job_service_available": true,
  "service": "PDF OCR",
  "message": "Optional status message"
}
```

## Utilizzo

### 1. Sottomettere un PDF per l'elaborazione

```python
import requests
import base64

# Carica il file PDF
with open('document.pdf', 'rb') as f:
    pdf_content = base64.b64encode(f.read()).decode('utf-8')

# Sottometti per l'elaborazione
response = requests.post(
    'http://localhost:8000/api/pdf/scrape',
    json={
        'file_name': 'document.pdf',
        'file_content': pdf_content,
        'webhook_url': 'https://your-app.com/webhook'
    }
)

job_id = response.json()['job_id']
print(f"Job submitted: {job_id}")
```

### 2. Verificare lo stato

```python
import requests

# Verifica lo stato del job
response = requests.get(f'http://localhost:8000/api/pdf/status/{job_id}')
status = response.json()

print(f"Status: {status['status']}")
print(f"Progress: {status.get('progress', 0)}%")

if status['status'] == 'completed':
    markdown_content = status['result']['markdown_content']
    print(f"Markdown content: {markdown_content[:200]}...")
```

### 3. Utilizzo con webhook

```python
# Il servizio invierà una notifica POST al webhook specificato
# quando l'elaborazione è completata

webhook_payload = {
    "job_id": "uuid-string",
    "status": "completed",
    "file_name": "document.pdf",
    "total_pages": 10,
    "processing_time": 45.2,
    "markdown_content": "# Page 1\n\nContent..."
}
```

## Elementi Estratti

Il servizio estrae i seguenti tipi di elementi:

- **Testo (para)**: Paragrafi di testo
- **Titoli (sec_0-sec_5)**: Titoli di diverse gerarchie
- **Tabelle (tab)**: Tabelle in formato HTML
- **Immagini (fig)**: Immagini con riferimenti ai file
- **Formule (equ)**: Formule matematiche in LaTeX
- **Codice (code)**: Blocchi di codice
- **Liste (list)**: Elementi di lista

## Configurazione

### Requisiti

- Python 3.12+
- Dolphin module (incluso in `tilellm/modules/pdf_ocr/Dolphin/`)
- Dipendenze FastAPI e Pydantic

### Dipendenze Dolphin

Il modulo Dolphin richiede:
- PyTorch
- Transformers
- OpenCV
- Pillow

## Note Tecniche

- **Elaborazione asincrona**: I PDF vengono processati in background
- **Gestione errori**: Errori dettagliati per ogni fase di elaborazione
- **Limitazione concorrenza**: Massimo 2 job processati contemporaneamente
- **Timeout**: Nessun timeout per elaborazioni lunghe
- **Memoria**: I job vengono mantenuti in memoria fino al riavvio

## Esempio di Risultato

```markdown
# Page 1

## Introduction

This is the introduction section of the document.

![Figure](figures/document_page_001_figure_001.png)

**Table:**
<table>
<tr><td>Header 1</td><td>Header 2</td></tr>
<tr><td>Data 1</td><td>Data 2</td></tr>
</table>

$$E = mc^2$$

---

# Page 2

## Methods

Detailed methodology section...
```