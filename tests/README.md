# Test Suite - Tiledesk LLM

## Esecuzione dei Test

### Prerequisiti

Assicurati di avere tutte le dipendenze installate:

```bash
poetry install
```

### Eseguire i Test del Tool Multimodale

```bash
# Dalla root del progetto
python tests/test_multimodal_llm_tool.py
```

Output atteso:
```
================================================================================
UNIT TESTS - Tool Multimodale Interno
================================================================================

test_create_multimodal_tool (__main__.TestMultimodalLLMTool) ... ok
test_tool_schema (__main__.TestMultimodalLLMTool) ... ok
test_invoke_tool_text_only (__main__.TestMultimodalLLMTool) ... ok
test_invoke_tool_with_images (__main__.TestMultimodalLLMTool) ... ok
test_invoke_tool_with_documents (__main__.TestMultimodalLLMTool) ... ok
test_invoke_tool_multimodal_complex (__main__.TestMultimodalLLMTool) ... ok
test_invoke_tool_with_data_uri_prefix (__main__.TestMultimodalLLMTool) ... ok
test_invoke_tool_error_handling (__main__.TestMultimodalLLMTool) ... ok
test_create_provider_specific_tool (__main__.TestMultimodalLLMTool) ... ok
test_tool_with_empty_images_list (__main__.TestMultimodalLLMTool) ... ok
test_multimodal_input_validation (__main__.TestMultimodalLLMTool) ... ok
test_tool_message_construction (__main__.TestMultimodalToolIntegration) ... ok
test_realistic_workflow (__main__.TestMultimodalToolIntegration) ... ok

----------------------------------------------------------------------
Ran 13 tests in 0.XXXs

OK

================================================================================
✅ TUTTI I TEST SONO PASSATI
================================================================================
```

### Eseguire Test Specifici

```bash
# Test singolo
python -m unittest tests.test_multimodal_llm_tool.TestMultimodalLLMTool.test_invoke_tool_with_images

# Test di una classe specifica
python -m unittest tests.test_multimodal_llm_tool.TestMultimodalLLMTool
```

### Con pytest (opzionale)

Se preferisci usare pytest:

```bash
# Installa pytest
poetry add --group dev pytest pytest-asyncio

# Esegui i test
pytest tests/test_multimodal_llm_tool.py -v

# Con coverage
pytest tests/test_multimodal_llm_tool.py --cov=tilellm.tools.multimodal_llm_tool
```

## Struttura dei Test

### TestMultimodalLLMTool
Test unitari per il tool multimodale:
- Creazione e configurazione del tool
- Validazione dello schema
- Invocazione con diversi tipi di input
- Gestione errori

### TestMultimodalToolIntegration
Test di integrazione:
- Costruzione messaggi
- Workflow realistici con MCP
- Integrazione con LangChain

## Debugging

### Abilitare Log Dettagliati

```bash
# Linux/Mac
export LOG_LEVEL=DEBUG
python tests/test_multimodal_llm_tool.py

# Windows
set LOG_LEVEL=DEBUG
python tests/test_multimodal_llm_tool.py
```

### Eseguire con Python Debugger

```bash
python -m pdb tests/test_multimodal_llm_tool.py
```

## CI/CD Integration

### GitHub Actions

Aggiungi al tuo `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install
      - name: Run tests
        run: |
          poetry run python tests/test_multimodal_llm_tool.py
```

## Note

- I test usano mock per evitare chiamate reali agli LLM
- Non sono richieste API key per eseguire i test
- I test sono indipendenti e possono essere eseguiti in qualsiasi ordine

## Prossimi Passi

Dopo aver verificato che i test passino:
1. Leggi la documentazione completa: `MULTIMODAL_TOOL_README.md`
2. Prova un esempio reale con l'API: vedi sezione "Esempi di Utilizzo"
3. Configura i tuoi server MCP secondo le tue esigenze
