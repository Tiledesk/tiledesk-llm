# Test Suite - Tiledesk LLM

## Overview

This test suite follows a **pyramid testing strategy** with three layers:
- **Unit Tests** (60%): Isolated tests of services, utilities, and core functions with mocked dependencies
- **Integration Tests** (30%): API endpoint tests with mocked external services
- **E2E Tests** (10%): Complete workflow tests simulating real user interactions

## Directory Structure

```
tests/
├── unit/                          # Unit tests (mocked dependencies)
│   ├── modules/                  # Module-specific unit tests
│   │   ├── conversion/          # Conversion service tests
│   │   └── knowledge_graph/     # Knowledge graph service tests
│   ├── controller/              # Controller function tests
│   ├── shared/                  # Shared utility tests
│   └── tools/                   # Tool tests (reranker, multimodal)
├── integration/                  # Integration tests
│   ├── payloads/                # Test payloads for API endpoints
│   ├── test_conversion_api.py   # Conversion API tests
│   ├── test_main_api.py         # Main API endpoint tests
│   ├── test_knowledge_graph_api.py # Knowledge graph API tests
│   └── test_pdf_ocr_api.py      # PDF OCR API tests
├── e2e/                         # End-to-end workflow tests
│   ├── test_rag_flow.py         # Complete RAG conversation flow
│   └── test_conversion_flow.py  # Conversion + RAG integration
├── utils/                       # Utility scripts
│   ├── check_neo4j.py          # Neo4j connection checker
│   ├── check_nodes.py          # Graph node checker
│   ├── check_qdrant.py         # Qdrant connection checker
│   └── test_neo4j.py           # Neo4j integration script
└── conftest.py                 # Shared pytest fixtures
```

## Running Tests

### Prerequisites

Ensure all dependencies are installed:
```bash
poetry install
```

### Running All Tests

```bash
# Using pytest (recommended)
poetry run pytest tests/

# Using the test script (configured in pyproject.toml)
poetry run test

# With verbose output
poetry run pytest tests/ -v

# With coverage report
poetry run pytest tests/ --cov=tilellm --cov-report=html
```

### Running Specific Test Categories

```bash
# Unit tests only
poetry run pytest tests/unit/

# Integration tests only
poetry run pytest tests/integration/

# E2E tests only
poetry run pytest tests/e2e/

# Specific test module
poetry run pytest tests/unit/modules/conversion/test_services.py

# Specific test class
poetry run pytest tests/unit/modules/conversion/test_services.py::TestConversionServiceCore

# Specific test method
poetry run pytest tests/unit/modules/conversion/test_services.py::TestConversionServiceCore::test_encode_image_to_base64
```

### Running with Different Output Formats

```bash
# Detailed output
poetry run pytest tests/ -v

# Quiet mode
poetry run pytest tests/ -q

# Show test durations
poetry run pytest tests/ --durations=5

# Generate JUnit XML report (for CI/CD)
poetry run pytest tests/ --junitxml=test-results.xml
```

## Test Payloads

Integration tests use payload files located in `tests/integration/payloads/`. These are JSON files that simulate real API requests.

### Available Payloads

| File | Purpose | Used in |
|------|---------|---------|
| `conversion_pdf_to_text.json` | PDF to text conversion | `test_conversion_api.py` |
| `pdf_scrape.json` | PDF scraping request | `test_pdf_ocr_api.py` |
| `post_ask_to_agent.json` | Agent question request | `test_main_api.py` |
| `post_ask_to_llm.json` | Direct LLM question | `test_main_api.py` |
| `post_ask_with_memory.json` | RAG with memory | `test_main_api.py` |
| `post_ask_with_memory_chain.json` | RAG chain with memory | `test_main_api.py` |
| `qdrant_scrape_*.json` | Qdrant configuration examples | Infrastructure tests |
| `pinecone_*.json` | Pinecone configuration examples | Infrastructure tests |

### Using Payloads in Tests

Payloads are loaded using the `get_payload_path()` helper function:

```python
def get_payload_path(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), 'payloads', filename)

def test_example(client: TestClient):
    with open(get_payload_path('post_ask_to_llm.json'), 'r') as f:
        payload = json.load(f)
    
    response = client.post("/api/ask", json=payload)
    assert response.status_code != 404
```

## Test Types

### Unit Tests

Unit tests mock all external dependencies and focus on business logic:

- **Service Tests**: Test core business logic in isolation
- **Controller Tests**: Test request/response handling
- **Utility Tests**: Test helper functions and utilities
- **Model Tests**: Test data models and validation

Example unit test structure:
```python
class TestConversionServiceCore:
    @pytest.mark.asyncio
    async def test_process_xlsx_to_csv_core_success(self):
        # Mock external dependencies
        with patch('module.external_dependency', return_value=mock_data):
            # Execute core function
            result = _process_xlsx_to_csv_core("test.xlsx", excel_bytes)
            # Assert results
            assert len(result) == 2
```

### Integration Tests

Integration tests use the FastAPI TestClient with mocked external services:

- **API Endpoint Tests**: Verify endpoints are properly wired
- **Module Integration**: Test interaction between components
- **Error Handling**: Test error responses and status codes

Example integration test:
```python
def test_post_convert_pdf_to_text(client: TestClient):
    payload_path = get_payload_path('conversion_pdf_to_text.json')
    with open(payload_path, 'r') as f:
        payload = json.load(f)
    
    response = client.post("/api/convert", json=payload)
    assert response.status_code != 404
```

### E2E Tests

E2E tests simulate complete user workflows:

- **RAG Conversations**: Multi-turn conversations with memory
- **Conversion + RAG**: Document conversion followed by RAG query
- **Error Flows**: Testing error scenarios in complete flows

Example E2E test:
```python
def test_complete_rag_conversation(self, client: TestClient):
    # Step 1: Initial question
    response1 = client.post("/api/qa", json=payload1)
    assert response1.status_code == 200
    
    # Step 2: Follow-up using history
    payload2 = payload1.copy()
    payload2['chat_history_dict'] = response1.json()['chat_history_dict']
    response2 = client.post("/api/qa", json=payload2)
    assert response2.status_code == 200
```

## Utility Scripts

Utility scripts in `tests/utils/` help with manual testing and infrastructure validation:

| Script | Purpose |
|--------|---------|
| `check_neo4j.py` | Verify Neo4j connection and query nodes |
| `check_nodes.py` | Check graph node counts and properties |
| `check_qdrant.py` | Verify Qdrant connection and list namespaces |
| `test_neo4j.py` | Test Neo4j integration and metadata propagation |

Run utility scripts directly:
```bash
python tests/utils/check_qdrant.py
python tests/utils/check_neo4j.py
```

## Mocking Strategy

Tests use `unittest.mock` to isolate components:

### External Services Mocked

1. **LLM APIs** (OpenAI, Anthropic, etc.): Mocked to return predefined responses
2. **Vector Databases** (Qdrant, Pinecone): Mocked to return test documents
3. **Graph Databases** (Neo4j): Mocked to return test nodes/relationships
4. **External APIs** (MinIO, Redis): Mocked to avoid network calls

### Example Mock

```python
@patch('tilellm.shared.utility._get_llm')
def test_ask_to_llm(mock_get_llm, client: TestClient):
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    mock_get_llm.return_value = mock_llm
    
    response = client.post("/api/ask", json=payload)
    assert response.status_code == 200
```

## Continuous Integration

### GitHub Actions Example

Add to `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12"]
    
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install Poetry
        run: pip install poetry
      
      - name: Install dependencies
        run: poetry install
      
      - name: Run unit tests
        run: poetry run pytest tests/unit/ --junitxml=unit-test-results.xml
      
      - name: Run integration tests
        run: poetry run pytest tests/integration/ --junitxml=integration-test-results.xml
      
      - name: Upload test results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-results
          path: |
            unit-test-results.xml
            integration-test-results.xml
```

## Debugging Tests

### Enable Detailed Logging

```bash
export LOG_LEVEL=DEBUG
poetry run pytest tests/ -v
```

### Run with Python Debugger

```bash
python -m pdb -m pytest tests/unit/modules/conversion/test_services.py
```

### Check Test Coverage

```bash
poetry run pytest tests/ --cov=tilellm --cov-report=term-missing
```

## Best Practices

1. **Isolation**: Each test should be independent and not rely on other tests
2. **Mocking**: Mock all external dependencies to ensure test reliability
3. **Naming**: Use descriptive test names that indicate what is being tested
4. **Organization**: Follow the pyramid structure (60% unit, 30% integration, 10% E2E)
5. **Performance**: Keep tests fast by using mocks and avoiding real network calls

## Adding New Tests

### 1. Unit Tests
- Place in appropriate `tests/unit/` subdirectory
- Mock all external dependencies
- Focus on single function/class behavior

### 2. Integration Tests
- Place in `tests/integration/`
- Use the FastAPI TestClient fixture
- Add payloads to `tests/integration/payloads/` if needed

### 3. E2E Tests
- Place in `tests/e2e/`
- Simulate complete user workflows
- May use multiple API calls in sequence

### 4. Utility Scripts
- Place in `tests/utils/`
- Include clear usage instructions in docstrings
- Keep independent from test suite

## Troubleshooting

### Common Issues

1. **Import errors after moving files**: Update import paths or `sys.path` references
2. **Missing dependencies**: Ensure all test dependencies are in `pyproject.toml` dev group
3. **Mocking issues**: Verify mock targets match the actual import paths
4. **Test database connections**: Use utility scripts to verify infrastructure

### Getting Help

- Check test output for specific error messages
- Run tests with `-v` flag for verbose output
- Use `pytest --tb=short` for shorter tracebacks
- Refer to existing tests as examples

## Next Steps

After verifying tests pass:
1. Run the full test suite: `poetry run test`
2. Check coverage: `poetry run pytest --cov=tilellm --cov-report=html`
3. Update tests when adding new features
4. Add integration tests for new API endpoints
5. Create E2E tests for new user workflows