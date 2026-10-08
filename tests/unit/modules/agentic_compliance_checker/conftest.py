import pytest
from cryptography.fernet import Fernet

_TEST_KEY = Fernet.generate_key().decode()


@pytest.fixture(autouse=True)
def _encryption_key(monkeypatch):
    """Sessions store the gptkey encrypted and refuse to open without a key."""
    monkeypatch.setenv("AGENTIC_COMPLIANCE_ENCRYPTION_KEY", _TEST_KEY)
