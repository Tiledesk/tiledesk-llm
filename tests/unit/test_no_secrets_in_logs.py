"""
The JWT carries the vector store credentials in its (only signed, not
encrypted) payload, and the webhook token authenticates our callbacks: neither
may reach the logs. Source scan, so a new log line reintroducing them fails.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2] / "tilellm"
LOG_CALL = re.compile(r"logger\.\w+\((?P<args>.*)")
LEAKS = re.compile(r"\{(token|raw_token|engine_dec|jwt|credentials)\b[^}]*\}|Raw token")


def test_no_log_line_interpolates_tokens_or_decoded_jwt():
    offenders = [
        f"{path.relative_to(ROOT.parent)}:{n}: {line.strip()}"
        for path in ROOT.rglob("*.py")
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if (m := LOG_CALL.search(line)) and LEAKS.search(m.group("args"))
    ]
    assert offenders == []
