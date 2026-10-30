from benchmarks.models.report import mean_seconds, render, share

RESULT = {
    "model": "tiny:1b",
    "size_gb": 1.2,
    "gpu_share": 1.0,
    "latency": {
        "cold": {"load_s": 2.5, "first_token_s": 3.1},
        "short": {"first_token_s": 0.2, "completion_tokens_per_s": 80.0},
        "long": {"prompt_tokens": 4100, "prompt_tokens_per_s": 900.0, "first_token_s": 4.6},
    },
    "quality": [{"id": "capital", "passed": True, "seconds": 2.0}, {"id": "gold", "passed": False, "seconds": 4.0}],
    "tools": [
        {"id": "weather", "passed": True, "right_tool": True, "expected_tool": "web_search", "called": ["web_search"]},
        {"id": "greeting", "passed": False, "right_tool": False, "expected_tool": None, "called": ["ask_user"]},
    ],
    "instructions": [{"id": "json-only", "passed": True, "seconds": 3.0}],
    "memory": {
        "full": [{"id": "q1", "passed": True, "prompt_tokens": 1500}],
        "compressed": [{"id": "q1", "passed": False, "prompt_tokens": 400}],
        "compression": {"messages": 18, "tokens_before": 1300, "tokens_after": 150, "seconds": 6.0},
    },
}


def test_share_counts_the_true_values() -> None:
    assert share(RESULT["quality"]) == "1/2"
    assert share(RESULT["tools"], "right_tool") == "1/2"
    assert share([]) == "-"


def test_mean_seconds() -> None:
    assert mean_seconds(RESULT["quality"]) == "3.0 s"


def test_render_has_a_row_per_model_in_every_table() -> None:
    text = render([RESULT, {"model": "empty:7b"}])

    assert "| `tiny:1b` | 1.2 GB | 100% | 80 tok/s | 1/2 | 1/2 | 1/2 | 1/1 | 1/1 | 0/1 | 3.0 s |" in text
    assert "| `empty:7b` | - | - | - | - | - | - | - | - | - | - |" in text
    assert "| greeting | none | ✗ ask_user | - |" in text
    assert "| `tiny:1b` | 18 | 1300 tok | 150 tok | 6 s | 1500 tok | 400 tok |" in text
