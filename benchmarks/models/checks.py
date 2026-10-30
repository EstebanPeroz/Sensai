"""Scoring of the answers of the model: each check reads an answer and says whether it passes."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

# Common words of each language, told apart by how many of them an answer holds.
FRENCH_WORDS = frozenset(
    (
        *("le", "la", "les", "des", "un", "une", "et", "est", "sont", "pour", "pas", "que", "qui", "dans", "du"),
        *("au", "aux", "avec", "sur", "vous", "votre", "ce", "cette", "en", "plus"),
    )
)
ENGLISH_WORDS = frozenset(
    (
        *("the", "a", "an", "and", "is", "are", "for", "not", "that", "which", "in", "of", "with", "on", "you"),
        *("your", "to", "this", "it", "more"),
    )
)
BULLET = re.compile(r"^\s*(?:[-*•+]|\d+[.)])\s+\S", re.MULTILINE)
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
# A word, with its apostrophes and hyphens, straight or typographic (U+2019).
WORD = re.compile(r"[\w'\u2019-]+")


def matches(answer: str, patterns: Iterable[str]) -> bool:
    """Whether one of the regular expressions `patterns` is found in `answer`, ignoring case.

    Without any pattern, every answer passes: the case only checks something else, such as the tool called.
    """
    patterns = list(patterns)
    return not patterns or any(re.search(pattern, answer, re.IGNORECASE) for pattern in patterns)


def language(answer: str) -> str | None:
    """Guess whether `answer` is written in French (`fr`) or English (`en`) from its most common words."""
    words = [word.lower() for word in WORD.findall(answer)]
    french = sum(word in FRENCH_WORDS for word in words)
    english = sum(word in ENGLISH_WORDS for word in words)
    if french == english:
        return None
    return "fr" if french > english else "en"


def bullet_count(answer: str) -> int:
    """Count the items of the bulleted or numbered lists of `answer`."""
    return len(BULLET.findall(answer))


def json_object(answer: str) -> dict | None:
    """Return the JSON object `answer` holds, alone or in a code block, None when there is none."""
    found = JSON_OBJECT.search(answer)
    if found is None:
        return None
    try:
        value = json.loads(found.group())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def word_count(answer: str) -> int:
    """Count the words of `answer`."""
    return len(WORD.findall(answer))


def passes(answer: str, check: Mapping[str, object]) -> bool:
    """Whether `answer` passes `check`, a table of the cases file with its `kind` and `value`.

    Raises:
        ValueError: If the kind of check is unknown.

    """
    kind, value = check["kind"], check["value"]
    match kind:
        case "language":
            return language(answer) == value
        case "bullets":
            return bullet_count(answer) == value
        case "json_keys":
            found = json_object(answer)
            return found is not None and isinstance(value, list) and set(value) <= found.keys()
        case "max_words":
            return isinstance(value, int) and word_count(answer) <= value
    msg = f"unknown check: {kind}"
    raise ValueError(msg)


def valid_arguments(arguments: object, schema: Mapping[str, object]) -> bool:
    """Whether `arguments`, given by the model to a tool, hold every argument its JSON `schema` requires."""
    if not isinstance(arguments, dict):
        return False
    required = schema.get("required", [])
    if not isinstance(required, list):
        return True
    return all(isinstance(arguments.get(name), str) and arguments[name].strip() for name in required)
