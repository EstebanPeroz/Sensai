import pytest

from benchmarks.models import checks


class TestMatches:
    def test_finds_one_of_the_patterns_ignoring_case(self) -> None:
        assert checks.matches("The capital is Canberra.", ["sydney", "canberra"])

    def test_fails_without_any_matching_pattern(self) -> None:
        assert not checks.matches("The capital is Sydney.", ["canberra"])

    def test_passes_without_patterns(self) -> None:
        assert checks.matches("anything", [])


class TestLanguage:
    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("Le vélo est rapide et il est bon pour la santé.", "fr"),
            ("Cycling is fast and it is good for the health.", "en"),
            ("Docker.", None),
        ],
    )
    def test_guesses_from_the_common_words(self, answer: str, expected: str | None) -> None:
        assert checks.language(answer) == expected


class TestPasses:
    @pytest.mark.parametrize(
        ("answer", "check", "expected"),
        [
            ("- one\n- two\n- three", {"kind": "bullets", "value": 3}, True),
            ("1. one\n2) two", {"kind": "bullets", "value": 3}, False),
            ('```json\n{"city": "Paris", "country": "France"}\n```', {"kind": "json_keys", "value": ["city"]}, True),
            ('{"city": "Paris"', {"kind": "json_keys", "value": ["city"]}, False),
            ("Blue.", {"kind": "max_words", "value": 2}, True),
            ("The sky is blue.", {"kind": "max_words", "value": 2}, False),
            ("C'est une plateforme de conteneurs.", {"kind": "language", "value": "fr"}, True),
        ],
    )
    def test_scores_each_kind(self, answer: str, check: dict, expected: bool) -> None:  # noqa: FBT001
        assert checks.passes(answer, check) is expected

    def test_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown check"):
            checks.passes("answer", {"kind": "rhymes", "value": True})


SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}


class TestValidArguments:
    @pytest.mark.parametrize(
        ("arguments", "expected"),
        [({"query": "tour de france"}, True), ({"query": " "}, False), ({}, False), ("query", False)],
    )
    def test_needs_every_required_argument(self, arguments: object, expected: bool) -> None:  # noqa: FBT001
        assert checks.valid_arguments(arguments, SCHEMA) is expected
