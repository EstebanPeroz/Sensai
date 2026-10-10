from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from sensai.error import SensaiError
from sensai.llm.persona import (
    DEFAULT_PERSONA,
    DuplicatePersonaError,
    InvalidPersonaError,
    Persona,
    PersonaError,
    PersonaNotRegisteredError,
)


def make_persona(**overrides: object) -> Persona:
    fields: dict[str, object] = {
        "uuid": None,
        "name": "Analyst",
        "description": "an analytical assistant",
        "prompt": "You are an expert analyst.",
    }
    return Persona(**(fields | overrides))  # type: ignore[arg-type]


class TestPersona:
    def test_keeps_the_fields_it_is_built_from(self) -> None:
        persona = make_persona()

        assert persona.name == "Analyst"
        assert persona.description == "an analytical assistant"
        assert persona.prompt == "You are an expert analyst."

    def test_has_no_identity_until_it_is_stored(self) -> None:
        persona = make_persona()

        assert persona.uuid is None
        assert persona.created_at is None

    def test_carries_the_identity_of_a_stored_record(self) -> None:
        uuid, created_at = uuid4(), datetime(2026, 1, 2, 3, 4, tzinfo=UTC)

        persona = make_persona(uuid=uuid, created_at=created_at)

        assert persona.uuid == uuid
        assert persona.created_at == created_at

    def test_is_immutable(self) -> None:
        persona = make_persona()

        with pytest.raises(FrozenInstanceError):
            persona.prompt = "changed"  # type: ignore[misc]

    @pytest.mark.parametrize("field", ["name", "description", "prompt"])
    @pytest.mark.parametrize("value", ["", "   "])
    def test_rejects_a_missing_or_empty_field(self, field: str, value: str) -> None:
        with pytest.raises(InvalidPersonaError) as err:
            make_persona(**{field: value})

        assert err.value.invalid_fields == [field]
        assert field in str(err.value)

    def test_rejects_a_field_that_is_not_text(self) -> None:
        with pytest.raises(InvalidPersonaError):
            make_persona(description=None)

    def test_reports_every_invalid_field_at_once(self) -> None:
        with pytest.raises(InvalidPersonaError) as err:
            make_persona(name="", description="")

        assert err.value.invalid_fields == ["name", "description"]

    def test_show_names_its_fields(self) -> None:
        assert make_persona().show() == {"name": "Analyst", "description": "an analytical assistant"}

    def test_apply_to_payload_puts_the_prompt_first(self) -> None:
        messages = [{"role": "user", "content": "hi"}]

        assert make_persona().apply_to_payload(messages) == {
            "messages": [
                {"role": "system", "content": "You are an expert analyst."},
                {"role": "user", "content": "hi"},
            ]
        }

    def test_apply_to_payload_leaves_the_given_messages_alone(self) -> None:
        messages = [{"role": "user", "content": "hi"}]

        make_persona().apply_to_payload(messages)

        assert messages == [{"role": "user", "content": "hi"}]


class TestDefaultPersona:
    def test_is_not_stored_yet(self) -> None:
        assert DEFAULT_PERSONA.uuid is None

    def test_is_ready_to_be_seeded(self) -> None:
        assert DEFAULT_PERSONA.name == "Analyst"
        assert DEFAULT_PERSONA.prompt.startswith("You are an expert analyst.")


class TestPersonaErrors:
    @pytest.mark.parametrize(
        "error",
        [
            InvalidPersonaError(["name"]),
            PersonaNotRegisteredError("ghost", ["analyst"]),
            DuplicatePersonaError("analyst"),
        ],
    )
    def test_is_handled_as_a_sensai_error(self, error: PersonaError) -> None:
        assert isinstance(error, PersonaError)
        assert isinstance(error, SensaiError)

    def test_is_tagged_with_the_raised_class(self) -> None:
        assert InvalidPersonaError(["name"]).message.startswith("[Error][InvalidPersonaError]")

    def test_not_registered_lists_the_available_names(self) -> None:
        error = PersonaNotRegisteredError("ghost", ["writer", "analyst"])

        assert error.persona_name == "ghost"
        assert "analyst, writer" in str(error)

    def test_not_registered_reports_an_empty_database(self) -> None:
        assert "none" in str(PersonaNotRegisteredError("ghost", []))

    def test_duplicate_names_the_taken_name(self) -> None:
        assert "analyst" in str(DuplicatePersonaError("analyst"))
