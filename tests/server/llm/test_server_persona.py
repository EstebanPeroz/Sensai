from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sensai_protocol.error import SensaiError
from sensai_protocol.server_messages import Role
from sensai_server.llm.persona import (
    MAX_PROMPT_LENGTH,
    DuplicatePersonaError,
    InvalidPersonaError,
    Persona,
    PersonaError,
    PersonaFileError,
    PersonaManager,
    PersonaNotRegisteredError,
)
from sensai_server.memory import Message


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

    def test_to_message_is_the_system_prompt(self) -> None:
        assert make_persona().to_message() == Message(None, Role.SYSTEM, "You are an expert analyst.")


class TestFromToml:
    def test_reads_an_unsaved_persona(self) -> None:
        text = 'name = "Analyst"\ndescription = "an analytical assistant"\nprompt = "You are an expert analyst."\n'

        assert Persona.from_toml(text, "a.toml") == make_persona()

    def test_invalid_toml_is_a_file_error(self) -> None:
        with pytest.raises(PersonaFileError, match="invalid TOML"):
            Persona.from_toml("name = ", "a.toml")

    def test_missing_and_unknown_keys_are_reported_together(self) -> None:
        with pytest.raises(PersonaFileError) as err:
            Persona.from_toml('name = "a"\ndescription = "b"\nextra = "c"\n', "a.toml")

        assert err.value.reason == "missing key(s): prompt; unknown key(s): extra"

    def test_a_key_that_is_not_text_is_reported(self) -> None:
        with pytest.raises(PersonaFileError, match="must be text: prompt"):
            Persona.from_toml('name = "a"\ndescription = "b"\nprompt = 3\n', "a.toml")

    def test_an_empty_key_is_reported(self) -> None:
        with pytest.raises(PersonaFileError, match="empty key"):
            Persona.from_toml('name = " "\ndescription = "b"\nprompt = "c"\n', "a.toml")

    def test_a_too_long_prompt_is_refused(self) -> None:
        prompt = "x" * (MAX_PROMPT_LENGTH + 1)

        with pytest.raises(PersonaFileError, match="prompt longer than"):
            Persona.from_toml(f'name = "a"\ndescription = "b"\nprompt = "{prompt}"\n', "a.toml")

    def test_the_error_names_the_file(self) -> None:
        with pytest.raises(PersonaFileError) as err:
            Persona.from_toml("", "dir/a.toml")

        assert err.value.path == "dir/a.toml"
        assert "dir/a.toml" in str(err.value)


class TestPersonaErrors:
    @pytest.mark.parametrize(
        "error",
        [
            InvalidPersonaError(["name"]),
            PersonaNotRegisteredError("ghost", ["analyst"]),
            DuplicatePersonaError("analyst"),
            PersonaFileError("a.toml", "bad"),
        ],
    )
    def test_is_handled_as_a_sensai_error(self, error: PersonaError) -> None:
        assert isinstance(error, PersonaError)
        assert isinstance(error, SensaiError)

    def test_is_tagged_with_the_raised_class(self) -> None:
        assert str(InvalidPersonaError(["name"])).startswith("[InvalidPersonaError]")

    def test_not_registered_lists_the_available_names(self) -> None:
        error = PersonaNotRegisteredError("ghost", ["writer", "analyst"])

        assert error.persona_name == "ghost"
        assert "analyst, writer" in str(error)

    def test_not_registered_reports_an_empty_database(self) -> None:
        assert "none" in str(PersonaNotRegisteredError("ghost", []))

    def test_duplicate_names_the_taken_name(self) -> None:
        assert "analyst" in str(DuplicatePersonaError("analyst"))


class TestPersonaManager:
    def test_default_persona_is_the_packaged_analyst(self, persona_manager: PersonaManager) -> None:
        persona = persona_manager.create_default_persona()

        assert persona.uuid is not None
        assert persona.name == "Analyst"
        assert persona.prompt.startswith("You are an expert analyst.")

    def test_created_persona_is_stored(self, persona_manager: PersonaManager) -> None:
        created = persona_manager.create_persona(make_persona())

        assert created.uuid is not None
        assert persona_manager.get_personas() == [created]

    def test_a_taken_name_is_refused_whatever_its_case(self, persona_manager: PersonaManager) -> None:
        persona_manager.create_persona(make_persona())

        with pytest.raises(DuplicatePersonaError):
            persona_manager.create_persona(make_persona(name="ANALYST"))

    def test_get_persona_ignores_the_case(self, persona_manager: PersonaManager) -> None:
        created = persona_manager.create_persona(make_persona())

        assert persona_manager.get_persona("analyst") == created

    def test_unknown_name_lists_the_available_ones(self, persona_manager: PersonaManager) -> None:
        persona_manager.create_persona(make_persona())

        with pytest.raises(PersonaNotRegisteredError) as err:
            persona_manager.get_persona("ghost")

        assert err.value.available == ["Analyst"]

    def test_update_changes_only_the_given_fields(self, persona_manager: PersonaManager) -> None:
        created = persona_manager.create_persona(make_persona())

        updated = persona_manager.update_persona("Analyst", prompt="Be brief.")

        assert updated.uuid == created.uuid
        assert (updated.name, updated.description, updated.prompt) == ("Analyst", created.description, "Be brief.")

    def test_update_to_the_same_name_in_another_case_is_allowed(self, persona_manager: PersonaManager) -> None:
        persona_manager.create_persona(make_persona())

        assert persona_manager.update_persona("Analyst", new_name="ANALYST").name == "ANALYST"

    def test_update_to_a_name_another_persona_uses_is_refused(self, persona_manager: PersonaManager) -> None:
        persona_manager.create_persona(make_persona())
        persona_manager.create_persona(make_persona(name="Writer"))

        with pytest.raises(DuplicatePersonaError):
            persona_manager.update_persona("Writer", new_name="analyst")

        assert persona_manager.get_persona("Writer").name == "Writer"

    def test_update_of_an_unknown_persona_is_refused(self, persona_manager: PersonaManager) -> None:
        with pytest.raises(PersonaNotRegisteredError):
            persona_manager.update_persona("ghost", prompt="x")

    def test_update_from_draft_keeps_the_name_and_takes_the_rest(self, persona_manager: PersonaManager) -> None:
        persona_manager.create_persona(make_persona())

        updated = persona_manager.update_persona_from_draft(
            make_persona(name="analyst", description="new", prompt="new prompt")
        )

        assert (updated.name, updated.description, updated.prompt) == ("Analyst", "new", "new prompt")
