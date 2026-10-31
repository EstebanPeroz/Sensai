from dataclasses import replace
from datetime import UTC, datetime

from sqlalchemy import Engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, defer

from sensai_server.llm.responses import ChatMetrics
from sensai_server.memory.persistent_db.error import DatabaseError
from sensai_server.memory.persistent_db.sql_alchemy.table import BranchTable, InteractionTable
from sensai_server.monitoring.interaction import Interaction


class SqlInteractionRepository:
    """`InteractionRepository` stored in the SQL database of the `SqlAlchemy` adapter, sharing its engine."""

    _engine: Engine
    _content_columns = (
        InteractionTable.request,
        InteractionTable.response,
        InteractionTable.thinking,
        InteractionTable.tool_calls,
    )

    def __init__(self, engine: Engine) -> None:
        """Store interactions through `engine`, whose tables already exist."""
        self._engine = engine

    def add(self, interaction: Interaction) -> None:
        """Store `interaction`, the record of one call to the model.

        A branch id that matches no stored branch is dropped instead of failing the insert.

        Raises:
            DatabaseError: If it could not be stored.

        """
        metrics = interaction.metrics
        with Session(self._engine) as session:
            try:
                branch_id = interaction.branch_id
                if branch_id is not None and session.get(BranchTable, branch_id) is None:
                    branch_id = None
                session.add(
                    InteractionTable(
                        id=interaction.id,
                        created_at=interaction.created_at.astimezone(UTC),
                        model=interaction.model,
                        status=interaction.status,
                        latency_ms=interaction.latency_ms,
                        branch_id=branch_id,
                        persona=interaction.persona,
                        request=interaction.request,
                        response=interaction.response,
                        thinking=interaction.thinking,
                        tool_calls=interaction.tool_calls,
                        error=interaction.error,
                        first_token_ms=interaction.first_token_ms,
                        prompt_tokens=metrics.prompt_tokens,
                        completion_tokens=metrics.completion_tokens,
                        total_duration_ms=metrics.total_duration_ms,
                        load_duration_ms=metrics.load_duration_ms,
                        prompt_eval_duration_ms=metrics.prompt_eval_duration_ms,
                        eval_duration_ms=metrics.eval_duration_ms,
                    )
                )
                session.commit()
            except SQLAlchemyError:
                msg = "the interaction could not be stored"
                raise DatabaseError(msg) from None

    def get_all(self, *, since: datetime | None = None, with_content: bool = True) -> list[Interaction]:
        """Return the stored interactions created at or after `since` (every one if None), oldest first.

        Without content, the request, response, thinking and tool calls are not even read.
        Dates are stored in UTC: SQLite keeps the wall time and drops the timezone, so both sides are
        converted to UTC before being compared.

        Raises:
            DatabaseError: If they could not be read.

        """
        query = select(InteractionTable).order_by(InteractionTable.created_at)
        if since is not None:
            query = query.where(InteractionTable.created_at >= since.astimezone(UTC))
        if not with_content:
            query = query.options(*(defer(column, raiseload=True) for column in self._content_columns))
        with Session(self._engine) as session:
            try:
                return [self._interaction_from_row(row, with_content=with_content) for row in session.scalars(query)]
            except SQLAlchemyError:
                msg = "the interactions could not be read"
                raise DatabaseError(msg) from None

    @staticmethod
    def _interaction_from_row(row: InteractionTable, *, with_content: bool = True) -> Interaction:
        """Convert a stored interaction back into an Interaction (SQLite drops the timezone: it is UTC)."""
        created_at = row.created_at if row.created_at.tzinfo else row.created_at.replace(tzinfo=UTC)
        interaction = Interaction(
            id=row.id,
            created_at=created_at,
            model=row.model,
            status=row.status,
            latency_ms=row.latency_ms,
            branch_id=row.branch_id,
            persona=row.persona,
            error=row.error,
            first_token_ms=row.first_token_ms,
            metrics=ChatMetrics(
                prompt_tokens=row.prompt_tokens,
                completion_tokens=row.completion_tokens,
                total_duration_ms=row.total_duration_ms,
                load_duration_ms=row.load_duration_ms,
                prompt_eval_duration_ms=row.prompt_eval_duration_ms,
                eval_duration_ms=row.eval_duration_ms,
            ),
        )
        if not with_content:
            return interaction
        return replace(
            interaction, request=row.request, response=row.response, thinking=row.thinking, tool_calls=row.tool_calls
        )
