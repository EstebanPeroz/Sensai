from datetime import UTC

from sqlalchemy import Engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from sensai.llm.responses import ChatMetrics
from sensai.memory.persistent_db.error import DatabaseError
from sensai.memory.persistent_db.sql_alchemy.table import BranchTable, InteractionTable
from sensai.monitoring.interaction import Interaction


class SqlInteractionRepository:
    """`InteractionRepository` stored in the SQL database of the `SqlAlchemy` adapter, sharing its engine."""

    _engine: Engine

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
                        created_at=interaction.created_at,
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

    def get_all(self) -> list[Interaction]:
        """Return every stored interaction, oldest first."""
        with Session(self._engine) as session:
            rows = session.scalars(select(InteractionTable).order_by(InteractionTable.created_at)).all()
            return [self._interaction_from_row(row) for row in rows]

    @staticmethod
    def _interaction_from_row(row: InteractionTable) -> Interaction:
        """Convert a stored interaction back into an Interaction (SQLite drops the timezone: it is UTC)."""
        created_at = row.created_at if row.created_at.tzinfo else row.created_at.replace(tzinfo=UTC)
        return Interaction(
            id=row.id,
            created_at=created_at,
            model=row.model,
            status=row.status,
            latency_ms=row.latency_ms,
            branch_id=row.branch_id,
            persona=row.persona,
            request=row.request,
            response=row.response,
            thinking=row.thinking,
            tool_calls=row.tool_calls,
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
