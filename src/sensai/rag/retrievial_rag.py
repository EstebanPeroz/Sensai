class Retrieval:
    """Answer a question from the chunks stored by the indexation."""

    def encode(self) -> None:
        """Turn the question into a vector, with the model that indexed the chunks."""

    def research(self) -> None:
        """Return the stored chunks closest to the encoded question."""

    def create_prompt(self) -> None:
        """Build the prompt that carries the question and the retrieved chunks."""

    def generation(self) -> None:
        """Ask the model to answer from the prompt."""

    def get_sources(self) -> None:
        """Report which documents the retrieved chunks came from."""
