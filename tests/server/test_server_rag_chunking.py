from itertools import pairwise

import pytest
from sensai_server.llm.error import EmbeddingError
from sensai_server.rag.chunking import Chunker, ChunkStrategy

from tests.server.fakes import FakeEmbedder

CATS_THEN_ENGINES = (
    "The cat sleeps all day. It purrs when you pet it. Cats sleep sixteen hours. "
    "The engine burns fuel. The engine efficiency is low. The engine wastes heat."
)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def chunker(embedder: FakeEmbedder) -> Chunker:
    return Chunker(embedder)


def many_sentences(count: int) -> str:
    """A text of `count` eight-word sentences, long enough to need several chunks."""
    return " ".join(f"Sentence number {i} holds a few words here." for i in range(count))


class TestClean:
    def test_strips_html_tags(self, chunker: Chunker) -> None:
        assert chunker.clean("<p>hello <b>world</b></p>") == "hello world"

    def test_removes_invisible_characters(self, chunker: Chunker) -> None:
        assert chunker.clean("a\u200bb\ufeffc\xadd") == "abcd"

    def test_collapses_spaces_and_tabs(self, chunker: Chunker) -> None:
        assert chunker.clean("hello \t  world") == "hello world"

    def test_collapses_runs_of_blank_lines(self, chunker: Chunker) -> None:
        assert chunker.clean("a\n\n\n\nb") == "a\n\nb"

    def test_strips_every_line(self, chunker: Chunker) -> None:
        assert chunker.clean("  a  \n  b  ") == "a\nb"

    def test_keeps_markdown_headings(self, chunker: Chunker) -> None:
        assert chunker.clean("# Title\nbody") == "# Title\nbody"

    def test_empty_text_stays_empty(self, chunker: Chunker) -> None:
        assert chunker.clean("") == ""


class TestGetSentences:
    def test_splits_on_sentence_endings(self, chunker: Chunker) -> None:
        assert chunker._get_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]

    def test_splits_before_accented_capitals(self, chunker: Chunker) -> None:
        assert chunker._get_sentences("Un chat. Élodie arrive.") == ["Un chat.", "Élodie arrive."]

    def test_does_not_split_before_a_lowercase_letter(self, chunker: Chunker) -> None:
        assert chunker._get_sentences("Mr. smith waits.") == ["Mr. smith waits."]

    def test_empty_text_yields_no_sentence(self, chunker: Chunker) -> None:
        assert chunker._get_sentences("   ") == []


class TestSentenceChunk:
    def test_keeps_a_short_text_whole(self, chunker: Chunker) -> None:
        assert chunker.sentence_chunk("One. Two.") == ["One. Two."]

    def test_cuts_once_the_word_budget_is_exceeded(self, chunker: Chunker) -> None:
        chunks = chunker.sentence_chunk(many_sentences(10))

        assert len(chunks) > 1
        assert all(len(chunk.split()) <= 50 for chunk in chunks)

    def test_every_sentence_appears_in_a_chunk(self, chunker: Chunker) -> None:
        text = many_sentences(10)

        chunks = chunker.sentence_chunk(text)

        assert all(sentence in " ".join(chunks) for sentence in chunker._get_sentences(text))

    def test_the_chunks_stay_in_order(self, chunker: Chunker) -> None:
        chunks = chunker.sentence_chunk(many_sentences(10))

        assert chunks[0].startswith("Sentence number 0")
        assert chunks[-1].endswith("Sentence number 9 holds a few words here.")

    def test_empty_text_yields_no_chunk(self, chunker: Chunker) -> None:
        assert chunker.sentence_chunk("") == []


class TestSentenceChunkOverlap:
    def test_consecutive_chunks_share_their_boundary_sentence(self, chunker: Chunker) -> None:
        chunks = chunker.sentence_chunk(many_sentences(10))

        for before, after in pairwise(chunks):
            assert chunker._get_sentences(before)[-1] == chunker._get_sentences(after)[0]

    def test_no_overlap_makes_the_chunks_disjoint(self, chunker: Chunker) -> None:
        chunks = chunker.sentence_chunk(many_sentences(10), overlap=0)

        for before, after in pairwise(chunks):
            assert chunker._get_sentences(before)[-1] != chunker._get_sentences(after)[0]

    def test_no_overlap_keeps_the_text_intact(self, chunker: Chunker) -> None:
        text = many_sentences(10)

        assert " ".join(chunker.sentence_chunk(text, overlap=0)) == text

    def test_a_wider_overlap_repeats_more_sentences(self, chunker: Chunker) -> None:
        text = many_sentences(10)

        words = [len(" ".join(chunker.sentence_chunk(text, overlap=n)).split()) for n in (0, 1, 2)]

        assert words[0] < words[1] < words[2]

    def test_a_single_chunk_has_nothing_to_overlap(self, chunker: Chunker) -> None:
        assert chunker.sentence_chunk("One. Two.", overlap=2) == ["One. Two."]

    def test_the_budget_still_holds_with_an_overlap(self, chunker: Chunker) -> None:
        chunks = chunker.sentence_chunk(many_sentences(20), overlap=2)

        assert all(len(chunk.split()) <= 50 for chunk in chunks)


class TestMarkdownChunk:
    def test_one_section_per_heading(self, chunker: Chunker) -> None:
        sections = chunker.markdown_chunk("# A\nbody a\n# B\nbody b")

        assert sections == [{"title": "A", "text": "body a"}, {"title": "B", "text": "body b"}]

    def test_text_before_the_first_heading_is_the_preamble(self, chunker: Chunker) -> None:
        sections = chunker.markdown_chunk("intro\n# A\nbody")

        assert sections[0] == {"title": "(preamble)", "text": "intro"}

    def test_skips_headings_with_an_empty_body(self, chunker: Chunker) -> None:
        assert chunker.markdown_chunk("# A\n# B\nbody b") == [{"title": "B", "text": "body b"}]

    def test_text_without_any_heading_is_one_preamble(self, chunker: Chunker) -> None:
        assert chunker.markdown_chunk("just text") == [{"title": "(preamble)", "text": "just text"}]


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self, chunker: Chunker) -> None:
        assert chunker._cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_scale_does_not_matter(self, chunker: Chunker) -> None:
        assert chunker._cosine_similarity([1.0, 1.0], [5.0, 5.0]) == pytest.approx(1.0)

    def test_perpendicular_vectors_score_zero(self, chunker: Chunker) -> None:
        assert chunker._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_score_minus_one(self, chunker: Chunker) -> None:
        assert chunker._cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_a_zero_vector_scores_zero_instead_of_dividing_by_zero(self, chunker: Chunker) -> None:
        assert chunker._cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_mismatched_dimensions_raise(self, chunker: Chunker) -> None:
        with pytest.raises(ValueError, match="argument 2 is shorter"):
            chunker._cosine_similarity([1.0, 2.0], [1.0])


class TestAdaptiveThreshold:
    def test_is_the_mean_minus_one_standard_deviation(self, chunker: Chunker) -> None:
        assert chunker.adaptive_threshold([0.0, 1.0]) == pytest.approx(0.0)

    def test_needs_at_least_two_scores(self, chunker: Chunker) -> None:
        assert chunker.adaptive_threshold([0.4]) == 0.0
        assert chunker.adaptive_threshold([]) == 0.0


class TestFindBreakpoints:
    def test_returns_the_indices_below_the_threshold(self, chunker: Chunker) -> None:
        assert chunker.find_breakpoints([0.9, 0.2, 0.8], 0.5) == [1]

    def test_the_comparison_is_strict(self, chunker: Chunker) -> None:
        assert chunker.find_breakpoints([0.5], 0.5) == []

    def test_no_score_yields_no_breakpoint(self, chunker: Chunker) -> None:
        assert chunker.find_breakpoints([], 0.5) == []


class TestSplitOnBreakpoints:
    def test_cuts_after_each_breakpoint(self, chunker: Chunker) -> None:
        assert chunker.split_on_breakpoints(["a", "b", "c", "d"], {1}) == ["a b", "c d"]

    def test_without_a_breakpoint_everything_is_one_chunk(self, chunker: Chunker) -> None:
        assert chunker.split_on_breakpoints(["a", "b"], set()) == ["a b"]

    def test_a_breakpoint_on_the_last_sentence_adds_no_empty_chunk(self, chunker: Chunker) -> None:
        assert chunker.split_on_breakpoints(["a", "b"], {1}) == ["a b"]

    def test_no_sentence_yields_no_chunk(self, chunker: Chunker) -> None:
        assert chunker.split_on_breakpoints([], {0}) == []


class TestConsecutiveSimilarities:
    def test_yields_one_score_less_than_the_sentence_count(self, chunker: Chunker) -> None:
        assert len(chunker.consecutive_similarities(["a", "b", "c"])) == 2

    def test_a_single_sentence_has_no_score(self, chunker: Chunker) -> None:
        assert chunker.consecutive_similarities(["only"]) == []

    def test_embeds_every_sentence_in_one_call(self, chunker: Chunker, embedder: FakeEmbedder) -> None:
        chunker.consecutive_similarities(["a", "b", "c"])

        assert embedder.batches == [["a", "b", "c"]]


class TestSemanticChunk:
    def test_cuts_where_the_topic_changes(self, chunker: Chunker) -> None:
        chunks = chunker.semantic_chunk(CATS_THEN_ENGINES)

        assert len(chunks) == 2
        assert chunks[0].startswith("The cat sleeps")
        assert chunks[1].startswith("The engine burns")

    def test_a_single_topic_stays_whole(self, chunker: Chunker) -> None:
        assert chunker.semantic_chunk("The cat sleeps. It purrs. Cats rest.") == [
            "The cat sleeps. It purrs. Cats rest."
        ]

    def test_a_single_sentence_needs_no_embedding(self, chunker: Chunker, embedder: FakeEmbedder) -> None:
        assert chunker.semantic_chunk("Only one.") == ["Only one."]
        assert embedder.batches == []

    def test_empty_text_yields_no_chunk(self, chunker: Chunker) -> None:
        assert chunker.semantic_chunk("") == []


class TestToChunks:
    def test_auto_uses_markdown_for_a_md_name(self, chunker: Chunker) -> None:
        chunks = chunker.to_chunks("# Engines\nThe engine burns fuel.\n# Cats\nThe cat sleeps.", "doc/a.md")

        assert chunks == ["Engines\n\nThe engine burns fuel.", "Cats\n\nThe cat sleeps."]

    def test_the_extension_is_matched_whatever_its_case(self, chunker: Chunker) -> None:
        assert chunker.to_chunks("# Cats\nThe cat sleeps.", "A.MD") == ["Cats\n\nThe cat sleeps."]

    def test_auto_uses_semantic_for_anything_else(self, chunker: Chunker) -> None:
        assert len(chunker.to_chunks(CATS_THEN_ENGINES, "a.txt")) == 2

    def test_auto_uses_semantic_for_a_url(self, chunker: Chunker) -> None:
        assert chunker.to_chunks("# Title\nThe cat sleeps.", "https://example.com/page") == ["# Title The cat sleeps."]

    def test_the_markdown_title_prefixes_every_chunk_of_its_section(self, chunker: Chunker) -> None:
        chunks = chunker.to_chunks(f"# Mixed\n{CATS_THEN_ENGINES}", "a.md")

        assert len(chunks) == 2
        assert all(chunk.startswith("Mixed\n\n") for chunk in chunks)

    def test_a_forced_strategy_wins_over_the_name(self, chunker: Chunker) -> None:
        chunks = chunker.to_chunks("# Title\nOne. Two.", "a.md", strategy=ChunkStrategy.SENTENCE)

        assert chunks == ["# Title One. Two."]

    def test_the_sentence_strategy_makes_no_embedding_call(self, chunker: Chunker, embedder: FakeEmbedder) -> None:
        chunker.to_chunks(CATS_THEN_ENGINES, "a.txt", strategy=ChunkStrategy.SENTENCE)

        assert embedder.batches == []

    def test_the_text_is_cleaned_first(self, chunker: Chunker) -> None:
        assert chunker.to_chunks("<p>The cat\u200b sleeps.</p>", "a.txt") == ["The cat sleeps."]

    def test_an_empty_text_yields_no_chunk(self, chunker: Chunker, embedder: FakeEmbedder) -> None:
        assert chunker.to_chunks("  \n\t ", "a.txt") == []
        assert embedder.batches == []


class TestVectorizeBatch:
    def test_returns_one_vector_per_text(self, chunker: Chunker) -> None:
        assert len(chunker.vectorize_batch(["a", "b", "c"])) == 3

    def test_no_text_makes_no_call(self, chunker: Chunker, embedder: FakeEmbedder) -> None:
        assert chunker.vectorize_batch([]) == []
        assert embedder.batches == []

    def test_texts_are_sent_in_batches_of_eight(self, chunker: Chunker, embedder: FakeEmbedder) -> None:
        texts = [f"text {i}" for i in range(19)]

        chunker.vectorize_batch(texts)

        assert [len(batch) for batch in embedder.batches] == [8, 8, 3]
        assert [text for batch in embedder.batches for text in batch] == texts

    def test_an_embedding_error_is_raised_as_is(self) -> None:
        with pytest.raises(EmbeddingError, match="not available"):
            Chunker(FakeEmbedder(failing=True)).vectorize_batch(["a"])

    def test_a_failing_embedder_fails_the_whole_chunking(self) -> None:
        with pytest.raises(EmbeddingError):
            Chunker(FakeEmbedder(failing=True)).to_chunks("The cat sleeps. The engine burns fuel.", "a.txt")
