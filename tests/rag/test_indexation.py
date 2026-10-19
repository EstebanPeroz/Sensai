from itertools import pairwise
from pathlib import Path

import pytest

from sensai.rag.error import (
    EmbeddingError,
    RagError,
    SourceError,
    SourceFetchError,
    SourceNotFoundError,
    UnsupportedFormatError,
)
from sensai.rag.indexation_rag import ChunkStrategy, Indexation
from tests.rag.fakes import FakeEmbedder, FakeIndexation, SilentEmbedder

CATS_THEN_ENGINES = (
    "The cat sleeps all day. It purrs when you pet it. Cats sleep sixteen hours. "
    "The engine burns fuel. The engine efficiency is low. The engine wastes heat."
)


class TestClean:
    def test_strips_html_tags(self, indexation: Indexation) -> None:
        assert indexation.clean("<p>hello <b>world</b></p>") == "hello world"

    def test_removes_invisible_characters(self, indexation: Indexation) -> None:
        assert indexation.clean("a\u200bb\ufeffc\xadd") == "abcd"

    def test_collapses_spaces_and_tabs(self, indexation: Indexation) -> None:
        assert indexation.clean("hello \t  world") == "hello world"

    def test_collapses_runs_of_blank_lines(self, indexation: Indexation) -> None:
        assert indexation.clean("a\n\n\n\nb") == "a\n\nb"

    def test_strips_every_line(self, indexation: Indexation) -> None:
        assert indexation.clean("  a  \n  b  ") == "a\nb"

    def test_keeps_markdown_headings(self, indexation: Indexation) -> None:
        assert indexation.clean("# Title\nbody") == "# Title\nbody"

    def test_empty_text_stays_empty(self, indexation: Indexation) -> None:
        assert indexation.clean("") == ""


class TestGetSentences:
    def test_splits_on_sentence_endings(self, indexation: Indexation) -> None:
        assert indexation._get_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]

    def test_splits_before_accented_capitals(self, indexation: Indexation) -> None:
        assert indexation._get_sentences("Un chat. Élodie arrive.") == ["Un chat.", "Élodie arrive."]

    def test_does_not_split_before_a_lowercase_letter(self, indexation: Indexation) -> None:
        assert indexation._get_sentences("Mr. smith waits.") == ["Mr. smith waits."]

    def test_empty_text_yields_no_sentence(self, indexation: Indexation) -> None:
        assert indexation._get_sentences("   ") == []


def many_sentences(count: int) -> str:
    """A text of `count` eight-word sentences, long enough to need several chunks."""
    return " ".join(f"Sentence number {i} holds a few words here." for i in range(count))


class TestSentenceChunk:
    def test_keeps_a_short_text_whole(self, indexation: Indexation) -> None:
        assert indexation.sentence_chunk("One. Two.") == ["One. Two."]

    def test_cuts_once_the_word_budget_is_exceeded(self, indexation: Indexation) -> None:
        chunks = indexation.sentence_chunk(many_sentences(10))

        assert len(chunks) > 1
        assert all(len(chunk.split()) <= 50 for chunk in chunks)

    def test_every_sentence_appears_in_a_chunk(self, indexation: Indexation) -> None:
        text = many_sentences(10)

        chunks = indexation.sentence_chunk(text)

        assert all(sentence in " ".join(chunks) for sentence in indexation._get_sentences(text))

    def test_the_chunks_stay_in_order(self, indexation: Indexation) -> None:
        chunks = indexation.sentence_chunk(many_sentences(10))

        assert chunks[0].startswith("Sentence number 0")
        assert chunks[-1].endswith("Sentence number 9 holds a few words here.")

    def test_empty_text_yields_no_chunk(self, indexation: Indexation) -> None:
        assert indexation.sentence_chunk("") == []


class TestSentenceChunkOverlap:
    def test_consecutive_chunks_share_their_boundary_sentence(self, indexation: Indexation) -> None:
        chunks = indexation.sentence_chunk(many_sentences(10))

        for before, after in pairwise(chunks):
            assert indexation._get_sentences(before)[-1] == indexation._get_sentences(after)[0]

    def test_no_overlap_makes_the_chunks_disjoint(self, indexation: Indexation) -> None:
        chunks = indexation.sentence_chunk(many_sentences(10), overlap=0)

        for before, after in pairwise(chunks):
            assert indexation._get_sentences(before)[-1] != indexation._get_sentences(after)[0]

    def test_no_overlap_keeps_the_text_intact(self, indexation: Indexation) -> None:
        text = many_sentences(10)

        assert " ".join(indexation.sentence_chunk(text, overlap=0)) == text

    def test_a_wider_overlap_repeats_more_sentences(self, indexation: Indexation) -> None:
        text = many_sentences(10)

        words = [len(" ".join(indexation.sentence_chunk(text, overlap=n)).split()) for n in (0, 1, 2)]

        assert words[0] < words[1] < words[2]

    def test_a_single_chunk_has_nothing_to_overlap(self, indexation: Indexation) -> None:
        assert indexation.sentence_chunk("One. Two.", overlap=2) == ["One. Two."]

    def test_the_budget_still_holds_with_an_overlap(self, indexation: Indexation) -> None:
        chunks = indexation.sentence_chunk(many_sentences(20), overlap=2)

        assert all(len(chunk.split()) <= 50 for chunk in chunks)


class TestMarkdownChunk:
    def test_one_section_per_heading(self, indexation: Indexation) -> None:
        sections = indexation.markdown_chunk("# A\nbody a\n# B\nbody b")

        assert sections == [{"title": "A", "text": "body a"}, {"title": "B", "text": "body b"}]

    def test_text_before_the_first_heading_is_the_preamble(self, indexation: Indexation) -> None:
        sections = indexation.markdown_chunk("intro\n# A\nbody")

        assert sections[0] == {"title": "(preamble)", "text": "intro"}

    def test_skips_headings_with_an_empty_body(self, indexation: Indexation) -> None:
        assert indexation.markdown_chunk("# A\n# B\nbody b") == [{"title": "B", "text": "body b"}]

    def test_text_without_any_heading_is_one_preamble(self, indexation: Indexation) -> None:
        assert indexation.markdown_chunk("just text") == [{"title": "(preamble)", "text": "just text"}]


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self, indexation: Indexation) -> None:
        assert indexation._cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_scale_does_not_matter(self, indexation: Indexation) -> None:
        assert indexation._cosine_similarity([1.0, 1.0], [5.0, 5.0]) == pytest.approx(1.0)

    def test_perpendicular_vectors_score_zero(self, indexation: Indexation) -> None:
        assert indexation._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_score_minus_one(self, indexation: Indexation) -> None:
        assert indexation._cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_a_zero_vector_scores_zero_instead_of_dividing_by_zero(self, indexation: Indexation) -> None:
        assert indexation._cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_mismatched_dimensions_raise(self, indexation: Indexation) -> None:
        with pytest.raises(ValueError, match="argument 2 is shorter"):
            indexation._cosine_similarity([1.0, 2.0], [1.0])


class TestAdaptiveThreshold:
    def test_is_the_mean_minus_one_standard_deviation(self, indexation: Indexation) -> None:
        assert indexation.adaptive_threshold([0.0, 1.0]) == pytest.approx(0.0)

    def test_needs_at_least_two_scores(self, indexation: Indexation) -> None:
        assert indexation.adaptive_threshold([0.4]) == 0.0
        assert indexation.adaptive_threshold([]) == 0.0


class TestFindBreakpoints:
    def test_returns_the_indices_below_the_threshold(self, indexation: Indexation) -> None:
        assert indexation.find_breakpoints([0.9, 0.2, 0.8], 0.5) == [1]

    def test_the_comparison_is_strict(self, indexation: Indexation) -> None:
        assert indexation.find_breakpoints([0.5], 0.5) == []

    def test_no_score_yields_no_breakpoint(self, indexation: Indexation) -> None:
        assert indexation.find_breakpoints([], 0.5) == []


class TestSplitOnBreakpoints:
    def test_cuts_after_each_breakpoint(self, indexation: Indexation) -> None:
        assert indexation.split_on_breakpoints(["a", "b", "c", "d"], {1}) == ["a b", "c d"]

    def test_without_a_breakpoint_everything_is_one_chunk(self, indexation: Indexation) -> None:
        assert indexation.split_on_breakpoints(["a", "b"], set()) == ["a b"]

    def test_a_breakpoint_on_the_last_sentence_adds_no_empty_chunk(self, indexation: Indexation) -> None:
        assert indexation.split_on_breakpoints(["a", "b"], {1}) == ["a b"]

    def test_no_sentence_yields_no_chunk(self, indexation: Indexation) -> None:
        assert indexation.split_on_breakpoints([], {0}) == []


class TestConsecutiveSimilarities:
    def test_yields_one_score_less_than_the_sentence_count(self, indexation: FakeIndexation) -> None:
        assert len(indexation.consecutive_similarities(["a", "b", "c"])) == 2

    def test_a_single_sentence_has_no_score(self, indexation: FakeIndexation) -> None:
        assert indexation.consecutive_similarities(["only"]) == []

    def test_embeds_every_sentence_in_one_call(self, indexation: FakeIndexation) -> None:
        indexation.consecutive_similarities(["a", "b", "c"])

        assert indexation.batches == [["a", "b", "c"]]


class TestSemanticChunk:
    def test_cuts_where_the_topic_changes(self, indexation: FakeIndexation) -> None:
        chunks = indexation.semantic_chunk(CATS_THEN_ENGINES)

        assert len(chunks) == 2
        assert chunks[0].startswith("The cat sleeps")
        assert chunks[1].startswith("The engine burns")

    def test_a_single_topic_stays_whole(self, indexation: FakeIndexation) -> None:
        assert indexation.semantic_chunk("The cat sleeps. It purrs. Cats rest.") == [
            "The cat sleeps. It purrs. Cats rest."
        ]

    def test_a_single_sentence_needs_no_embedding(self, indexation: FakeIndexation) -> None:
        assert indexation.semantic_chunk("Only one.") == ["Only one."]
        assert indexation.batches == []

    def test_empty_text_yields_no_chunk(self, indexation: FakeIndexation) -> None:
        assert indexation.semantic_chunk("") == []


class TestCollectFile:
    def test_reads_a_text_file(self, indexation: Indexation, tmp_path: Path) -> None:
        source = tmp_path / "a.txt"
        source.write_text("hello", encoding="utf-8")

        assert indexation._collect_file(str(source)) == "hello"

    def test_reads_a_markdown_file(self, indexation: Indexation, tmp_path: Path) -> None:
        source = tmp_path / "a.md"
        source.write_text("# Title\nbody", encoding="utf-8")

        assert indexation._collect_file(str(source)) == "# Title\nbody"

    def test_extracts_the_text_of_an_html_file(self, indexation: Indexation, tmp_path: Path) -> None:
        source = tmp_path / "a.html"
        source.write_text("<p>hello</p><script>ignored()</script>", encoding="utf-8")

        assert indexation._collect_file(str(source)) == "hello"

    def test_an_empty_file_is_not_an_error(self, indexation: Indexation, tmp_path: Path) -> None:
        source = tmp_path / "a.txt"
        source.write_text("", encoding="utf-8")

        assert indexation._collect_file(str(source)) == ""

    def test_a_missing_file_raises(self, indexation: Indexation, tmp_path: Path) -> None:
        with pytest.raises(SourceNotFoundError, match="no such file"):
            indexation._collect_file(str(tmp_path / "absent.md"))

    def test_an_unknown_extension_raises(self, indexation: Indexation, tmp_path: Path) -> None:
        source = tmp_path / "a.docx"
        source.write_text("data", encoding="utf-8")

        with pytest.raises(UnsupportedFormatError, match=r"\.docx"):
            indexation._collect_file(str(source))

    def test_no_extension_at_all_raises(self, indexation: Indexation, tmp_path: Path) -> None:
        source = tmp_path / "plain"
        source.write_text("data", encoding="utf-8")

        with pytest.raises(UnsupportedFormatError, match="no extension"):
            indexation._collect_file(str(source))

    def test_text_that_is_not_utf8_raises(self, indexation: Indexation, tmp_path: Path) -> None:
        source = tmp_path / "a.txt"
        source.write_bytes(b"\xff\xfe\x00bad")

        with pytest.raises(SourceFetchError, match="could not read"):
            indexation._collect_file(str(source))

    def test_a_file_that_is_not_a_pdf_raises(self, indexation: Indexation, tmp_path: Path) -> None:
        source = tmp_path / "a.pdf"
        source.write_text("not a pdf at all", encoding="utf-8")

        with pytest.raises(SourceFetchError, match="could not read"):
            indexation._collect_file(str(source))

    def test_an_unreachable_url_raises(self, indexation: Indexation) -> None:
        with pytest.raises(SourceFetchError, match="could not fetch"):
            indexation._collect_file("http://localhost:9/nothing")

    def test_a_malformed_url_raises(self, indexation: Indexation) -> None:
        with pytest.raises(SourceFetchError, match="could not fetch"):
            indexation._collect_file("http://")

    def test_every_failure_is_a_source_error(self, indexation: Indexation, tmp_path: Path) -> None:
        with pytest.raises(SourceError):
            indexation._collect_file(str(tmp_path / "absent.md"))


class TestToChunks:
    def test_auto_uses_markdown_for_a_md_source(self, indexation: FakeIndexation, tmp_path: Path) -> None:
        source = tmp_path / "a.md"
        source.write_text("# Engines\nThe engine burns fuel.\n# Cats\nThe cat sleeps.", encoding="utf-8")

        chunks = indexation.to_chunks(str(source))

        assert chunks == ["Engines\n\nThe engine burns fuel.", "Cats\n\nThe cat sleeps."]

    def test_auto_uses_semantic_for_anything_else(self, indexation: FakeIndexation, tmp_path: Path) -> None:
        source = tmp_path / "a.txt"
        source.write_text(CATS_THEN_ENGINES, encoding="utf-8")

        assert len(indexation.to_chunks(str(source))) == 2

    def test_the_markdown_title_prefixes_every_chunk_of_its_section(
        self, indexation: FakeIndexation, tmp_path: Path
    ) -> None:
        source = tmp_path / "a.md"
        source.write_text(f"# Mixed\n{CATS_THEN_ENGINES}", encoding="utf-8")

        chunks = indexation.to_chunks(str(source))

        assert len(chunks) == 2
        assert all(chunk.startswith("Mixed\n\n") for chunk in chunks)

    def test_a_forced_strategy_wins_over_the_extension(self, indexation: FakeIndexation, tmp_path: Path) -> None:
        source = tmp_path / "a.md"
        source.write_text("# Title\nOne. Two.", encoding="utf-8")

        chunks = indexation.to_chunks(str(source), strategy=ChunkStrategy.SENTENCE)

        assert chunks == ["# Title One. Two."]

    def test_the_sentence_strategy_makes_no_embedding_call(self, indexation: FakeIndexation, tmp_path: Path) -> None:
        source = tmp_path / "a.txt"
        source.write_text(CATS_THEN_ENGINES, encoding="utf-8")

        indexation.to_chunks(str(source), strategy=ChunkStrategy.SENTENCE)

        assert indexation.batches == []

    def test_an_empty_source_yields_no_chunk(self, indexation: FakeIndexation, tmp_path: Path) -> None:
        source = tmp_path / "a.txt"
        source.write_text("   ", encoding="utf-8")

        assert indexation.to_chunks(str(source)) == []

    def test_a_missing_source_raises(self, indexation: FakeIndexation, tmp_path: Path) -> None:
        with pytest.raises(SourceNotFoundError):
            indexation.to_chunks(str(tmp_path / "absent.md"))


class TestModel:
    def test_reports_the_model_in_use(self, indexation: Indexation) -> None:
        assert indexation.model == "bge-m3"

    def test_the_model_can_be_chosen(self) -> None:
        assert Indexation(FakeEmbedder(), "another-model").model == "another-model"

    def test_the_chosen_model_is_the_one_asked_of_the_embedder(self) -> None:
        embedder = FakeEmbedder()

        Indexation(embedder, "another-model").vectorize_batch(["a"])

        assert embedder.models == ["another-model"]

    def test_the_default_model_is_the_one_asked_of_the_embedder(self) -> None:
        embedder = FakeEmbedder()

        Indexation(embedder).vectorize_batch(["a"])

        assert embedder.models == ["bge-m3"]


class TestVectorize:
    def test_returns_one_vector_per_text(self, indexation: FakeIndexation) -> None:
        assert len(indexation.vectorize_batch(["a", "b", "c"])) == 3

    def test_a_single_text_gets_a_single_vector(self, indexation: FakeIndexation) -> None:
        assert indexation.vectorize("The cat sleeps.") == [1.0, 0.1]

    def test_an_embedder_returning_nothing_raises(self) -> None:
        with pytest.raises(EmbeddingError, match="returned 0 vectors for 2 texts"):
            Indexation(SilentEmbedder()).vectorize_batch(["a", "b"])

    def test_the_error_names_the_model_that_failed(self) -> None:
        with pytest.raises(EmbeddingError, match="pas-un-modele"):
            Indexation(SilentEmbedder(), "pas-un-modele").vectorize_batch(["a"])

    def test_the_error_is_a_rag_error(self) -> None:
        with pytest.raises(RagError):
            Indexation(SilentEmbedder()).vectorize_batch(["a"])

    def test_a_silent_embedder_fails_the_whole_chunking(self) -> None:
        with pytest.raises(EmbeddingError):
            Indexation(SilentEmbedder()).semantic_chunk("The cat sleeps. The engine burns fuel.")
