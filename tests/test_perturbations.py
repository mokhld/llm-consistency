"""Tests for llm_consistency.perturbations module."""

from __future__ import annotations

import pytest

from llm_consistency.perturbations import (
    BasePerturbation,
    FormatChangePerturbation,
    OptionReorderPerturbation,
    SeparatorChangePerturbation,
    _reset_registry,
    get,
    list_registered,
    register,
)
from llm_consistency.types import (
    MCOption,
    MCQuestion,
    PerturbationType,
    PerturbedVariant,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry() -> None:  # noqa: PT004
    """Reset the perturbation registry before and after each test."""
    _reset_registry()
    yield  # type: ignore[misc]
    _reset_registry()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OPTIONS = (
    MCOption(label="A", text="Paris", is_correct=True),
    MCOption(label="B", text="London", is_correct=False),
)

_QUESTION = MCQuestion(id="q1", stem="Capital of France?", options=_OPTIONS)


class _StubPerturbation(BasePerturbation):
    """Minimal concrete subclass for testing."""

    @property
    def perturbation_type(self) -> PerturbationType:
        return PerturbationType.OPTION_REORDER

    def generate_variants(
        self,
        question: MCQuestion,
        *,
        seed: int = 0,
        n: int | None = None,
    ) -> tuple[PerturbedVariant, ...]:
        return (
            PerturbedVariant(
                original_question_id=question.id,
                perturbation_type=self.perturbation_type,
                seed=seed,
                variant_index=0,
                stem=question.stem,
                options=question.options,
            ),
        )


# ---------------------------------------------------------------------------
# BasePerturbation ABC tests
# ---------------------------------------------------------------------------


class TestBasePerturbationABC:
    """Tests for BasePerturbation abstract base class enforcement."""

    def test_cannot_instantiate_directly(self) -> None:
        """BasePerturbation cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePerturbation()  # type: ignore[abstract]

    def test_subclass_missing_generate_variants_raises(self) -> None:
        """A subclass that does not implement generate_variants cannot be instantiated."""

        class _Incomplete(BasePerturbation):
            @property
            def perturbation_type(self) -> PerturbationType:
                return PerturbationType.OPTION_REORDER

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]

    def test_subclass_missing_perturbation_type_raises(self) -> None:
        """A subclass that does not implement perturbation_type cannot be instantiated."""

        class _Incomplete(BasePerturbation):
            def generate_variants(
                self,
                question: MCQuestion,
                *,
                seed: int = 0,
                n: int | None = None,
            ) -> tuple[PerturbedVariant, ...]:
                return ()

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self) -> None:
        """A fully-implementing subclass can be instantiated."""
        instance = _StubPerturbation()
        assert isinstance(instance, BasePerturbation)

    def test_concrete_subclass_returns_tuple_of_perturbed_variants(self) -> None:
        """generate_variants returns a tuple[PerturbedVariant, ...]."""
        instance = _StubPerturbation()
        result = instance.generate_variants(_QUESTION, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], PerturbedVariant)
        assert result[0].original_question_id == "q1"
        assert result[0].perturbation_type == PerturbationType.OPTION_REORDER
        assert result[0].seed == 42
        assert result[0].variant_index == 0


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for the perturbation plugin registry."""

    def test_register_succeeds_with_base_perturbation_instance(self) -> None:
        """register() accepts a BasePerturbation instance."""
        stub = _StubPerturbation()
        register("stub", stub)
        assert "stub" in list_registered()

    def test_register_raises_type_error_for_non_perturbation(self) -> None:
        """register() raises TypeError for non-BasePerturbation objects."""
        with pytest.raises(TypeError, match="BasePerturbation"):
            register("bad", "not a perturbation")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="BasePerturbation"):
            register("bad", 42)  # type: ignore[arg-type]

    def test_register_raises_value_error_on_duplicate(self) -> None:
        """register() raises ValueError when name already registered (no force)."""
        stub = _StubPerturbation()
        register("dup", stub)
        with pytest.raises(ValueError, match="already registered"):
            register("dup", stub)

    def test_register_force_allows_overwrite(self) -> None:
        """register() with force=True allows overwriting an existing entry."""
        stub1 = _StubPerturbation()
        stub2 = _StubPerturbation()
        register("over", stub1)
        register("over", stub2, force=True)
        assert get("over") is stub2

    def test_get_returns_registered_instance(self) -> None:
        """get() returns the registered perturbation instance."""
        stub = _StubPerturbation()
        register("my_stub", stub)
        assert get("my_stub") is stub

    def test_get_raises_key_error_for_unknown_name(self) -> None:
        """get() raises KeyError with helpful message for unknown names."""
        stub = _StubPerturbation()
        register("known", stub)
        with pytest.raises(KeyError, match="unknown_name"):
            get("unknown_name")

    def test_get_error_message_includes_registered_names(self) -> None:
        """get() KeyError message includes list_registered() output."""
        stub = _StubPerturbation()
        register("alpha", stub)
        register("beta", stub, force=True)
        with pytest.raises(KeyError, match="alpha") as exc_info:
            get("missing")
        # The error message should mention available names
        assert "beta" in str(exc_info.value)

    def test_list_registered_returns_sorted_list(self) -> None:
        """list_registered() returns names in sorted order (includes builtins)."""
        stub = _StubPerturbation()
        register("zebra", stub)
        register("apple", stub)
        register("mango", stub)
        result = list_registered()
        assert result == ["apple", "mango", "option_reorder", "zebra"]

    def test_list_registered_contains_builtins_after_reset(self) -> None:
        """list_registered() contains built-in perturbations after _reset_registry()."""
        stub = _StubPerturbation()
        register("temp", stub)
        assert "temp" in list_registered()
        _reset_registry()
        # Built-in option_reorder is always present after reset
        assert "option_reorder" in list_registered()
        assert "temp" not in list_registered()

    def test_reset_registry_clears_custom_and_re_registers_builtins(self) -> None:
        """_reset_registry() clears custom entries and re-registers built-ins."""
        stub = _StubPerturbation()
        register("custom1", stub)
        register("custom2", stub)
        assert "custom1" in list_registered()
        assert "custom2" in list_registered()
        _reset_registry()
        assert "custom1" not in list_registered()
        assert "custom2" not in list_registered()
        # Built-in option_reorder is re-registered
        assert "option_reorder" in list_registered()


# ---------------------------------------------------------------------------
# Fixtures for OptionReorderPerturbation tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_question() -> MCQuestion:
    """A 4-option question: A=Paris (correct), B=London, C=Berlin, D=Madrid."""
    return MCQuestion(
        id="q-capitals",
        stem="What is the capital of France?",
        options=(
            MCOption(label="A", text="Paris", is_correct=True),
            MCOption(label="B", text="London", is_correct=False),
            MCOption(label="C", text="Berlin", is_correct=False),
            MCOption(label="D", text="Madrid", is_correct=False),
        ),
    )


# ---------------------------------------------------------------------------
# OptionReorderPerturbation tests
# ---------------------------------------------------------------------------


class TestOptionReorderCore:
    """Core behavior tests for OptionReorderPerturbation."""

    def test_option_reorder_returns_all_non_identity_permutations(
        self, sample_question: MCQuestion
    ) -> None:
        """With 4 options and n=None, expect exactly 23 variants (4! - 1)."""
        pert = OptionReorderPerturbation()
        variants = pert.generate_variants(sample_question, seed=0)
        assert len(variants) == 23
        for v in variants:
            assert v.options is not None
            assert len(v.options) == 4

    def test_option_reorder_excludes_identity(
        self, sample_question: MCQuestion
    ) -> None:
        """No variant has options in the original order."""
        pert = OptionReorderPerturbation()
        variants = pert.generate_variants(sample_question, seed=0)
        original_texts = tuple(o.text for o in sample_question.options)
        for v in variants:
            assert v.options is not None
            variant_texts = tuple(o.text for o in v.options)
            assert variant_texts != original_texts

    def test_option_reorder_label_reassignment(
        self, sample_question: MCQuestion
    ) -> None:
        """Labels always match position; is_correct follows text content."""
        pert = OptionReorderPerturbation()
        variants = pert.generate_variants(sample_question, seed=0)
        for v in variants:
            assert v.options is not None
            labels = [o.label for o in v.options]
            assert labels == ["A", "B", "C", "D"]
            # The option with text "Paris" must always be correct
            for opt in v.options:
                if opt.text == "Paris":
                    assert opt.is_correct is True
                else:
                    assert opt.is_correct is False

    def test_option_reorder_correct_answer_label_changes(
        self, sample_question: MCQuestion
    ) -> None:
        """The correct answer's label appears at different positions across variants."""
        pert = OptionReorderPerturbation()
        variants = pert.generate_variants(sample_question, seed=0)
        correct_labels = set()
        for v in variants:
            assert v.options is not None
            for opt in v.options:
                if opt.is_correct:
                    correct_labels.add(opt.label)
        # The correct answer should NOT always be "A"
        assert len(correct_labels) > 1


class TestOptionReorderNSampling:
    """Tests for N-sampling behavior."""

    def test_option_reorder_n_sampling(
        self, sample_question: MCQuestion
    ) -> None:
        """With n=5, expect exactly 5 variants."""
        pert = OptionReorderPerturbation()
        variants = pert.generate_variants(sample_question, seed=42, n=5)
        assert len(variants) == 5

    def test_option_reorder_n_larger_than_available(
        self, sample_question: MCQuestion
    ) -> None:
        """With n=30 (exceeds 23 available), expect all 23 variants."""
        pert = OptionReorderPerturbation()
        variants = pert.generate_variants(sample_question, seed=42, n=30)
        assert len(variants) == 23


class TestOptionReorderSeedReproducibility:
    """Seed reproducibility tests (PERT-08)."""

    def test_option_reorder_seed_reproducibility(
        self, sample_question: MCQuestion
    ) -> None:
        """Same seed + same input = identical output."""
        pert = OptionReorderPerturbation()
        variants_a = pert.generate_variants(sample_question, seed=42, n=5)
        variants_b = pert.generate_variants(sample_question, seed=42, n=5)
        assert variants_a == variants_b

    def test_option_reorder_different_seeds_differ(
        self, sample_question: MCQuestion
    ) -> None:
        """Different seeds produce different output."""
        pert = OptionReorderPerturbation()
        variants_a = pert.generate_variants(sample_question, seed=42, n=5)
        variants_b = pert.generate_variants(sample_question, seed=99, n=5)
        assert variants_a != variants_b


class TestOptionReorderMetadata:
    """Tests for perturbation type and variant metadata."""

    def test_option_reorder_perturbation_type(self) -> None:
        """OptionReorderPerturbation().perturbation_type is OPTION_REORDER."""
        pert = OptionReorderPerturbation()
        assert pert.perturbation_type == PerturbationType.OPTION_REORDER

    def test_option_reorder_variant_metadata(
        self, sample_question: MCQuestion
    ) -> None:
        """Each variant has correct provenance metadata."""
        pert = OptionReorderPerturbation()
        seed = 42
        variants = pert.generate_variants(sample_question, seed=seed, n=3)
        for i, v in enumerate(variants):
            assert v.original_question_id == sample_question.id
            assert v.perturbation_type == PerturbationType.OPTION_REORDER
            assert v.seed == seed
            assert v.variant_index == i
            assert v.stem == sample_question.stem


class TestOptionReorderEdgeCases:
    """Edge case tests."""

    def test_option_reorder_two_options(self) -> None:
        """With 2 options, expect exactly 1 variant (2! - 1 = 1)."""
        question = MCQuestion(
            id="q-binary",
            stem="Is the sky blue?",
            options=(
                MCOption(label="A", text="Yes", is_correct=True),
                MCOption(label="B", text="No", is_correct=False),
            ),
        )
        pert = OptionReorderPerturbation()
        variants = pert.generate_variants(question, seed=0)
        assert len(variants) == 1


# ---------------------------------------------------------------------------
# FormatChangePerturbation tests
# ---------------------------------------------------------------------------


class TestFormatChangeCore:
    """Core behavior tests for FormatChangePerturbation."""

    def test_format_change_returns_variants(
        self, sample_question: MCQuestion
    ) -> None:
        """With n=None, returns at least 6 variants (one per template)."""
        pert = FormatChangePerturbation()
        variants = pert.generate_variants(sample_question, seed=0)
        assert len(variants) >= 6
        for v in variants:
            # Stem contains original stem content
            assert "capital of France" in v.stem.lower() or "Capital of France" in v.stem
            # All option texts present in the rendered stem
            for opt in sample_question.options:
                assert opt.text in v.stem

    def test_format_change_templates_differ(
        self, sample_question: MCQuestion
    ) -> None:
        """All returned variant stems must be distinct from each other."""
        pert = FormatChangePerturbation()
        variants = pert.generate_variants(sample_question, seed=0)
        stems = [v.stem for v in variants]
        assert len(stems) == len(set(stems)), "All templates should produce distinct output"

    def test_format_change_preserves_option_texts(
        self, sample_question: MCQuestion
    ) -> None:
        """Every variant contains all original option texts."""
        pert = FormatChangePerturbation()
        variants = pert.generate_variants(sample_question, seed=0)
        option_texts = [opt.text for opt in sample_question.options]
        for v in variants:
            for text in option_texts:
                assert text in v.stem, f"Option text '{text}' missing from variant stem"


class TestFormatChangeNSampling:
    """N-sampling tests for FormatChangePerturbation."""

    def test_format_change_n_sampling(
        self, sample_question: MCQuestion
    ) -> None:
        """With n=3, returns exactly 3 variants."""
        pert = FormatChangePerturbation()
        variants = pert.generate_variants(sample_question, seed=0, n=3)
        assert len(variants) == 3


class TestFormatChangeSeedReproducibility:
    """Seed reproducibility tests for FormatChangePerturbation."""

    def test_format_change_seed_reproducibility(
        self, sample_question: MCQuestion
    ) -> None:
        """Same seed + same input = identical output."""
        pert = FormatChangePerturbation()
        variants_a = pert.generate_variants(sample_question, seed=42, n=3)
        variants_b = pert.generate_variants(sample_question, seed=42, n=3)
        assert variants_a == variants_b

    def test_format_change_different_seeds_differ(
        self, sample_question: MCQuestion
    ) -> None:
        """Different seeds produce different output (for n < total templates)."""
        pert = FormatChangePerturbation()
        variants_a = pert.generate_variants(sample_question, seed=42, n=3)
        variants_b = pert.generate_variants(sample_question, seed=99, n=3)
        assert variants_a != variants_b


class TestFormatChangeMetadata:
    """Tests for FormatChangePerturbation type and variant metadata."""

    def test_format_change_perturbation_type(self) -> None:
        """FormatChangePerturbation().perturbation_type is FORMAT_CHANGE."""
        pert = FormatChangePerturbation()
        assert pert.perturbation_type == PerturbationType.FORMAT_CHANGE

    def test_format_change_variant_metadata(
        self, sample_question: MCQuestion
    ) -> None:
        """Each variant has correct metadata. Options is None."""
        pert = FormatChangePerturbation()
        seed = 42
        variants = pert.generate_variants(sample_question, seed=seed)
        for i, v in enumerate(variants):
            assert v.original_question_id == sample_question.id
            assert v.perturbation_type == PerturbationType.FORMAT_CHANGE
            assert v.seed == seed
            assert v.variant_index == i
            assert v.options is None


# ---------------------------------------------------------------------------
# SeparatorChangePerturbation tests
# ---------------------------------------------------------------------------


class TestSeparatorChangeCore:
    """Core behavior tests for SeparatorChangePerturbation."""

    def test_separator_change_returns_variants(
        self, sample_question: MCQuestion
    ) -> None:
        """With n=None, returns at least 8 variants."""
        pert = SeparatorChangePerturbation()
        variants = pert.generate_variants(sample_question, seed=0)
        assert len(variants) >= 8
        for v in variants:
            # Stem contains original stem content
            assert "capital of France" in v.stem.lower() or "Capital of France" in v.stem
            # All option texts present
            for opt in sample_question.options:
                assert opt.text in v.stem

    def test_separator_change_separators_differ(
        self, sample_question: MCQuestion
    ) -> None:
        """Returned variants use different delimiters (stems are distinct)."""
        pert = SeparatorChangePerturbation()
        variants = pert.generate_variants(sample_question, seed=0)
        stems = [v.stem for v in variants]
        assert len(stems) == len(set(stems)), "All separators should produce distinct output"

    def test_separator_change_preserves_option_texts(
        self, sample_question: MCQuestion
    ) -> None:
        """All original option texts appear in each variant's stem."""
        pert = SeparatorChangePerturbation()
        variants = pert.generate_variants(sample_question, seed=0)
        option_texts = [opt.text for opt in sample_question.options]
        for v in variants:
            for text in option_texts:
                assert text in v.stem, f"Option text '{text}' missing from variant stem"


class TestSeparatorChangeNSampling:
    """N-sampling tests for SeparatorChangePerturbation."""

    def test_separator_change_n_sampling(
        self, sample_question: MCQuestion
    ) -> None:
        """With n=3, returns exactly 3 variants."""
        pert = SeparatorChangePerturbation()
        variants = pert.generate_variants(sample_question, seed=0, n=3)
        assert len(variants) == 3


class TestSeparatorChangeSeedReproducibility:
    """Seed reproducibility tests for SeparatorChangePerturbation."""

    def test_separator_change_seed_reproducibility(
        self, sample_question: MCQuestion
    ) -> None:
        """Same seed + same input = identical output."""
        pert = SeparatorChangePerturbation()
        variants_a = pert.generate_variants(sample_question, seed=42, n=3)
        variants_b = pert.generate_variants(sample_question, seed=42, n=3)
        assert variants_a == variants_b

    def test_separator_change_different_seeds_differ(
        self, sample_question: MCQuestion
    ) -> None:
        """Different seeds produce different output."""
        pert = SeparatorChangePerturbation()
        variants_a = pert.generate_variants(sample_question, seed=42, n=3)
        variants_b = pert.generate_variants(sample_question, seed=99, n=3)
        assert variants_a != variants_b


class TestSeparatorChangeMetadata:
    """Tests for SeparatorChangePerturbation type and variant metadata."""

    def test_separator_change_perturbation_type(self) -> None:
        """SeparatorChangePerturbation().perturbation_type is SEPARATOR_CHANGE."""
        pert = SeparatorChangePerturbation()
        assert pert.perturbation_type == PerturbationType.SEPARATOR_CHANGE

    def test_separator_change_variant_metadata(
        self, sample_question: MCQuestion
    ) -> None:
        """Each variant has correct metadata. Options is None."""
        pert = SeparatorChangePerturbation()
        seed = 42
        variants = pert.generate_variants(sample_question, seed=seed)
        for i, v in enumerate(variants):
            assert v.original_question_id == sample_question.id
            assert v.perturbation_type == PerturbationType.SEPARATOR_CHANGE
            assert v.seed == seed
            assert v.variant_index == i
            assert v.options is None
