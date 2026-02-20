"""Tests for llm_consistency.perturbations module."""

from __future__ import annotations

import pytest

from llm_consistency.perturbations import (
    BasePerturbation,
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
        """list_registered() returns names in sorted order."""
        stub = _StubPerturbation()
        register("zebra", stub)
        register("apple", stub)
        register("mango", stub)
        result = list_registered()
        assert result == ["apple", "mango", "zebra"]

    def test_list_registered_empty_after_reset(self) -> None:
        """list_registered() returns empty list after _reset_registry()."""
        stub = _StubPerturbation()
        register("temp", stub)
        assert len(list_registered()) > 0
        _reset_registry()
        # No built-ins registered yet (plans 02-03), so empty
        assert list_registered() == []

    def test_reset_registry_clears_and_re_registers_builtins(self) -> None:
        """_reset_registry() clears all entries and re-registers built-ins.

        In this plan, built-ins are empty (concrete perturbations in plans 02-03),
        so after reset the registry should be empty.
        """
        stub = _StubPerturbation()
        register("custom1", stub)
        register("custom2", stub)
        assert len(list_registered()) == 2
        _reset_registry()
        assert list_registered() == []
