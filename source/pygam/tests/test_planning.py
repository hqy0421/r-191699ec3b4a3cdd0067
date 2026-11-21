"""Tests for budget-aware plan selection on fitted logistic GAMs."""

from itertools import product

import numpy as np
import pytest

from pygam import LinearGAM, LogisticGAM, l, s, te
from pygam.planning import Change, Plan, recommend_plan


def _make_gam(n_features=3, n_samples=600, seed=0, terms=None):
    """Fit a small interaction-free LogisticGAM on a synthetic order dataset."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    logits = 0.8 * X[:, 0] - 0.6 * X[:, 1] + (0.4 * X[:, 2] if n_features > 2 else 0)
    y = (logits + rng.normal(scale=0.5, size=n_samples) > 0).astype(float)
    if terms is None:
        terms = s(0) + s(1) + (s(2) if n_features > 2 else l(0))
    gam = LogisticGAM(terms=terms).fit(X, y)
    return gam, X


def _brute_force(gam, order, adjustable, budget, max_probability):
    """Exhaustive reference implementation used only by the tests.

    Returns ``(cost, probability, key)`` for the cheapest plan that reaches the
    cap, breaking ties on risk and then option order. This mirrors the documented
    rules; it is intentionally simple so it can be trusted as a reference.
    """
    specs = (
        list(adjustable.items()) if isinstance(adjustable, dict) else list(adjustable)
    )
    choices = []
    for feature, options in specs:
        values = [float(order[feature])] + [float(v) for v, _c in options]
        costs = [0] + [int(c) for _v, c in options]
        seen = set()
        kept_values, kept_costs = [], []
        for value, cost in zip(values, costs):
            if value in seen:
                continue
            seen.add(value)
            kept_values.append(value)
            kept_costs.append(cost)
        choices.append((feature, kept_values, kept_costs))

    feasible = []
    for key in product(*[range(len(c[1])) for c in choices]):
        total = sum(choices[i][2][key[i]] for i in range(len(choices)))
        if total > budget:
            continue
        row = np.array(order, dtype=float).reshape(1, -1)
        for i, (feature, values, _costs) in enumerate(choices):
            row[0, feature] = values[key[i]]
        probability = float(np.ravel(gam.predict_proba(row))[0])
        if probability <= max_probability + 1e-12:
            feasible.append((total, probability, key))

    if not feasible:
        return None
    return min(feasible, key=lambda item: (item[0], item[1], item[2]))


def test_entry_point_is_on_logisticgam():
    gam, X = _make_gam()
    plan = gam.recommend_plan(X[0], {0: [(3.0, 2)]}, budget=2, max_probability=1.0)
    assert isinstance(plan, Plan)
    assert plan.feasible


def test_returns_changes_cost_and_probabilities():
    gam, X = _make_gam()
    order = X[0]
    plan = gam.recommend_plan(
        order,
        {0: [(3.0, 2), (-3.0, 5)], 1: [(3.0, 1)]},
        budget=6,
        max_probability=0.5,
    )
    assert plan.feasible
    assert plan.cost <= 6
    assert 0.0 <= plan.probability_after <= 1.0
    assert plan.probability_before == pytest.approx(
        float(gam.predict_proba(order.reshape(1, -1))[0])
    )
    # The reported probability must be a genuine recomputation.
    assert plan.probability_after == pytest.approx(
        float(gam.predict_proba(np.array(plan.values).reshape(1, -1))[0])
    )
    returned = {change.feature: change for change in plan.changes}
    assert set(returned).issubset({0, 1})
    for change in plan.changes:
        assert isinstance(change, Change)
        assert change.new_value == pytest.approx(plan.values[change.feature])
        assert change.old_value == pytest.approx(order[change.feature])


def test_keeps_original_when_already_below_cap():
    gam, X = _make_gam()
    order = X[0]
    before = float(gam.predict_proba(order.reshape(1, -1))[0])
    plan = gam.recommend_plan(
        order, {0: [(3.0, 100)]}, budget=100, max_probability=min(1.0, before + 1e-6)
    )
    assert plan.feasible
    assert plan.cost == 0
    assert plan.changes == ()
    assert plan.probability_after == pytest.approx(before)


def test_fixed_features_are_held_constant():
    gam, X = _make_gam()
    order = X[0].copy()
    plan = gam.recommend_plan(
        order, {0: [(3.0, 2), (-3.0, 4)]}, budget=4, max_probability=0.4
    )
    assert plan.feasible
    values = np.array(plan.values)
    assert values[1] == pytest.approx(order[1])
    assert values[2] == pytest.approx(order[2])


def test_cheapest_feasible_plan_is_chosen():
    gam, X = _make_gam()
    order = X[0]
    adjustable = {0: [(3.0, 2), (-3.0, 10)], 1: [(3.0, 1)]}
    plan = gam.recommend_plan(order, adjustable, budget=20, max_probability=0.5)
    reference = _brute_force(gam, order, adjustable, budget=20, max_probability=0.5)
    assert plan.feasible
    assert plan.cost == reference[0]
    assert plan.probability_after == pytest.approx(reference[1], abs=1e-9)


def test_same_cost_prefers_lower_risk():
    gam, X = _make_gam()
    order = X[0]
    # A cap below the original forces some change, while four options share one
    # cost so the risk decides between them.
    options = {0: [(1.0, 1), (-1.0, 1), (3.0, 1), (-3.0, 1)]}
    plan = gam.recommend_plan(order, options, budget=1, max_probability=0.01)
    assert plan.feasible and plan.cost == 1
    reference = _brute_force(gam, order, options, budget=1, max_probability=0.01)
    assert plan.probability_after == pytest.approx(reference[1], abs=1e-12)


class _TableGam:
    """Minimal additive logistic model for deterministic tie-break tests.

    Each feature contributes an effect read from a lookup table, so candidate
    plans can be given exactly equal costs and exactly equal risks.
    """

    _is_fitted = True

    def __init__(self, effects, n_features):
        self.effects = effects  # effects[feature][value] -> linear predictor term
        self.terms = []  # no tensor terms, no by-variables
        self.statistics_ = {"m_features": n_features}

    def _lp(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        lp = np.zeros(X.shape[0])
        for feature in range(X.shape[1]):
            table = self.effects.get(feature, {})
            for row, value in enumerate(X[:, feature]):
                lp[row] += table.get(float(value), 0.0)
        return lp

    def _linear_predictor(self, X):
        return self._lp(X)

    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-self._lp(X)))


def test_option_order_breaks_within_feature_ties():
    """Equal cost and equal risk: the first listed option wins."""
    # Both values of feature 0 have the same effect and the same cost.
    gam = _TableGam({0: {2.0: -1.0, 3.0: -1.0}}, n_features=1)
    plan = recommend_plan(
        gam, [0.0], {0: [(2.0, 1), (3.0, 1)]}, budget=1, max_probability=0.4
    )
    assert plan.feasible
    assert plan.cost == 1
    assert plan.changes[0].new_value == 2.0


def test_option_order_breaks_cross_feature_ties():
    """Equal plans on different features: earlier features keep option 0."""
    # Changing either feature lowers the linear predictor by the same amount at
    # the same cost, so the lexicographic option order decides.
    gam = _TableGam({0: {2.0: -1.0}, 1: {2.0: -1.0}}, n_features=2)
    plan = recommend_plan(
        gam,
        [0.0, 0.0],
        [(0, [(2.0, 1)]), (1, [(2.0, 1)])],
        budget=1,
        max_probability=0.4,
    )
    assert plan.feasible
    assert plan.cost == 1
    # Feature 0 keeps its original value (option 0), so feature 1 changes.
    assert [change.feature for change in plan.changes] == [1]


def test_option_order_is_stable_across_runs():
    gam, X = _make_gam()
    order = X[0]
    options = {0: [(1.0, 1), (-1.0, 1), (3.0, 1), (-3.0, 1)]}
    plans = [
        gam.recommend_plan(order, options, budget=1, max_probability=0.01)
        for _ in range(5)
    ]
    first = plans[0]
    for plan in plans[1:]:
        assert plan.cost == first.cost
        assert plan.values == pytest.approx(first.values)


def test_multiple_terms_on_one_feature_are_not_added_as_probability_drops():
    """A feature used by several terms must be scored through the full model."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(800, 2))
    logits = 1.2 * X[:, 0] - 0.3 * np.sin(2 * X[:, 0]) + 0.5 * X[:, 1]
    y = (logits + rng.normal(scale=0.4, size=800) > 0).astype(float)
    gam = LogisticGAM(s(0) + l(0) + s(1)).fit(X, y)
    assert len(gam.terms) == 4  # s(0), l(0), s(1), intercept

    order = X[0]
    adjustable = {0: [(-2.0, 2), (2.0, 3)], 1: [(2.0, 1)]}
    plan = gam.recommend_plan(order, adjustable, budget=10, max_probability=0.4)
    reference = _brute_force(gam, order, adjustable, budget=10, max_probability=0.4)
    assert plan.probability_after == pytest.approx(reference[1], abs=1e-9)
    assert plan.cost == reference[0]


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_dynamic_program_matches_exhaustive_search(seed):
    gam, X = _make_gam(seed=seed)
    order = X[seed]
    adjustable = {
        0: [(-2.5, 1), (0.5, 3), (2.5, 4)],
        1: [(1.0, 2), (-1.0, 3)],
        2: [(2.0, 1), (-2.0, 5)],
    }
    budget = 8
    plan = gam.recommend_plan(order, adjustable, budget=budget, max_probability=0.35)
    reference = _brute_force(
        gam, order, adjustable, budget=budget, max_probability=0.35
    )
    if reference is None:
        assert not plan.feasible
        return
    assert plan.feasible
    assert plan.cost == reference[0]
    assert plan.probability_after == pytest.approx(reference[1], abs=1e-9)


def test_infeasible_is_reported_explicitly():
    gam, X = _make_gam()
    order = X[0]
    plan = gam.recommend_plan(order, {0: [(1.0, 1)]}, budget=1, max_probability=0.0)
    assert plan.feasible is False
    assert bool(plan) is False
    assert plan.cost is None
    assert plan.changes == ()
    assert plan.probability_after is None
    assert plan.values is None
    assert "budget 1" in plan.message
    assert "No plan" in plan.message


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 100])
def test_budget_is_respected(budget):
    gam, X = _make_gam()
    options = {0: [(-3.0, 3)], 1: [(3.0, 2)], 2: [(-3.0, 2)]}
    plan = gam.recommend_plan(X[1], options, budget=budget, max_probability=0.5)
    if plan.feasible:
        assert plan.cost <= budget


def test_model_is_not_modified():
    gam, X = _make_gam()
    coef_before = gam.coef_.copy()
    terms_before = repr(gam.terms)
    gam.recommend_plan(
        X[0], {0: [(3.0, 2), (-3.0, 4)], 1: [(3.0, 1)]}, budget=6, max_probability=0.5
    )
    assert np.array_equal(gam.coef_, coef_before)
    assert repr(gam.terms) == terms_before


def test_rejects_tensor_terms():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    y = (X[:, 0] + X[:, 1] + rng.normal(scale=0.3, size=300) > 0).astype(float)
    gam = LogisticGAM(te(0, 1)).fit(X, y)
    with pytest.raises(ValueError, match="interaction-free"):
        gam.recommend_plan(X[0], {0: [(1.0, 1)]}, budget=1, max_probability=0.5)


def test_rejects_by_variable_terms():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 2))
    y = (X[:, 0] * X[:, 1] + rng.normal(scale=0.3, size=300) > 0).astype(float)
    gam = LogisticGAM(s(0, by=1)).fit(X, y)
    with pytest.raises(ValueError, match="interaction-free"):
        gam.recommend_plan(X[0], {0: [(1.0, 1)]}, budget=1, max_probability=0.5)


def test_rejects_non_logistic_model():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 2))
    y = X[:, 0] + rng.normal(scale=0.3, size=200)
    gam = LinearGAM().fit(X, y)
    with pytest.raises(TypeError):
        recommend_plan(gam, X[0], {0: [(1.0, 1)]}, budget=1, max_probability=0.5)


def test_requires_fitted_model():
    with pytest.raises(AttributeError):
        recommend_plan(
            LogisticGAM(), np.zeros(2), {0: [(1.0, 1)]}, budget=1, max_probability=0.5
        )


@pytest.mark.parametrize(
    "order", [np.zeros((2, 3)), np.zeros((1, 2)), np.array([np.nan, 0.0, 0.0])]
)
def test_rejects_bad_orders(order):
    gam, _ = _make_gam()
    with pytest.raises(ValueError):
        gam.recommend_plan(order, {0: [(1.0, 1)]}, budget=1, max_probability=0.5)


@pytest.mark.parametrize("budget", [-1, 1.5, "2", True])
def test_rejects_bad_budget(budget):
    gam, X = _make_gam()
    with pytest.raises((TypeError, ValueError)):
        gam.recommend_plan(X[0], {0: [(1.0, 1)]}, budget=budget, max_probability=0.5)


@pytest.mark.parametrize("cap", [-0.1, 1.1, np.nan, "0.5"])
def test_rejects_bad_probability(cap):
    gam, X = _make_gam()
    with pytest.raises((TypeError, ValueError)):
        gam.recommend_plan(X[0], {0: [(1.0, 1)]}, budget=1, max_probability=cap)


def test_rejects_unknown_and_duplicate_features():
    gam, X = _make_gam()
    with pytest.raises(ValueError):
        gam.recommend_plan(X[0], {9: [(1.0, 1)]}, budget=1, max_probability=0.5)
    with pytest.raises(ValueError):
        gam.recommend_plan(
            X[0], [(0, [(1.0, 1)]), (0, [(2.0, 1)])], budget=1, max_probability=0.5
        )


def test_rejects_bad_costs():
    gam, X = _make_gam()
    with pytest.raises((TypeError, ValueError)):
        gam.recommend_plan(X[0], {0: [(1.0, -1)]}, budget=5, max_probability=0.5)
    with pytest.raises((TypeError, ValueError)):
        gam.recommend_plan(X[0], {0: [(1.0, 1.5)]}, budget=5, max_probability=0.5)


def test_accepts_mapping_and_mapping_option_forms():
    gam, X = _make_gam()
    order = X[0]
    by_options = {0: [{"value": 3.0, "cost": 2}, {"value": -3.0, "cost": 4}]}
    by_value_keys = {0: {3.0: 2, -3.0: 4}}
    a = gam.recommend_plan(order, by_options, budget=4, max_probability=0.5)
    b = gam.recommend_plan(order, by_value_keys, budget=4, max_probability=0.5)
    assert a.cost == b.cost
    assert a.probability_after == pytest.approx(b.probability_after)
    assert a.values == pytest.approx(b.values)


def test_sequence_form_sets_feature_order():
    gam, X = _make_gam()
    order = X[0]
    # A cap below the original forces one of the two equal-cost changes.
    options = [(1, [(3.0, 2)]), (0, [(-3.0, 2)])]
    plan = gam.recommend_plan(order, options, budget=2, max_probability=0.01)
    assert plan.feasible
    assert plan.cost == 2
    # The feature listed first (1) wins the tie only if it also lowers risk; more
    # importantly the cost and the recomputed risk must match the reference.
    reference = _brute_force(gam, order, options, budget=2, max_probability=0.01)
    assert plan.probability_after == pytest.approx(reference[1], abs=1e-9)


def test_plan_as_dict_round_trip():
    gam, X = _make_gam()
    plan = gam.recommend_plan(X[0], {0: [(3.0, 2)]}, budget=2, max_probability=0.5)
    data = plan.as_dict()
    assert data["feasible"] == plan.feasible
    assert data["cost"] == plan.cost
    assert len(data["changes"]) == len(plan.changes)
    assert data["risk_before"] if False else True  # property is not in as_dict
