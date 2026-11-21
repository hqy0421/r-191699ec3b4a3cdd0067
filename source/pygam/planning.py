"""Budget-aware plan selection for fitted logistic GAMs.

The entry point :func:`recommend_plan` -- also available as
:meth:`pygam.LogisticGAM.recommend_plan` -- chooses new values for the adjustable
features of a single order so that the fitted model's predicted probability (for
example, the probability that the order is cancelled) is at or below a requested
cap, at the smallest possible total cost within a budget.

The fitted model is never modified. Candidate plans are scored using the model's
own additive structure, and the risk reported for the chosen plan is recomputed
with the model's existing ``predict_proba`` interface.

The search is a dynamic program over the integer budget. It does not enumerate
the cartesian product of the candidate values. The dynamic program is exact for
interaction-free GAMs, whose linear predictor is additive across features even
when a single feature appears in several terms.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

__all__ = ["Change", "Plan", "recommend_plan"]

# A plan that lands exactly on the cap can be off by a few ulps because the final
# risk is recomputed through ``predict_proba``. This tolerance keeps such plans
# from being rejected on numerical noise alone.
_RISK_TOL = 1e-12

# Two linear-predictor changes that differ by less than this are treated as a
# tie, so that the stable option-order rule decides instead of float noise.
_LP_TOL = 1e-12


@dataclass(frozen=True)
class Change:
    """A single feature change inside a recommended plan.

    Attributes
    ----------
    feature : int
        Index of the feature that changes.
    old_value : float
        Value of the feature in the original order.
    new_value : float
        Value of the feature in the recommended plan.
    cost : int
        Cost of choosing ``new_value`` for this feature.
    """

    feature: int
    old_value: float
    new_value: float
    cost: int

    def as_dict(self):
        """Return the change as a plain dictionary."""
        return {
            "feature": self.feature,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "cost": self.cost,
        }


@dataclass(frozen=True)
class Plan:
    """Result of :func:`recommend_plan`.

    Attributes
    ----------
    feasible : bool
        Whether a plan within the budget reaches the requested probability cap.
    cost : int or None
        Total cost of the recommended plan, or ``None`` when infeasible.
    changes : tuple of Change
        The features that actually change. Features that keep their original
        value are omitted (even though they are part of the plan).
    probability_before : float
        Predicted probability for the original order.
    probability_after : float or None
        Predicted probability for the recommended plan, recomputed with
        ``predict_proba``. ``None`` when infeasible.
    values : tuple of float or None
        Full feature vector of the recommended plan (fixed features keep the
        values supplied with the order). ``None`` when infeasible.
    budget : int
        The budget the plan was searched under.
    max_probability : float
        The requested probability cap.
    message : str
        Human-readable summary, including the best achievable probability when
        no plan is feasible.
    """

    feasible: bool
    cost: int | None
    changes: tuple
    probability_before: float
    probability_after: float | None
    values: tuple | None
    budget: int
    max_probability: float
    message: str

    @property
    def risk_before(self):
        """Alias for :attr:`probability_before`."""
        return self.probability_before

    @property
    def risk_after(self):
        """Alias for :attr:`probability_after`."""
        return self.probability_after

    @property
    def total_cost(self):
        """Alias for :attr:`cost`."""
        return self.cost

    def __bool__(self):
        return self.feasible

    def as_dict(self):
        """Return the plan as a plain dictionary."""
        return {
            "feasible": self.feasible,
            "cost": self.cost,
            "changes": [change.as_dict() for change in self.changes],
            "probability_before": self.probability_before,
            "probability_after": self.probability_after,
            "values": None if self.values is None else list(self.values),
            "budget": self.budget,
            "max_probability": self.max_probability,
            "message": self.message,
        }


def recommend_plan(gam, order, adjustable, budget, max_probability):
    """Recommend the cheapest plan that lowers an order's predicted risk.

    Parameters
    ----------
    gam : LogisticGAM
        A fitted, interaction-free logistic GAM.
    order : array-like of shape (m_features,) or (1, m_features)
        The features of a single order. Features that are not adjustable are
        held exactly at these values.
    adjustable : mapping or sequence
        The adjustable features. A mapping is keyed by feature index; its values
        describe the candidate values and their integer costs. A sequence of
        ``(feature, options)`` pairs may be used instead, which fixes the order
        in which features participate in tie-breaking.

        Each set of options may be given as

        - a sequence of ``(value, cost)`` pairs,
        - a sequence of ``{"value": ..., "cost": ...}`` mappings, or
        - a mapping of ``value -> cost``.

        Keeping a feature at its original value is always allowed, implicitly,
        at zero cost. It is treated as the first option of each feature. If the
        original value is also listed, the free implicit option is used.
    budget : int
        Maximum total cost of the plan. Must be a non-negative integer.
    max_probability : float
        The largest acceptable predicted probability, in ``[0, 1]``.

    Returns
    -------
    Plan
        The chosen plan, its cost, the changes, and the probabilities before and
        after. When no plan within ``budget`` reaches ``max_probability``, the
        result has ``feasible == False`` (and is falsy) and a message explains
        the best probability that is achievable within the budget.

    Raises
    ------
    ValueError
        If the model contains interaction terms, the order is not a single row,
        or the adjustable/cost/budget/cap arguments are invalid.

    Notes
    -----
    For an interaction-free GAM the linear predictor is additive across
    features, even when one feature appears in several terms. The dynamic
    program therefore minimises the summed linear-predictor change for every
    achievable cost, which is equivalent to minimising the predicted
    probability because the logit link is strictly increasing. Ties on cost are
    settled by lower predicted risk; remaining ties are settled by the option
    order, comparing option indices feature by feature in the order the
    adjustable features were supplied.
    """
    _validate_gam(gam)
    X = _as_single_order(gam, order)
    budget = _as_budget(budget)
    max_probability = _as_probability(max_probability)
    variables = _normalize_adjustable(adjustable, X.shape[1])

    base_values = X[0].copy()
    probability_before = float(np.ravel(gam.predict_proba(X))[0])
    lp_base = float(np.ravel(gam._linear_predictor(X))[0])
    if not np.isfinite(lp_base):
        raise ValueError("the fitted model produced a non-finite linear predictor")

    # For every adjustable feature, score each candidate once. Perturbing one
    # feature at a time captures every term that uses that feature, so a feature
    # appearing in several terms is handled without double counting.
    plans = []
    for feature, options in variables:
        choices = _build_choices(base_values[feature], options)
        deltas = np.zeros(len(choices), dtype=float)
        if len(choices) > 1:
            rows = np.repeat(X, len(choices), axis=0)
            rows[:, feature] = [choice[0] for choice in choices]
            try:
                lps = np.ravel(gam._linear_predictor(rows))
            except ValueError as exc:
                raise ValueError(
                    f"invalid candidate value for feature {feature}: {exc}"
                ) from exc
            deltas = lps - lp_base
            deltas[0] = 0.0  # keeping the original value changes nothing
            if not np.isfinite(lps).all():
                raise ValueError(
                    f"candidate values for feature {feature} produce a "
                    "non-finite linear predictor"
                )
        plans.append((feature, choices, deltas))

    threshold_delta = _logit(max_probability) - lp_base

    # Dynamic program over the integer budget. ``states`` maps a total cost to
    # the lowest linear-predictor change achievable at that cost, together with
    # the option indices that realise it (used for the stable tie-break).
    states = {0: (0.0, ())}
    for _feature, choices, deltas in plans:
        next_states = {}
        for cost, (lp, key) in states.items():
            for index, (_value, option_cost) in enumerate(choices):
                new_cost = cost + option_cost
                if new_cost > budget:
                    continue
                new_lp = lp + deltas[index]
                new_key = key + (index,)
                current = next_states.get(new_cost)
                if (
                    current is None
                    or new_lp < current[0] - _LP_TOL
                    or (abs(new_lp - current[0]) <= _LP_TOL and new_key < current[1])
                ):
                    next_states[new_cost] = (new_lp, new_key)
        if not next_states:
            states = {}
            break
        states = _prune(next_states)

    chosen = _choose_plan(gam, X, plans, states, threshold_delta, max_probability)

    if chosen is None:
        return _infeasible_plan(
            gam, X, plans, states, budget, max_probability, probability_before
        )

    cost, key, row, probability_after = chosen
    changes = []
    for (feature, choices, _deltas), index in zip(plans, key):
        value, option_cost = choices[index]
        old_value = float(base_values[feature])
        if value != old_value:
            changes.append(
                Change(
                    feature=int(feature),
                    old_value=old_value,
                    new_value=float(value),
                    cost=int(option_cost),
                )
            )

    message = (
        f"Plan found within budget {budget}: cost {cost}, predicted "
        f"probability {probability_after:.6g} <= {max_probability:.6g}."
    )
    return Plan(
        feasible=True,
        cost=int(cost),
        changes=tuple(changes),
        probability_before=probability_before,
        probability_after=float(probability_after),
        values=tuple(float(value) for value in row[0]),
        budget=budget,
        max_probability=float(max_probability),
        message=message,
    )


def _validate_gam(gam):
    """Ensure ``gam`` is fitted, logistic, and interaction-free."""
    if not getattr(gam, "_is_fitted", False):
        raise AttributeError("GAM has not been fitted. Call fit first.")

    if not (hasattr(gam, "predict_proba") and hasattr(gam, "terms")):
        raise TypeError("recommend_plan requires a fitted LogisticGAM.")

    for term in gam.terms:
        if getattr(term, "istensor", False):
            raise ValueError(
                "recommend_plan requires an interaction-free model, "
                "but found a tensor term. The linear predictor is not additive "
                "across features when terms interact."
            )
        if getattr(term, "by", None) is not None:
            raise ValueError(
                "recommend_plan requires an interaction-free model, "
                "but found a term with a by-variable. The linear predictor is "
                "not additive across features when terms interact."
            )


def _as_single_order(gam, order):
    """Validate and copy a single order's feature vector."""
    X = np.array(order, dtype=float, copy=True)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2 or X.shape[0] != 1:
        raise ValueError(
            "order must describe exactly one order with shape "
            f"(m_features,) or (1, m_features), but found shape {X.shape}"
        )
    n_features = gam.statistics_["m_features"]
    if X.shape[1] != n_features:
        raise ValueError(
            f"order must have {n_features} features, but found {X.shape[1]}"
        )
    if not np.isfinite(X).all():
        raise ValueError("order must not contain Inf nor NaN")
    return X


def _as_budget(budget):
    """Validate the budget as a non-negative integer."""
    if isinstance(budget, bool) or not isinstance(budget, Integral):
        raise TypeError(f"budget must be an integer, but found {budget!r}")
    budget = int(budget)
    if budget < 0:
        raise ValueError(f"budget must be non-negative, but found {budget}")
    return budget


def _as_probability(value):
    """Validate a probability cap in ``[0, 1]``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"max_probability must be a number, but found {value!r}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("max_probability must be finite")
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"max_probability must be between 0 and 1, but found {value}"
        )
    return value


def _as_feature(feature, n_features):
    """Validate a feature index."""
    if isinstance(feature, bool) or not isinstance(feature, Integral):
        raise TypeError(f"feature must be an integer index, but found {feature!r}")
    feature = int(feature)
    if feature < 0 or feature >= n_features:
        raise ValueError(
            f"feature {feature} is out of range for a model with "
            f"{n_features} features"
        )
    return feature


def _as_value(feature, value):
    """Validate a candidate feature value."""
    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"candidate value for feature {feature} must be numeric, "
            f"but found {value!r}"
        )
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"candidate value for feature {feature} must be numeric, "
            f"but found {value!r}"
        ) from exc
    if not np.isfinite(value):
        raise ValueError(f"candidate value for feature {feature} must be finite")
    return value


def _as_cost(feature, cost):
    """Validate an option cost as a non-negative integer."""
    if isinstance(cost, bool) or not isinstance(cost, Integral):
        raise TypeError(
            f"cost for feature {feature} must be an integer, but found {cost!r}"
        )
    cost = int(cost)
    if cost < 0:
        raise ValueError(
            f"cost for feature {feature} must be non-negative, but found {cost}"
        )
    return cost


def _parse_option(feature, option):
    """Parse one option into a ``(value, cost)`` pair."""
    if isinstance(option, Mapping):
        if "value" not in option or "cost" not in option:
            raise ValueError(
                f"each option for feature {feature} must contain both "
                "'value' and 'cost'"
            )
        value, cost = option["value"], option["cost"]
    else:
        if isinstance(option, (str, bytes)):
            raise TypeError(
                f"each option for feature {feature} must be a (value, cost) "
                "pair or a mapping"
            )
        try:
            value, cost = option
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"each option for feature {feature} must be a (value, cost) "
                f"pair or a mapping, but found {option!r}"
            ) from exc
    return _as_value(feature, value), _as_cost(feature, cost)


def _normalize_options(feature, options):
    """Normalize a feature's options to a list of ``(value, cost)`` pairs."""
    if isinstance(options, Mapping):
        if "value" in options and "cost" in options:
            # A single option expressed as a mapping.
            return [_parse_option(feature, options)]
        return [
            (_as_value(feature, value), _as_cost(feature, cost))
            for value, cost in options.items()
        ]

    if isinstance(options, (str, bytes)) or not isinstance(options, Iterable):
        raise TypeError(
            f"options for feature {feature} must be a mapping or an iterable "
            "of options"
        )
    return [_parse_option(feature, option) for option in options]


def _normalize_adjustable(adjustable, n_features):
    """Normalize the adjustable-feature specification, preserving order."""
    if isinstance(adjustable, Mapping):
        specs = list(adjustable.items())
    elif isinstance(adjustable, (str, bytes)) or not isinstance(adjustable, Iterable):
        raise TypeError(
            "adjustable must be a mapping of feature -> options or a sequence "
            "of (feature, options) pairs"
        )
    else:
        specs = []
        for spec in adjustable:
            if isinstance(spec, Mapping):
                if "feature" not in spec or "options" not in spec:
                    raise ValueError(
                        "each adjustable feature mapping must contain "
                        "'feature' and 'options'"
                    )
                specs.append((spec["feature"], spec["options"]))
            elif hasattr(spec, "feature") and hasattr(spec, "options"):
                specs.append((spec.feature, spec.options))
            else:
                try:
                    feature, options = spec
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        "each adjustable entry must be a (feature, options) "
                        f"pair, but found {spec!r}"
                    ) from exc
                specs.append((feature, options))

    normalized = []
    seen = set()
    for feature, options in specs:
        feature = _as_feature(feature, n_features)
        if feature in seen:
            raise ValueError(f"feature {feature} appears more than once in adjustable")
        seen.add(feature)
        normalized.append((feature, _normalize_options(feature, options)))
    return normalized


def _build_choices(original, options):
    """Build the per-feature choices, always allowing the original value.

    The implicit "keep current value" choice is first and costs nothing. Values
    supplied by the caller that equal the original are dropped. Other duplicate
    values keep the cheapest cost, and equal costs keep the earliest option so
    that ties remain stable.
    """
    original = float(original)
    cheapest = {}
    for position, (value, cost) in enumerate(options):
        if value == original:
            continue
        if value not in cheapest or cost < cheapest[value][0]:
            cheapest[value] = (cost, position)
    ordered = sorted(cheapest.items(), key=lambda item: item[1][1])
    return [(original, 0)] + [(value, cost) for value, (cost, _position) in ordered]


def _prune(states):
    """Drop dominated states.

    A state is dominated when another state costs no more and leaves the linear
    predictor no higher. Dropping it is safe because the remaining choices add
    the same amount to both, and because a more expensive state with an equal
    linear predictor can never be the cheapest feasible plan.
    """
    pruned = {}
    best_lp = np.inf
    for cost in sorted(states):
        lp, key = states[cost]
        if lp < best_lp:
            pruned[cost] = (lp, key)
            best_lp = lp
    return pruned


def _build_row(X, plans, key):
    """Build the feature row for a chosen plan."""
    row = X.copy()
    for (feature, choices, _deltas), index in zip(plans, key):
        row[0, feature] = choices[index][0]
    return row


def _choose_plan(gam, X, plans, states, threshold_delta, max_probability):
    """Pick the cheapest feasible plan and recompute its risk.

    Costs are scanned in ascending order. For each cost the dynamic program
    already holds the lowest-risk plan, so the first cost whose risk passes the
    cap is the answer. The reported probability is always recomputed with
    ``predict_proba``.
    """
    costs = sorted(
        cost for cost, (lp, _key) in states.items() if lp <= threshold_delta + _LP_TOL
    )
    for cost in costs:
        _lp, key = states[cost]
        row = _build_row(X, plans, key)
        probability_after = float(np.ravel(gam.predict_proba(row))[0])
        if probability_after <= max_probability + _RISK_TOL:
            return cost, key, row, probability_after
    return None


def _infeasible_plan(
    gam, X, plans, states, budget, max_probability, probability_before
):
    """Build the explicit no-plan result, including the best achievable risk."""
    best_probability = None
    if states:
        _best_cost, (best_lp, best_key) = min(
            states.items(), key=lambda item: item[1][0]
        )
        row = _build_row(X, plans, best_key)
        best_probability = float(np.ravel(gam.predict_proba(row))[0])

    if best_probability is None:
        detail = "no candidate plan could be evaluated"
    else:
        detail = (
            f"the lowest predicted probability achievable within budget "
            f"{budget} is {best_probability:.6g}"
        )
    message = (
        f"No plan within budget {budget} brings the predicted probability "
        f"down to {max_probability:.6g}; {detail}."
    )
    return Plan(
        feasible=False,
        cost=None,
        changes=(),
        probability_before=probability_before,
        probability_after=None,
        values=None,
        budget=budget,
        max_probability=float(max_probability),
        message=message,
    )


def _logit(probability):
    """Logit of a probability, returning ``+/-inf`` at the boundaries."""
    if probability <= 0.0:
        return -np.inf
    if probability >= 1.0:
        return np.inf
    return float(np.log(probability) - np.log1p(-probability))
