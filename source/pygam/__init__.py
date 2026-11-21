"""GAM toolkit"""

from importlib.metadata import PackageNotFoundError, version

from pygam.pygam import (
    GAM,
    ExpectileGAM,
    GammaGAM,
    InvGaussGAM,
    LinearGAM,
    LogisticGAM,
    PoissonGAM,
)
from pygam.planning import Change, Plan, recommend_plan
from pygam.terms import f, intercept, l, s, te

__all__ = [
    "GAM",
    "LinearGAM",
    "LogisticGAM",
    "GammaGAM",
    "PoissonGAM",
    "InvGaussGAM",
    "ExpectileGAM",
    "Change",
    "Plan",
    "recommend_plan",
    "l",
    "s",
    "f",
    "te",
    "intercept",
]


__version__ = "0.0.0"  # placeholder for dynamic versioning
try:
    __version__ = version("pygam")
except PackageNotFoundError:
    # package is not installed
    pass
