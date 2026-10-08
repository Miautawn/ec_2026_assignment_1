"""The EAs the experiment can run: a name -> the EA class, a label and a colour.

To add one: `VARIANTS["my_ea"] = Variant(MyEA, "My EA", "#009E73")`.
"""

from harness.variants import Variant

from .ea import RandomSearch, SelfAdaptiveEA, StaticSigmaEA

#: name -> EA class, plus the label and colour every figure uses for it.
VARIANTS = {
    "random_search": Variant(RandomSearch, "Random search (baseline)", "#7F7F7F"),
    "static_sigma": Variant(StaticSigmaEA, "Fixed σ", "#0072B2"),
    "self_adaptive": Variant(SelfAdaptiveEA, "Self-adaptive σ", "#D55E00"),
}
