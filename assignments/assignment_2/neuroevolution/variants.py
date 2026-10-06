"""The variant registry: which EAs exist, and how every figure shows them.

To add an EA: subclass `ea.NeuroEA` (see its docstring), then register it here.
The harness runs anything registered -- nothing else needs to change.

    VARIANTS["my_ea"] = Variant(MyEA, "My EA (what it does)", "#0072B2")

Colour-blind-safe colours that pair with the baseline's grey:
#0072B2 (blue), #D55E00 (vermillion), #009E73 (green), #CC79A7 (pink).
"""

from harness.variants import Variant

from .ea import RandomSearch

#: name -> EA class, plus the label and colour every figure uses for it.
VARIANTS = {
    "random_search": Variant(RandomSearch, "Random search (baseline)", "#7F7F7F"),
}
