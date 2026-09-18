"""Experiment harness for Standard Assignment 1.

The harness is deliberately split so that the *evolutionary algorithm* and the
*experiment around it* can be developed independently:

    config.py          all tunable parameters, in one frozen dataclass
    fitness.py         target bodies + the assignment's official fitness
    variants.py        the EA subclasses + registry.  THE ONLY SWAP POINT.
    runner.py          executes the (variant x seed) grid, one database each
    dataset.py         database -> tidy DataFrame with a frozen schema
    analysis.py        theoretical floor, summary table, significance tests
    figures.py         every figure that goes into the report

Nothing downstream of `variants.py` knows or cares which EA produced a run,
because every EA writes the same ARIEL database schema.
"""
