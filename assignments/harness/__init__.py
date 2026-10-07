"""Experiment machinery shared by Assignments 1 and 2.

Runs every (EA, seed) pair in parallel, turns the run databases into tables,
compares EAs statistically and draws the shared figures. Each assignment brings
its own EAs and fitness, and puts `assignments/` on the path so this package
can be imported as `harness`.
"""
