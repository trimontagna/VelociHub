"""Benchmark of VelociHub on analytic flows: their exact displacement fields sampled as the PIV/PTV would give them,
run through VelociHub, and compared with the exact fields, with checks and a report (report.py). Run when VelociHub
changes, to catch the regressions and to see how new features behave:

    python -m velocihub.benchmark                     # all the cases, the report in ~/VelociHub/benchmarks
    python -m velocihub.benchmark cylinder jet --out report --compare previous_report

or in the interface (menu Tests), or as tests (tests/test_benchmark.py). For now the level of the regression only:
the data that follow the PIV software (particle links, vectors), without images (regression_level.py)."""

from .cases import CASES, METHODS, SCENARIOS
from .flows import FLOWS
from .regression_level import run_benchmark

__all__ = ["CASES", "FLOWS", "METHODS", "SCENARIOS", "run_benchmark"]
