"""Experiment suites per stage, defined in a spec file and reviewed in MLflow; see ``cli.py`` for the workflow."""

from .spec import Arm, Experiment, MetricInfo, Spec, Variant

__all__ = ['Arm', 'Experiment', 'MetricInfo', 'Spec', 'Variant']
