"""Placeholder for future PyTorch training/evaluation instrumentation.

PyTorch is intentionally not imported here: no hooks are implemented and loading
it in every Scikit-learn subprocess adds significant startup cost. Torch-based
runs with no supported events are UNVERIFIABLE in the current alpha.
"""


def install(trace, record, fingerprint, jsonable, nest, depth, label) -> bool:  # noqa: ARG001
    return False
