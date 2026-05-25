from .simulator import (
    ExpertOffloadSimulator,
    get_expert_offload_simulator,
    has_expert_offload_simulator,
    maybe_init_expert_offload_simulator,
    maybe_record_expert_offload_simulation,
)

__all__ = [
    "ExpertOffloadManager",
    "ExpertOffloadSimulator",
    "get_expert_offload_simulator",
    "has_expert_offload_simulator",
    "maybe_init_expert_offload_simulator",
    "maybe_record_expert_offload_simulation",
]


def __getattr__(name):
    if name == "ExpertOffloadManager":
        from .expert_offload_manager import ExpertOffloadManager

        return ExpertOffloadManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
