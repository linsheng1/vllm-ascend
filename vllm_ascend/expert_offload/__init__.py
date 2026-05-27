from .finemoe_simulator import (
    get_finemoe_data_simulator,
    has_finemoe_data_simulator,
    maybe_init_finemoe_data_simulator,
    maybe_record_finemoe_data,
)

__all__ = [
    "ExpertOffloadManager",
    "get_finemoe_data_simulator",
    "has_finemoe_data_simulator",
    "maybe_init_finemoe_data_simulator",
    "maybe_record_finemoe_data",
]


def __getattr__(name):
    if name == "ExpertOffloadManager":
        from .expert_offload_manager import ExpertOffloadManager

        return ExpertOffloadManager
    raise AttributeError(name)
