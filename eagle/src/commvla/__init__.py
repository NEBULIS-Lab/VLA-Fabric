"""CommVLA: communication-enabled multi-agent vision-language-action models."""

__all__ = [
    "CommVLANAgentConfig",
    "CommVLANAgentModel",
    "CommVLANativeV3Config",
    "CommVLANativeV3Pair",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name in __all__:
        from commvla import models

        return getattr(models, name)
    raise AttributeError(name)
