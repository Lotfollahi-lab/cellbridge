"""Model adapters used by the shared embedding runner."""

from importlib import import_module

METHODS = (
    "uni",
    "cellbridge",
    "past",
    "volta",
    "spatialwhisperer",
    "lemon",
    "virchow2",
    "hibou_l",
)

_MODULES = {
    "uni": "uni",
    "cellbridge": "lora",
    "past": "past",
    "volta": "volta",
    "spatialwhisperer": "spatialwhisperer",
    "lemon": "lemon",
    "virchow2": "virchow2",
    "hibou_l": "hibou_l",
}


def _method(name):
    try:
        module = _MODULES[name]
    except KeyError as error:
        raise ValueError(
            f"unknown embedding method {name!r}; choose from {METHODS}"
        ) from error
    return import_module(f"{__name__}.{module}")


def arms(name) -> tuple[str, ...]:
    """Return the named representations emitted by one method."""
    return tuple(_method(name).ARMS)


def load(name, **kwargs):
    """Load one registered embedding method."""
    return _method(name).load(**kwargs)
