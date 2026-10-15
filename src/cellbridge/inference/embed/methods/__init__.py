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
)

_MODULES = {
    "uni": "uni",
    "cellbridge": "lora",
    "past": "past",
    "volta": "volta",
    "spatialwhisperer": "spatialwhisperer",
    "lemon": "lemon",
    "virchow2": "virchow2",
}


def _method(name):
    try:
        module = _MODULES[name]
    except KeyError as error:
        raise ValueError(
            f"unknown embedding method {name!r}; choose from {METHODS}"
        ) from error
    return import_module(f"{__name__}.{module}")


def arms(name, selected=None) -> tuple[str, ...]:
    """Return the named representations emitted by one method."""
    available = tuple(_method(name).ARMS)
    if selected is None:
        return ("lora_adapted",) if name == "cellbridge" else available
    selected = tuple(dict.fromkeys(selected))
    if not selected or set(selected) - set(available):
        raise ValueError(f"invalid arms for {name}: {selected}; choose from {available}")
    return selected


def load(name, **kwargs):
    """Load one registered embedding method."""
    return _method(name).load(**kwargs)
