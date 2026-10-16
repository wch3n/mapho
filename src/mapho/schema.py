"""Read legacy schema names while writing the canonical MAPHO namespace."""


def canonical_schema(value):
    """Normalize only the former project prefix; preserve format and version."""
    if isinstance(value, str) and value.startswith("mlham."):
        return "mapho." + value[len("mlham.") :]
    return value
