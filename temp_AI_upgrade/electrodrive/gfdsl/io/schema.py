"""Schema constants for GFDSL serialization."""

SCHEMA_NAME = "electrodrive.gfdsl"
SCHEMA_VERSION = 1


def schema_header() -> dict:
    return {"schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION}


def check_schema(header: dict) -> None:
    name = header.get("schema_name")
    version = header.get("schema_version")
    if name != SCHEMA_NAME:
        raise ValueError(f"Unknown GFDSL schema '{name}' (expected {SCHEMA_NAME})")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Incompatible GFDSL schema_version={version}, expected {SCHEMA_VERSION}"
        )

