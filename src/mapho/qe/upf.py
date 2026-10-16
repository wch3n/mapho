"""Inspect UPF provenance and restrict the first backend to NC/PBE scalar data."""

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET


def read_upf_metadata(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError("This backend requires a well-formed UPF v2 XML file") from exc
    header = root.find("PP_HEADER")
    if (
        root.tag != "UPF"
        or header is None
        or not root.attrib.get("version", "").startswith("2.")
    ):
        raise ValueError("Expected UPF v2 with a PP_HEADER")
    h = header.attrib
    flag = lambda key: h.get(key, "F").strip().upper() in ("T", "TRUE", ".TRUE.")
    if (
        h.get("pseudo_type", "").strip() != "NC"
        or flag("is_paw")
        or flag("is_ultrasoft")
    ):
        raise ValueError("Only norm-conserving pseudopotentials are supported")
    if flag("has_so") or h.get("relativistic", "").strip() == "full":
        raise ValueError("Spin-orbit pseudopotentials are not supported")
    if h.get("functional", "").strip().upper() != "PBE":
        raise ValueError("The first backend requires a PBE pseudopotential")
    for tag in ("PP_LOCAL", "PP_NONLOCAL", "PP_RHOATOM"):
        if root.find(tag) is None:
            raise ValueError(f"Missing required UPF section {tag}")
    zval = float(h["z_valence"])
    if not 0 < zval <= 118:
        raise ValueError("Invalid valence charge")
    return {
        "filename": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "element": h["element"].strip(),
        "pseudo_type": "NC",
        "functional": "PBE",
        "z_valence": zval,
        "relativistic": h.get("relativistic", "").strip(),
        "core_correction": flag("core_correction"),
        "number_of_proj": int(h["number_of_proj"]),
    }
