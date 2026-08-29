"""WindFusion v0.2 - physics-confidence fusion for wind-turbine PdM.

Advisory-only research software. It must never be connected to turbine control.
"""

from .provenance import PROVENANCE, provenance_table, validate_provenance

__version__ = "0.2.0"

__all__ = ["PROVENANCE", "__version__", "provenance_table", "validate_provenance"]
