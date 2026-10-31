"""Information-barrier helpers for sample metadata.

The campaign planner records both simulator-facing fields (situation,
character notes) and control-layer fields (target mode/strategy, private
rationale, naturalization constraints). The simulator must never see the
control-layer fields, or its behavior degrades into "operator following
explicit attack targets" instead of "user pursuing a situational goal."

Convention: anything the simulator should not see is nested under
``metadata["_private"]``. ``public_metadata`` makes the split explicit at
every prompt-render boundary.
"""

from __future__ import annotations

from typing import Any

PRIVATE_KEY = "_private"


def public_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Return metadata with the private sub-dict stripped.

    Use this at every render boundary that feeds the simulator (or any
    other agent that must remain ignorant of control-layer fields).
    """
    if not metadata:
        return {}
    return {k: v for k, v in metadata.items() if k != PRIVATE_KEY}

