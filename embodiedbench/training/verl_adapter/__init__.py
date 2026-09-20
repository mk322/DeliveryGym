"""verl adapter (design plan §11, Contract F).

Converts our trainer-neutral rollout into verl's ``DataProto`` and drives verl's
own actor. The point of R1 is to test *verl's* handling of multimodal multi-turn
masking and log-probs (design plan §11.1), so the optimizer step, the loss, and the
log-prob recomputation are all verl's. Nothing here reimplements them.
"""

from embodiedbench.training.verl_adapter.dataproto import (
    RolloutLayout,
    build_dataproto,
    describe_layout,
)

__all__ = ["RolloutLayout", "build_dataproto", "describe_layout"]
