"""Training adapters (design plan §11, Contract F).

The simulator emits a trainer-neutral ``Trajectory``; framework adapters convert
it to framework-specific samples. This keeps the environment API from becoming a
verl or slime extension point.
"""

from embodiedbench.training.core import (
    TrainingSample,
    build_training_sample,
    validate_sample,
)

__all__ = ["TrainingSample", "build_training_sample", "validate_sample"]
