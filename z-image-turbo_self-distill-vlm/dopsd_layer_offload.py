from __future__ import annotations

import sys
from pathlib import Path


runtime_path = Path(__file__).resolve().parents[1] / "trainer_runtime"
if str(runtime_path) not in sys.path:
    sys.path.insert(0, str(runtime_path))

from dopsd_trainer.layer_offload import attach_layer_offload  # noqa: E402

