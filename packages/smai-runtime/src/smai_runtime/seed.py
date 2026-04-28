"""``seed_everything`` — runtime substrate seeding utility.

Per ``10-runtime-and-templates.md`` §8.1. Wraps ``random``, ``numpy.random``,
``torch.manual_seed``, ``torch.cuda.manual_seed_all``, and sets
``torch.backends.cudnn.deterministic = True``. ``numpy`` and ``torch`` are
imported lazily so the runtime substrate library is importable in
test/development environments without the full ML stack pre-installed
(per §8.5: the ML stack is a substrate-side guarantee, not a hard
smai-runtime install requirement).
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    """Seed every deterministic source the v1 ML stack exposes."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np  # type: ignore[import-not-found,import-untyped]
    except ImportError:
        pass
    else:
        np.random.seed(seed)  # type: ignore[no-untyped-call,attr-defined]

    try:
        import torch  # type: ignore[import-not-found,import-untyped]
    except ImportError:
        return

    torch.manual_seed(seed)  # type: ignore[no-untyped-call]
    if torch.cuda.is_available():  # type: ignore[no-untyped-call,attr-defined]
        torch.cuda.manual_seed_all(seed)  # type: ignore[no-untyped-call,attr-defined]
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
