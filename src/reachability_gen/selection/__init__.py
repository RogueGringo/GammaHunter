# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Selection normalizers: softmax, entmax-1.5, sparsemax and length-scaled softmax (SSMax).

A portable package: relative imports only, and nothing beyond the standard
library and torch (``tests/test_selection_isolation.py`` enforces both), so it
can be lifted into another repository or re-implemented as new kernels.

* ``spec``: what each normalizer is (its α, its closed form, its guarantees);
* ``reference``: torch backends with exact sort-based forwards and analytic
  backward passes, masking by exact zeros;
* ``certificate``: a framework-free KKT checker that grades any kernel's output
  without trusting any implementation (exact in ``Fraction`` for sparsemax);
* ``conformance``: the battery every kernel must pass, and a deliberately
  broken backend that it is tested to reject;
* ``stage_a``: the operator-level measurement (margin needed against the number
  of distractors).

``science_open=false`` always.
"""

from .spec import NORMALIZERS, NormalizerSpec

__all__ = ["NORMALIZERS", "NormalizerSpec"]
