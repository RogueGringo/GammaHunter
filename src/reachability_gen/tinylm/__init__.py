# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""A small language model built from the ground up: byte tokens and a decoder in plain PyTorch.

No pretrained weights and no model library: every parameter starts random and
every operation is written here, so what the model can read is exactly what it
was trained on.
"""
