# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""SM100 DSL SDPA-forward kernel templates.

Filenames encode the coverage matrix: ``<phase>_d<dim>_<dtype-family>_sm<arch>.py``
(e.g. ``prefill_d512_f16_sm100.py`` — f16 covers fp16 and bf16, picked by TemplateParams).

Each template specializes on a :class:`cudnn.sdpa.fwd.config_sm100.TemplateParams`
instance at import time (module global ``FROST_TEMPLATE_PARAMS``, injected by
``cudnn.frost.template_loader.load_template``). Import one directly only for
the all-defaults standalone/benchmark path (``python <template>.py``).
"""
