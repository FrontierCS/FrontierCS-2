# Flow-OPD runtime assets

Powered by Stability AI.

The SD3.5 Medium runtime uses the Stability AI Community License, July 5, 2024,
at pinned revision `b940f670f0eda2d07fbb75229e779da1ad11eb80`. Its model card and
`LICENSE.md` were retrieved from that exact revision without credentials.
The pinned agreement is retained beside this notice as
`evidence/asset-notices/STABILITY_AI_LICENSE.md`.

The agreement's section IV(a) requires a license copy, a `Notice` attribution
file, and a prominent "Powered by Stability AI" attribution when the materials
or a product using them are made available. The original runtime image already includes the pinned agreement at
`/opt/flow-opd/STABILITY-AI-LICENSE.md` and the attribution at
`/opt/flow-opd/NOTICE`; the qualified public runtime inherits both files.
The public task documentation will point to those existing notices in its
ordinary wording revision. The upstream MIT source license remains distinct
from the model license. No asset rebuild is required for this documentation fix.

The OCR teacher is `jieliu/SD3.5M-FlowGRPO-Text` revision
`1a0e6426ac8b8f53347ac66d0d544a1fc2253791`. Its pinned card describes a Flow-GRPO
LoRA for SD3.5 Medium and provides loading code. The card declares no independent
license. The Flow-OPD repository's MIT source license is not evidence that these
teacher weights independently use MIT. Keep that redistribution limitation
explicit; existing authorized research access and measured execution do not
establish unrestricted public weight redistribution.

No weights or credentials are committed. Immutable runtime images retain the
already verified asset bytes. Public task layers contain neither the private
reference implementation nor hidden evaluation workloads.
