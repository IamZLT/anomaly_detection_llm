"""Input plumbing for outcome-multibox-v1: reuse the single-box collator with
multibox GT validation swapped in."""
from __future__ import annotations

from outcome import protocol_multibox as pb
from outcome.inputs import OutcomeCollator, OutcomeDataset


class OutcomeMultiboxDataset(OutcomeDataset):
    validate_gt_fn = staticmethod(pb.validate_gt)


class OutcomeMultiboxCollator(OutcomeCollator):
    validate_gt_fn = staticmethod(pb.validate_gt)
