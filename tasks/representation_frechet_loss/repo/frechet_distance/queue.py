import logging

import torch


logger = logging.getLogger("FD_loss")


class FeatureQueue(torch.nn.Module):
    def __init__(
        self,
        size: int = 50000,
        feat_dim: int = 2048,
        online_accum: bool = False,
        ema_beta: float = 0.0,
    ):
        super().__init__()
        self.size = size
        self.feat_dim = feat_dim
        self.online_accum = online_accum
        self.ema_beta = ema_beta
        self.ema_stats = ema_beta > 0.0

        if self.ema_stats:
            self.register_buffer("mu_ema", torch.zeros(feat_dim, dtype=torch.float64))
            self.register_buffer("m2_ema", torch.zeros(feat_dim, feat_dim, dtype=torch.float64))
            self.register_buffer("_ema_count", torch.zeros(1, dtype=torch.long))
        else:
            self.register_buffer("feats", torch.empty(size, feat_dim))
            self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
            if online_accum and size > 0:
                self.register_buffer("feat_sum_old", torch.zeros(feat_dim, dtype=torch.float64))
                self.register_buffer("feat_outer_old", torch.zeros(feat_dim, feat_dim, dtype=torch.float64))

    @property
    def pointer(self) -> int:
        raise NotImplementedError("FeatureQueue.pointer must be implemented")

    @torch.no_grad()
    def _init_accumulators(self):
        raise NotImplementedError("FeatureQueue._init_accumulators must be implemented")

    @torch.no_grad()
    def accumulate_batch(self, feats: torch.Tensor):
        raise NotImplementedError("FeatureQueue.accumulate_batch must be implemented")

    @torch.no_grad()
    def _finalize_streaming_init(self):
        raise NotImplementedError("FeatureQueue._finalize_streaming_init must be implemented")

    def build_feats_stats(self, new_feats: torch.Tensor):
        raise NotImplementedError("FeatureQueue.build_feats_stats must be implemented")

    def _build_feats_stats_ema(self, new_feats: torch.Tensor):
        raise NotImplementedError("FeatureQueue statistics must be implemented")

    def _snapshot(self, buf: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("FeatureQueue snapshots must be implemented")

    def build_feats_snapshot(self, new_feats: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("FeatureQueue.build_feats_snapshot must be implemented")

    @torch.no_grad()
    def enqueue(self, new_feats: torch.Tensor):
        raise NotImplementedError("FeatureQueue.enqueue must be implemented")

    def _get_evicted_feats(self, n: int) -> torch.Tensor:
        raise NotImplementedError("FeatureQueue eviction must be implemented")
