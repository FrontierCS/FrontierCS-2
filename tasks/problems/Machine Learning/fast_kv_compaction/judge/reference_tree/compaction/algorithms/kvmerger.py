# compaction/algorithms/kvmerger.py
"""
KVMerger-style KV cache compaction algorithm.

Implements the KVMerger approach (Wang et al., 2024, arxiv:2407.08454) within the
Attention Matching framework. Groups consecutive keys with high cosine similarity
using agglomerative hierarchical clustering, then merges each group via Gaussian
kernel weighted averaging.

Key differences from the paper's standalone implementation:
- Adapts to the (C1, beta, C2, indices) interface so it can be used as a drop-in
  replacement for any other compaction algorithm in this codebase.
- Optionally supports fitting beta (NNLS) and C2 (ridge regression) on top of
  the merged keys for improved quality.
"""
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
from .base import CompactionAlgorithm


class KVMergerCompaction(CompactionAlgorithm):
    """Merge consecutive similar keys via Gaussian kernel weighted averaging."""

    def __init__(
        self,
        top_k_ratio: float = 0.0,
        c2_method: str = 'merge',
        beta_method: str = 'zero',
        nnls_iters: int = 0,
        nnls_lower_bound: float = None,
        nnls_upper_bound: float = None,
        c2_ridge_lambda: float = 0,
        c2_solver: str = 'lstsq',
        c2_ridge_scale: str = 'spectral',
        threshold_search_steps: int = 20,
    ):
        """
        Parameters
        ----------
        top_k_ratio : float
            Fraction of keys protected from merging (attention sinks / heavy hitters).
            These are kept as singleton clusters. (default: 0.0).
        c2_method : str
            How to compute compacted values:
            - 'merge': Gaussian kernel weighted average of original values (KVMerger style)
            - 'lsq': Ridge regression fit (AM style, generally better quality)
            - 'direct': Use pivotal token's value directly
        beta_method : str
            How to compute bias terms:
            - 'zero': No bias correction (pure KVMerger)
            - 'nnls': Fit via NNLS to match partition function (AM-style enhancement)
        nnls_iters : int
            Projected gradient iterations for NNLS. 0 = clamped lstsq.
        nnls_lower_bound : float, optional
            Lower bound for NNLS solution.
        nnls_upper_bound : float, optional
            Upper bound for NNLS solution.
        c2_ridge_lambda : float
            Ridge regularization for C2 when c2_method='lsq'.
        c2_solver : str
            Solver for C2 regression: 'lstsq', 'cholesky', or 'pinv'.
        c2_ridge_scale : str
            How to scale ridge lambda: 'spectral', 'frobenius', or 'fixed'.
        threshold_search_steps : int
            Max binary search iterations to find threshold yielding target size t.
        """
        self.top_k_ratio = top_k_ratio
        if c2_method not in ['merge', 'lsq', 'direct']:
            raise ValueError(f"c2_method must be 'merge', 'lsq', or 'direct', got '{c2_method}'")
        self.c2_method = c2_method
        if beta_method not in ['zero', 'nnls']:
            raise ValueError(f"beta_method must be 'zero' or 'nnls', got '{beta_method}'")
        self.beta_method = beta_method
        self.nnls_iters = nnls_iters
        self.nnls_lower_bound = nnls_lower_bound
        self.nnls_upper_bound = nnls_upper_bound
        self.c2_ridge_lambda = c2_ridge_lambda
        self.c2_solver = c2_solver
        self.c2_ridge_scale = c2_ridge_scale
        self.threshold_search_steps = threshold_search_steps

    def name(self) -> str:
        return "KVMerger"

    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        attention_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Compute compacted cache using KVMerger-style clustering and merging.

        Parameters
        ----------
        K : Tensor, shape (T, d)
        V : Tensor, shape (T, d)
        queries : Tensor, shape (n, d)
        t : int
            Target compacted size.
        attention_bias : Tensor, optional

        Returns
        -------
        C1 : Tensor, shape (t, d) — merged keys
        beta : Tensor, shape (t,) — bias terms
        C2 : Tensor, shape (t, d) — merged/fitted values
        indices : list of int — pivotal indices (one per cluster)
        """
        T, d = K.shape
        device = K.device
        dtype_param = K.dtype

        if t >= T:
            # Nothing to compact
            beta = torch.zeros(T, dtype=dtype_param, device=device)
            return K.clone(), beta, V.clone(), list(range(T))

        # --- Step 1: Compute aggregated attention scores for pivotal state selection
        #     and protected key identification ---
        agg_attn = self._compute_aggregated_attention(K, queries, attention_bias)  # (T,)

        # Identify protected keys (top-k heavy hitters, not eligible for merging)
        n_protected = max(1, int(T * self.top_k_ratio))
        n_protected = min(n_protected, t)  # can't protect more than target size
        protected_indices = agg_attn.topk(n_protected).indices  # (n_protected,)
        protected_set = set(protected_indices.cpu().tolist())

        # --- Step 2: Compute adjacent cosine similarity ---
        K_norm = F.normalize(K.float(), dim=-1)  # (T, d) fp32
        adj_sim = (K_norm[:-1] * K_norm[1:]).sum(dim=-1)  # (T-1,)

        # --- Step 3: Binary search on threshold to get ~t merging sets ---
        merging_sets = self._find_merging_sets_with_target(
            adj_sim, protected_set, T, t
        )

        # --- Step 4: Gaussian kernel weighted merging ---
        C1_list = []
        C2_list = []
        pivotal_indices = []

        for s in merging_sets:
            if len(s) == 1:
                # Singleton: keep original key/value
                idx = s[0]
                C1_list.append(K[idx])
                C2_list.append(V[idx])
                pivotal_indices.append(idx)
            else:
                # Find pivotal state: member with highest aggregated attention
                set_attn = agg_attn[s]  # (|s|,)
                pivot_local = set_attn.argmax().item()
                pivot_idx = s[pivot_local]

                keys_in_set = K[s].float()  # (|s|, d)
                vals_in_set = V[s].float()  # (|s|, d)
                pivot_key = K[pivot_idx].float()  # (d,)

                # Gaussian kernel weights
                diffs = keys_in_set - pivot_key.unsqueeze(0)  # (|s|, d)
                dists_sq = (diffs ** 2).sum(dim=-1)  # (|s|,)

                # sigma = mean of Gaussian kernel values / sqrt(2) * |S_k|
                # Paper: sigma = sum(g_pi) / (sqrt(2) * |S_k|)
                # We compute g_pi first with a preliminary sigma, then set sigma
                # from the paper's formula. Use the mean distance as initial sigma.
                g_raw = torch.exp(-dists_sq / (2 * dists_sq.mean().clamp(min=1e-12)))
                sigma = g_raw.sum() / (len(s) * (2 ** 0.5))
                sigma = sigma.clamp(min=1e-8)

                # Recompute weights with final sigma
                g = torch.exp(-dists_sq / (2 * sigma ** 2))  # (|s|,)

                # Merging weights for keys: w_i = g_i / sum(g), w_p = 1/sum(g)
                g_sum = g.sum().clamp(min=1e-12)
                w_keys = g / g_sum  # (|s|,)

                # Merged key: weighted sum
                merged_key = (w_keys.unsqueeze(-1) * keys_in_set).sum(dim=0)  # (d,)

                # Merging weights for values: same weights but scaled by |S_v|
                # (paper eq 7: value weights include the set size multiplier)
                w_vals = w_keys * len(s)
                merged_val = (w_vals.unsqueeze(-1) * vals_in_set).sum(dim=0) / len(s)  # (d,)

                C1_list.append(merged_key.to(dtype_param))
                C2_list.append(merged_val.to(dtype_param))
                pivotal_indices.append(pivot_idx)

        C1 = torch.stack(C1_list)  # (t', d)
        C2_merged = torch.stack(C2_list)  # (t', d)
        actual_t = C1.shape[0]

        # --- Step 5: Compute beta ---
        if self.beta_method == 'zero':
            beta = torch.zeros(actual_t, dtype=dtype_param, device=device)
        else:  # 'nnls'
            beta = self._compute_beta_nnls(C1, K, queries, attention_bias)

        # --- Step 6: Compute C2 based on method ---
        if self.c2_method == 'merge':
            C2 = C2_merged
        elif self.c2_method == 'lsq':
            C2 = self._compute_C2(
                C1, beta, K, V, queries,
                attention_bias=attention_bias,
                ridge_lambda=self.c2_ridge_lambda,
                solver=self.c2_solver,
                ridge_scale=self.c2_ridge_scale,
            )
        elif self.c2_method == 'direct':
            C2 = self._direct_C2(C1, K, V, indices=pivotal_indices)

        return C1, beta, C2, pivotal_indices

    def _compute_aggregated_attention(
        self,
        K: torch.Tensor,
        queries: torch.Tensor,
        attention_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute aggregated attention scores for each key position.

        Returns
        -------
        agg_attn : Tensor, shape (T,)
            Sum of attention weights across all queries for each key.
        """
        d = K.shape[1]
        inv_sqrt_d = (1.0 / d) ** 0.5

        scores_raw = queries @ K.T  # (n, T)
        scores32 = scores_raw.to(torch.float32) * inv_sqrt_d
        if attention_bias is not None:
            bias32 = torch.broadcast_to(
                attention_bias.to(torch.float32), scores32.shape
            )
            scores32 = scores32 + bias32
        max_scores = scores32.max(dim=1, keepdim=True)[0]
        exp_scores = torch.exp(scores32 - max_scores)
        attn = exp_scores / exp_scores.sum(dim=1, keepdim=True)  # (n, T)
        return attn.sum(dim=0)  # (T,)

    def _find_merging_sets_with_target(
        self,
        adj_sim: torch.Tensor,
        protected_set: set,
        T: int,
        t: int,
    ) -> List[List[int]]:
        """
        Binary search on the cosine similarity threshold to produce ~t merging sets.

        Parameters
        ----------
        adj_sim : Tensor, shape (T-1,)
            Cosine similarity between adjacent key pairs.
        protected_set : set of int
            Indices of protected keys (forced to be singletons).
        T : int
            Total sequence length.
        t : int
            Target number of merging sets (= compacted cache size).

        Returns
        -------
        merging_sets : list of list of int
            Each inner list contains consecutive token indices belonging to one cluster.
        """
        adj_sim_cpu = adj_sim.cpu()

        lo, hi = 0.0, 1.0
        best_sets = None
        best_diff = float('inf')

        for _ in range(self.threshold_search_steps):
            mid = (lo + hi) / 2
            sets = self._greedy_ahc(adj_sim_cpu, protected_set, T, mid)
            n_sets = len(sets)
            diff = abs(n_sets - t)

            if diff < best_diff:
                best_diff = diff
                best_sets = sets

            if n_sets == t:
                break
            elif n_sets < t:
                # Too much merging — increase threshold (less merging)
                lo = mid
            else:
                # Too little merging — decrease threshold (more merging)
                hi = mid

        # If we overshot or undershot, trim or pad to exactly t
        if len(best_sets) > t:
            best_sets = self._trim_sets(best_sets, t)
        elif len(best_sets) < t:
            best_sets = self._split_sets(best_sets, t)

        return best_sets

    @staticmethod
    def _greedy_ahc(
        adj_sim: torch.Tensor,
        protected_set: set,
        T: int,
        threshold: float,
    ) -> List[List[int]]:
        """
        Greedy agglomerative hierarchical clustering (Algorithm 1 from the paper).

        Scans from the last token to the first, grouping consecutive tokens whose
        adjacent cosine similarity exceeds the threshold. Protected tokens always
        start a new cluster.
        """
        sets: List[List[int]] = []
        current_set = [T - 1]

        for i in range(T - 2, -1, -1):
            # Start new set if:
            #  - current token is protected
            #  - next token (i+1) is protected
            #  - similarity between i and i+1 is below threshold
            if (i in protected_set
                    or (i + 1) in protected_set
                    or adj_sim[i].item() < threshold):
                sets.append(current_set)
                current_set = [i]
            else:
                current_set.append(i)

        sets.append(current_set)

        # Reverse each set so indices are ascending, and reverse the list order
        return [sorted(s) for s in reversed(sets)]

    @staticmethod
    def _trim_sets(sets: List[List[int]], t: int) -> List[List[int]]:
        """
        Reduce the number of sets to t by merging the smallest adjacent pairs.
        """
        while len(sets) > t:
            # Find the pair of adjacent sets with smallest combined size
            min_combined = float('inf')
            merge_idx = 0
            for i in range(len(sets) - 1):
                combined = len(sets[i]) + len(sets[i + 1])
                if combined < min_combined:
                    min_combined = combined
                    merge_idx = i
            # Merge sets[merge_idx] and sets[merge_idx + 1]
            merged = sets[merge_idx] + sets[merge_idx + 1]
            sets = sets[:merge_idx] + [merged] + sets[merge_idx + 2:]
        return sets

    @staticmethod
    def _split_sets(sets: List[List[int]], t: int) -> List[List[int]]:
        """
        Increase the number of sets to t by splitting the largest sets.
        """
        while len(sets) < t:
            # Find largest set
            max_size = 0
            split_idx = 0
            for i, s in enumerate(sets):
                if len(s) > max_size:
                    max_size = len(s)
                    split_idx = i
            if max_size <= 1:
                break  # Can't split singletons
            s = sets[split_idx]
            mid = len(s) // 2
            sets = sets[:split_idx] + [s[:mid], s[mid:]] + sets[split_idx + 1:]
        return sets

    def _compute_beta_nnls(
        self,
        C1: torch.Tensor,
        K: torch.Tensor,
        queries: torch.Tensor,
        attention_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        """Fit beta via NNLS to match the original partition function."""
        d = K.shape[1]
        dtype_param = K.dtype
        device = K.device
        inv_sqrt_d = (1.0 / d) ** 0.5

        # Original exp scores
        scores_raw = queries @ K.T
        scores32 = scores_raw.to(torch.float32) * inv_sqrt_d
        if attention_bias is not None:
            bias32 = torch.broadcast_to(
                attention_bias.to(torch.float32), scores32.shape
            )
            scores32 = scores32 + bias32
        max_scores = scores32.max(dim=1, keepdim=True)[0]
        exp_scores_orig = torch.exp(scores32 - max_scores)  # (n, T)

        # Compacted exp scores
        sC_raw = queries @ C1.T
        sC32 = sC_raw.to(torch.float32) * inv_sqrt_d
        # Use same max for numerical consistency
        max_C = sC32.max(dim=1, keepdim=True)[0]
        # Adjust: we need exp(sC - max_orig) to be on the same scale
        # Actually, for NNLS we need: M @ B ≈ target
        # where M[i,j] = exp(q_i · C1_j / sqrt(d) - max_i)
        # and target[i] = sum_j exp(q_i · K_j / sqrt(d) - max_i)
        M = torch.exp(sC32 - max_scores)  # (n, t) — same max as original

        target = exp_scores_orig.sum(dim=1)  # (n,)

        B = self._nnls_pg(M, target, self.nnls_iters,
                          self.nnls_lower_bound, self.nnls_upper_bound)
        beta32 = torch.log(B)  # (t,)
        return beta32.to(dtype_param)
