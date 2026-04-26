import numpy as np


def find_target_ratio(similarities, min_ratio=0.2, max_ratio=0.8):
    """
    Estimate a compression keep ratio from the sentence-score distribution.

    A concentrated score head means a small number of sentences probably covers the
    answer evidence, so the ratio can be lower. A flatter distribution usually means
    evidence is spread across the context, so the ratio should be higher.
    """
    sims = np.array(similarities, dtype=float)

    if len(sims) == 0:
        return 0.5

    if len(sims) <= 3:
        return max_ratio

    sims = np.sort(sims)[::-1]
    s_min, s_max = sims.min(), sims.max()
    if np.isclose(s_max, s_min):
        return 0.6

    norm_sims = (sims - s_min) / (s_max - s_min + 1e-9)
    high_relevance_count = np.sum(norm_sims >= 0.7)
    high_relevance_ratio = high_relevance_count / len(norm_sims)

    k = max(1, int(len(norm_sims) * 0.2))
    topk_mass = norm_sims[:k].sum() / (norm_sims.sum() + 1e-9)

    first_gap = norm_sims[0] - norm_sims[min(1, len(norm_sims) - 1)]
    avg_gap = np.mean(np.diff(norm_sims[: min(5, len(norm_sims))])) if len(norm_sims) >= 2 else 0
    concentration = 0.5 * topk_mass + 0.3 * first_gap - 0.2 * avg_gap

    raw_ratio = 0.75 - 0.45 * concentration + 0.35 * high_relevance_ratio
    return float(np.clip(raw_ratio, min_ratio, max_ratio))
