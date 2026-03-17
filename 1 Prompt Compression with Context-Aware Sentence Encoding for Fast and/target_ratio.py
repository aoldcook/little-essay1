import numpy as np


def find_target_ratio(similarities, min_ratio=0.2, max_ratio=0.8):
    """
    根据相似度分布自适应决定压缩比例。

    思路：
    1. 相似度越“头部集中”，说明少量句子就够，ratio 越小
    2. 相似度越“分散”，说明需要保留更多上下文，ratio 越大
    3. 比单纯 elbow 更稳，尤其适合句子数不多的情况
    """
    sims = np.array(similarities, dtype=float)

    if len(sims) == 0:
        return 0.5

    if len(sims) <= 3:
        return max_ratio

    # 降序
    sims = np.sort(sims)[::-1]

    # 归一化到 0~1
    s_min, s_max = sims.min(), sims.max()
    if np.isclose(s_max, s_min):
        return 0.6

    norm_sims = (sims - s_min) / (s_max - s_min + 1e-9)

    # 头部句子占比：有多少句子“明显相关”
    high_relevance_count = np.sum(norm_sims >= 0.7)
    high_relevance_ratio = high_relevance_count / len(norm_sims)

    # 分布集中度：top-k 累积贡献
    k = max(1, int(len(norm_sims) * 0.2))   # 前20%
    topk_mass = norm_sims[:k].sum() / (norm_sims.sum() + 1e-9)

    # gap 越大，说明前几句越关键
    first_gap = norm_sims[0] - norm_sims[min(1, len(norm_sims) - 1)]
    avg_gap = np.mean(np.diff(norm_sims[:min(5, len(norm_sims))])) if len(norm_sims) >= 2 else 0
    concentration = 0.5 * topk_mass + 0.3 * first_gap - 0.2 * avg_gap

    # 核心映射：
    # 高集中 -> 小 ratio
    # 高相关句较多 -> 大 ratio
    raw_ratio = 0.75 - 0.45 * concentration + 0.35 * high_relevance_ratio

    ratio = float(np.clip(raw_ratio, min_ratio, max_ratio))
    return ratio
