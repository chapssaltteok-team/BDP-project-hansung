"""
sampling_strategy.py
====================
세 가지 샘플링 전략 함수 + centroid 안정성 측정.

샘플링 전략은 이 실험의 핵심 독립 변수다.
어떤 1,000건으로 centroid 를 만드느냐가 전체 시스템 품질을 결정한다.

세 전략
────────
Sequential  : 상위 N 건을 순서대로 자른다.
              → 시계열 편향 발생 (2025년 7~8월 집중) — 대조군 역할.
              → 예상: FA 시즌(11~12월), 스프링캠프(2~3월) 표현이 centroid 에
                누락되어 해당 시기 기사가 fast path 를 통과하지 못하고
                slow path(재처리) 비율이 높아질 것.

Random      : 전체에서 무작위 N 건 추출.
              → 시계열 편향 없음.
              → 단점: 소수 카테고리(부상 38건, 기타 41건)가 전체의 0.3~0.4% 수준이므로
                1000건 중 랜덤으로 3~4건만 들어갈 수 있음.
                → 소수 카테고리 centroid 가 불안정해질 가능성 있음.

Stratified  : 카테고리 비율을 맞춰서 추출 (GroupBy + proportional sampling).
              → 소수 카테고리도 최소 n건 보장.
              → 예상: centroid 품질 최고, fast path 통과율 최고.
              → 단점: 카테고리 레이블이 사전에 있어야 함(여기서는 prepro.py 결과 활용).

안정성 측정
────────────
동일 전략으로 N 번 반복 샘플링 후 centroid 벡터의 분산을 계산.
분산이 낮을수록 샘플 의존성이 낮고 안정적인 전략.
→ 이 값 자체가 "어떤 전략이 robust 한가"를 정량화하는 독립 지표.
"""

import numpy as np
import pandas as pd
from config import SEED, SAMPLE_N, STABILITY_K, CAT_CLASSES, NUM_CATS


# ──────────────────────────────────────────────────────────────────────────────
# 1. 샘플링 함수 3종
# ──────────────────────────────────────────────────────────────────────────────

def sample_sequential(df: pd.DataFrame, n: int = SAMPLE_N) -> pd.DataFrame:
    """
    상위 n 건을 인덱스 순서대로 반환.
    prepro.py 가 날짜 오름차순으로 정렬하므로 가장 오래된 기사부터 n 건.
    시계열 편향 실험의 대조군.
    """
    return df.iloc[:n].copy()


def sample_random(df: pd.DataFrame, n: int = SAMPLE_N,
                  seed: int = SEED) -> pd.DataFrame:
    """
    전체에서 무작위 n 건 추출.
    seed 를 고정해 재현성을 보장한다.
    안정성 측정 시에는 seed 를 바꿔 반복 호출한다.
    """
    return df.sample(n=min(n, len(df)), random_state=seed).copy()


def sample_stratified(df: pd.DataFrame, n: int = SAMPLE_N,
                      seed: int = SEED) -> pd.DataFrame:
    """
    카테고리 비율을 맞춰 n 건 추출 (proportional stratified sampling).

    각 카테고리에서 뽑을 수 = round(n × 해당카테고리비율).
    소수 카테고리는 전체 건수가 n_cat < n_alloc 일 수 있으므로
    min(n_alloc, n_cat) 로 상한을 둔다.
    반올림 오차로 합계가 n 과 1~2건 차이날 수 있으므로
    부족분은 가장 큰 카테고리에서 보충한다.

    왜 proportional 인가 : 전체 데이터의 카테고리 분포를 반영한
    centroid 를 만들어야 실제 데이터를 가장 잘 대표한다.
    """
    result_parts = []
    total        = len(df)
    remainder    = n   # 아직 뽑아야 할 남은 수

    # 카테고리별 비율 계산
    cat_counts = df['category'].value_counts()
    alloc      = {}
    for cat in CAT_CLASSES:
        cnt         = cat_counts.get(cat, 0)
        n_alloc     = round(n * cnt / total)
        alloc[cat]  = min(n_alloc, cnt)   # 전체 건수 초과 방지

    # 부족분 보충 : 가장 많은 카테고리에서 추가 추출
    allocated = sum(alloc.values())
    if allocated < n:
        biggest_cat = cat_counts.index[0]
        extra       = min(n - allocated,
                          cat_counts[biggest_cat] - alloc[biggest_cat])
        alloc[biggest_cat] += extra

    # 카테고리별 샘플링
    for cat, k in alloc.items():
        if k <= 0:
            continue
        subset = df[df['category'] == cat]
        sampled = subset.sample(n=k, random_state=seed, replace=False)
        result_parts.append(sampled)

    result = pd.concat(result_parts, ignore_index=True)
    return result.copy()


# ──────────────────────────────────────────────────────────────────────────────
# 2. 전략별 샘플 반환 (공통 진입점)
# ──────────────────────────────────────────────────────────────────────────────

def get_sample(df: pd.DataFrame, strategy: str,
               n: int = SAMPLE_N, seed: int = SEED) -> pd.DataFrame:
    """
    strategy 키에 따라 적절한 샘플링 함수를 호출한다.
    main.py 와 evaluator.py 에서 strategy 를 문자열로 전달해 분기한다.

    strategy : 'sequential' | 'random' | 'stratified'
    """
    if strategy == 'sequential':
        return sample_sequential(df, n)
    elif strategy == 'random':
        return sample_random(df, n, seed)
    elif strategy == 'stratified':
        return sample_stratified(df, n, seed)
    else:
        raise ValueError(f'알 수 없는 전략: {strategy}')


# ──────────────────────────────────────────────────────────────────────────────
# 3. Centroid 안정성 측정
# ──────────────────────────────────────────────────────────────────────────────

def measure_stability(df: pd.DataFrame, strategy: str,
                      vec_matrix: np.ndarray,
                      k: int = STABILITY_K,
                      n: int = SAMPLE_N) -> dict:
    """
    동일 전략으로 k 회 반복 샘플링 후 카테고리별 centroid 분산을 계산.

    vec_matrix : (전체 데이터 수, 벡터 차원) numpy 배열.
                 TF-IDF 방식에서는 TF-IDF 행렬,
                 BERT 방식에서는 BERT 임베딩 행렬.

    centroid 분산 계산 방법
    ──────────────────────
    1. k 번 샘플링 → 각 회차마다 카테고리별 centroid 계산 → (k, NUM_CATS, dim) 텐서
    2. 카테고리별로 k 개 centroid 의 분산(variance)을 평균냄
       → 스칼라 하나가 해당 카테고리의 "불안정도"
    3. 전체 카테고리 평균 분산 → 전략 전체의 안정성 지표

    값이 낮을수록 샘플 의존성이 낮고 안정적인 전략.

    반환
    ────
    {
      'strategy'        : 전략 이름,
      'per_cat_var'     : {카테고리: 평균분산} — 카테고리별 불안정도,
      'mean_var'        : 전체 평균 분산 — 전략 전체 안정성 지표,
      'centroids_all'   : (k, NUM_CATS, dim) — 반복별 centroid (시각화용),
    }
    """
    n_samples = len(df)
    dim       = vec_matrix.shape[1]

    # k 회 반복 centroid 수집
    centroids_all = np.zeros((k, NUM_CATS, dim))   # (k, 6, dim)

    for trial in range(k):
        seed_t  = SEED + trial   # 매 회차 다른 seed
        sample  = get_sample(df, strategy, n=n, seed=seed_t)

        # 샘플 인덱스로 vec_matrix 에서 행 추출
        # df.index 와 vec_matrix 행 순서가 일치한다고 가정
        # (data_loader 에서 reset_index 를 했으므로 일치함)
        idx = sample.index.tolist()

        for cat_i, cat in enumerate(CAT_CLASSES):
            mask     = sample['category'] == cat
            cat_idx  = [idx[j] for j in range(len(idx)) if mask.iloc[j]]
            if len(cat_idx) == 0:
                # 해당 카테고리 샘플이 없으면 전체 평균으로 대체
                centroids_all[trial, cat_i] = vec_matrix.mean(axis=0)
            else:
                centroids_all[trial, cat_i] = vec_matrix[cat_idx].mean(axis=0)

    # 카테고리별 분산 계산
    # centroids_all[:, cat_i, :] : k 개 centroid 벡터 → 차원별 분산 → 평균
    per_cat_var = {}
    for cat_i, cat in enumerate(CAT_CLASSES):
        var = centroids_all[:, cat_i, :].var(axis=0).mean()  # 스칼라
        per_cat_var[cat] = float(var)

    mean_var = float(np.mean(list(per_cat_var.values())))

    print(f'  [{strategy}] 안정성 평균분산={mean_var:.6f}  '
          + '  '.join(f'{c}:{v:.5f}' for c, v in per_cat_var.items()))

    return {
        'strategy'     : strategy,
        'per_cat_var'  : per_cat_var,
        'mean_var'     : mean_var,
        'centroids_all': centroids_all,
    }
