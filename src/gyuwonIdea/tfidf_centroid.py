"""
tfidf_centroid.py
=================
실험 A : TF-IDF 기반 centroid 계산 + fast/slow path 카테고리 배정.

CPU 만으로 실행 가능 — 로컬 환경에서도 전체 실험 완료 가능.

동작 흐름
──────────
1. 전체 데이터(train+test)를 TF-IDF 벡터화 (fit : train, transform : 전체)
2. 샘플링 전략(sequential/random/stratified)으로 1,000건 추출
3. 추출된 샘플로 카테고리별 centroid(TF-IDF 평균 벡터) 계산
4. 전체 데이터와 centroid 간 코사인 유사도 계산 → (N, NUM_CATS) 행렬
5. θ 별로 fast/slow path 분류 + 정확도/속도 평가
6. centroid 안정성 측정 (k=10 회 반복)

TF-IDF 와 코사인 유사도를 선택한 이유
───────────────────────────────────────
- TF-IDF 는 L04/L06 수업 범위 내 기초 방법론 → 발표 설명이 용이
- 희소 행렬(sparse matrix)이라 메모리 효율적 (11,531건 × 30,000 어휘)
- sklearn.metrics.pairwise.cosine_similarity 로 벡터화 연산 가능
  → 건당 반복문 없이 행렬 연산 → 속도 측정에 유리
- BERT 임베딩 대비 품질 열위가 예상되지만 "얼마나 열위인가"가
  bert_centroid.py 와의 비교에서 핵심 발견이 된다.
"""

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (CAT_CLASSES, NUM_CATS, TFIDF_MAX_FEATURES,
                    TFIDF_NGRAM_RANGE, THETA_LIST, STRATEGY_KEYS, SEED)
from sampling_strategy import get_sample, measure_stability
from evaluator import run_assignment, evaluate, build_result_record


# ──────────────────────────────────────────────────────────────────────────────
# 1. TF-IDF 벡터화
# ──────────────────────────────────────────────────────────────────────────────

def build_tfidf(df_train: pd.DataFrame,
                df_all: pd.DataFrame):
    """
    TF-IDF Vectorizer 를 train 데이터로 fit 하고 전체 데이터를 transform.

    fit 을 train 에서만 하는 이유 :
      test 데이터의 어휘가 학습에 영향을 주는 data leakage 를 방지한다.
      실제 서비스 환경에서는 새 기사(미래 데이터)의 어휘를 미리 알 수 없으므로
      train 어휘 기반 vectorizer 가 더 현실적이다.

    반환
    ────
    vectorizer : 학습된 TfidfVectorizer (centroid 계산에도 재사용)
    X_all      : (len(df_all), max_features) sparse matrix — 전체 벡터
    X_train    : (len(df_train), max_features) sparse matrix — train 벡터
    """
    print('  [TF-IDF] Vectorizer fit...')
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,    # TF 에 log(1+TF) 적용 → 고빈도 단어 과대평가 완화
        min_df=2,             # 2회 미만 등장 단어 제거 → 노이즈 감소
    )
    X_train = vectorizer.fit_transform(df_train['text'].fillna(''))
    X_all   = vectorizer.transform(df_all['text'].fillna(''))
    print(f'  [TF-IDF] 어휘={len(vectorizer.vocabulary_)}  '
          f'train shape={X_train.shape}  all shape={X_all.shape}')
    return vectorizer, X_all, X_train


# ──────────────────────────────────────────────────────────────────────────────
# 2. Centroid 계산
# ──────────────────────────────────────────────────────────────────────────────

def compute_centroids_tfidf(sample_df: pd.DataFrame,
                             X_all: object,
                             df_all: pd.DataFrame) -> np.ndarray:
    """
    샘플(sample_df)의 TF-IDF 벡터로 카테고리별 centroid 계산.

    sample_df 의 인덱스가 df_all(전체) 의 인덱스와 일치한다고 가정.
    (data_loader 에서 reset_index 완료되어 있음)

    centroid = 해당 카테고리 샘플 벡터들의 산술 평균 벡터.
    카테고리에 속하는 샘플이 없으면 전체 평균으로 대체
    (소수 카테고리 + sequential 샘플링 조합에서 발생 가능).

    반환 : (NUM_CATS, max_features) dense numpy 배열
    """
    dim        = X_all.shape[1]
    centroids  = np.zeros((NUM_CATS, dim))

    for cat_i, cat in enumerate(CAT_CLASSES):
        mask    = sample_df['category'] == cat
        indices = sample_df[mask].index.tolist()

        if len(indices) == 0:
            # 해당 카테고리 샘플 없음 → 전체 평균으로 대체
            vec = X_all.mean(axis=0)
            centroids[cat_i] = np.asarray(vec).flatten()
            print(f'    ⚠️  {cat}: 샘플 없음 → 전체 평균 대체')
        else:
            # sparse 행 추출 후 평균
            rows = X_all[indices]
            if issparse(rows):
                vec = np.asarray(rows.mean(axis=0)).flatten()
            else:
                vec = rows.mean(axis=0)
            centroids[cat_i] = vec

    return centroids   # (NUM_CATS, dim)


# ──────────────────────────────────────────────────────────────────────────────
# 3. 유사도 행렬 계산
# ──────────────────────────────────────────────────────────────────────────────

def compute_similarity_matrix(X_target: object,
                               centroids: np.ndarray) -> np.ndarray:
    """
    X_target(대상 기사 벡터) 과 centroids(카테고리 중심) 간 코사인 유사도 계산.

    cosine_similarity(A, B) : A(n, dim) × B(m, dim) → (n, m) 유사도 행렬.
    각 행 i = i번째 기사와 모든 카테고리 centroid 의 유사도 벡터.

    반환 : (len(X_target), NUM_CATS) dense numpy 배열
    """
    # centroids 는 dense, X_target 은 sparse 일 수 있음 → cosine_similarity 자동 처리
    sim = cosine_similarity(X_target, centroids)  # (N, NUM_CATS)
    return sim


# ──────────────────────────────────────────────────────────────────────────────
# 4. 단일 전략 실행
# ──────────────────────────────────────────────────────────────────────────────

def run_strategy_tfidf(strategy: str,
                       df_train: pd.DataFrame,
                       df_test: pd.DataFrame,
                       df_all: pd.DataFrame,
                       X_all: np.ndarray,
                       X_test: np.ndarray) -> dict:
    """
    단일 전략(strategy)에 대해 θ 전체를 순회하며 평가 결과 수집.

    df_train : centroid 계산용 샘플링 모집단
    df_test  : 평가 대상
    df_all   : 전체 데이터 (안정성 측정용 모집단)
    X_all    : 전체 TF-IDF 행렬 (centroid 계산에 인덱싱)
    X_test   : test TF-IDF 행렬 (유사도 계산 대상)

    반환 : build_result_record() 형식 dict
    """
    print(f'\n  ── 전략: {strategy} ──────────────────────')

    # 1,000건 샘플링 후 centroid 계산
    sample      = get_sample(df_train, strategy, seed=SEED)
    centroids   = compute_centroids_tfidf(sample, X_all, df_all)

    # 유사도 행렬 : test 기사 × 카테고리 centroid
    sim_matrix  = compute_similarity_matrix(X_test, centroids)

    # θ 별 배정 + 평가
    theta_results = []
    for theta in THETA_LIST:
        assignment = run_assignment(df_test, sim_matrix, theta)
        result     = evaluate(df_test, assignment)
        theta_results.append(result)

    # centroid 안정성 측정 (train 전체 모집단, TF-IDF 벡터)
    # X_all 이 sparse 이므로 dense 변환 후 전달
    print(f'  ── 안정성 측정 ({strategy}) ──')
    X_all_dense = np.asarray(X_all.todense()) if issparse(X_all) else X_all
    stability   = measure_stability(df_all, strategy, X_all_dense)

    return build_result_record('tfidf', strategy, theta_results, stability)


# ──────────────────────────────────────────────────────────────────────────────
# 5. 전체 TF-IDF 실험 실행 (main.py 에서 호출)
# ──────────────────────────────────────────────────────────────────────────────

def run_tfidf_experiment(splits: dict) -> list:
    """
    모든 전략(sequential/random/stratified) × 모든 θ 조합 실험 수행.

    splits : data_loader.load_splits() 반환값

    반환 : [build_result_record(), ...] — 전략 수만큼의 결과 리스트
    """
    print('\n' + '='*60)
    print('  실험 A : TF-IDF Centroid 기반 카테고리 배정')
    print('='*60)

    df_train = splits['train'].reset_index(drop=True)
    df_test  = splits['test'].reset_index(drop=True)
    df_all   = pd.concat([splits['train'], splits['val'], splits['test']],
                          ignore_index=True).reset_index(drop=True)

    # TF-IDF 벡터화 (train fit, 전체 transform)
    vectorizer, X_all, X_train = build_tfidf(df_train, df_all)

    # test 벡터 추출 (df_test 인덱스 기준 → df_all 에서 해당 행 추출)
    # df_all 의 앞 len(train)+len(val) 행이 train+val, 나머지가 test
    n_train_val = len(splits['train']) + len(splits['val'])
    X_test      = X_all[n_train_val:]   # test 부분만

    all_results = []
    for strategy in STRATEGY_KEYS:
        result = run_strategy_tfidf(
            strategy, df_train, df_test, df_all, X_all, X_test)
        all_results.append(result)

    print('\n  ✅ TF-IDF 실험 완료')
    return all_results
