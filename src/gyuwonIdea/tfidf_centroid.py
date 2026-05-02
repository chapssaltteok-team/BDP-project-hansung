import time
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import LinearSVC

from config import (
    CAT_CLASSES, NUM_CATS, TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE, THETA_LIST_TFIDF, STRATEGY_KEYS, SEED
)
from sampling_strategy import get_sample, measure_stability
from evaluator import run_assignment, evaluate, build_result_record


def build_tfidf(df_train: pd.DataFrame,
                df_all: pd.DataFrame):
    print('  [TF-IDF] Vectorizer fit...')

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
        min_df=2,
    )

    X_train = vectorizer.fit_transform(df_train['text'].fillna(''))
    X_all = vectorizer.transform(df_all['text'].fillna(''))

    print(f'  [TF-IDF] 어휘={len(vectorizer.vocabulary_)}  '
          f'train shape={X_train.shape}  all shape={X_all.shape}')

    return vectorizer, X_all, X_train


def compute_centroids_tfidf(sample_df: pd.DataFrame,
                            X_all: object,
                            df_all: pd.DataFrame) -> np.ndarray:
    dim = X_all.shape[1]
    centroids = np.zeros((NUM_CATS, dim))

    for cat_i, cat in enumerate(CAT_CLASSES):
        mask = sample_df['category'] == cat
        indices = sample_df[mask].index.tolist()

        if len(indices) == 0:
            vec = X_all.mean(axis=0)
            centroids[cat_i] = np.asarray(vec).flatten()
            print(f'    ⚠️  {cat}: 샘플 없음 → 전체 평균 대체')
        else:
            rows = X_all[indices]
            if issparse(rows):
                vec = np.asarray(rows.mean(axis=0)).flatten()
            else:
                vec = rows.mean(axis=0)
            centroids[cat_i] = vec

    return centroids


def compute_similarity_matrix(X_target: object,
                              centroids: np.ndarray) -> np.ndarray:
    return cosine_similarity(X_target, centroids)


def train_tfidf_slow_classifier(X_train, df_train: pd.DataFrame):
    """
    slow path용 supervised classifier.
    기존 centroid argmax보다 정확한 fallback 역할.
    """
    clf = LinearSVC(random_state=SEED)
    clf.fit(X_train, df_train['cat_id'])
    return clf


def predict_slow_classifier(clf, X_test):
    """
    slow classifier 전체 test 예측값과 평균 추론시간(ms/건) 계산.
    실제 routing에서는 slow 대상만 이 비용이 추가된다고 가정한다.
    """
    t0 = time.time()
    preds = clf.predict(X_test)
    elapsed = time.time() - t0
    ms_per = elapsed / X_test.shape[0] * 1000 if X_test.shape[0] else 0.0
    print(f'  [TF-IDF Slow Classifier] 예측 속도={ms_per:.4f}ms/건')
    return preds, ms_per


def run_strategy_tfidf(strategy: str,
                       df_train: pd.DataFrame,
                       df_test: pd.DataFrame,
                       df_all: pd.DataFrame,
                       X_all,
                       X_test,
                       slow_preds,
                       slow_ms_per: float) -> dict:
    print(f'\n  ── 전략: {strategy} ──────────────────────')

    sample = get_sample(df_train, strategy, seed=SEED)
    centroids = compute_centroids_tfidf(sample, X_all, df_all)

    sim_matrix = compute_similarity_matrix(X_test, centroids)

    theta_results = []
    for theta in THETA_LIST_TFIDF:
        assignment = run_assignment(
            df_test,
            sim_matrix,
            theta,
            slow_preds=slow_preds,
            slow_ms_per=slow_ms_per
        )
        result = evaluate(df_test, assignment)
        theta_results.append(result)

    print(f'  ── 안정성 측정 ({strategy}) ──')
    X_all_dense = np.asarray(X_all.todense()) if issparse(X_all) else X_all
    stability = measure_stability(df_all, strategy, X_all_dense)

    return build_result_record('tfidf', strategy, theta_results, stability)


def run_tfidf_experiment(splits: dict) -> list:
    print('\n' + '=' * 60)
    print('  실험 A : TF-IDF Centroid + Slow Classifier')
    print('=' * 60)

    df_train = splits['train'].reset_index(drop=True)
    df_test = splits['test'].reset_index(drop=True)
    df_all = pd.concat(
        [splits['train'], splits['val'], splits['test']],
        ignore_index=True
    ).reset_index(drop=True)

    vectorizer, X_all, X_train = build_tfidf(df_train, df_all)

    n_train_val = len(splits['train']) + len(splits['val'])
    X_test = X_all[n_train_val:]

    print('  [TF-IDF Slow Classifier] LinearSVC 학습...')
    slow_clf = train_tfidf_slow_classifier(X_train, df_train)
    slow_preds, slow_ms_per = predict_slow_classifier(slow_clf, X_test)

    all_results = []
    for strategy in STRATEGY_KEYS:
        result = run_strategy_tfidf(
            strategy,
            df_train,
            df_test,
            df_all,
            X_all,
            X_test,
            slow_preds,
            slow_ms_per
        )
        all_results.append(result)

    print('\n  ✅ TF-IDF 실험 완료')
    return all_results