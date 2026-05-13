import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from config import (
    CAT_CLASSES, NUM_CATS, BERT_MODEL_NAME,
    BERT_MAX_LEN, BERT_BATCH_SIZE,
    THETA_LIST_BERT, STRATEGY_KEYS, SEED
)
from sampling_strategy import get_sample, measure_stability
from evaluator import run_assignment, evaluate, build_result_record

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_bert_embeddings(texts: list,
                            tokenizer,
                            model,
                            max_len: int = BERT_MAX_LEN,
                            batch_size: int = BERT_BATCH_SIZE) -> np.ndarray:
    model.eval()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        enc = tokenizer(
            batch_texts,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        input_ids = enc['input_ids'].to(DEVICE)
        attn_mask = enc['attention_mask'].to(DEVICE)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask)

        embeddings = out.pooler_output.detach().cpu().numpy()
        all_embeddings.append(embeddings)

        if (i // batch_size + 1) % 10 == 0:
            print(f'    임베딩 추출 {i + len(batch_texts)}/{len(texts)}건...')

    return np.vstack(all_embeddings)


def compute_centroids_bert(sample_df: pd.DataFrame,
                           emb_all: np.ndarray,
                           df_all: pd.DataFrame) -> np.ndarray:
    centroids = np.zeros((NUM_CATS, emb_all.shape[1]))

    for cat_i, cat in enumerate(CAT_CLASSES):
        mask = sample_df['category'] == cat
        indices = sample_df[mask].index.tolist()

        if len(indices) == 0:
            centroids[cat_i] = emb_all.mean(axis=0)
            print(f'    ⚠️  {cat}: 샘플 없음 → 전체 평균 대체')
        else:
            centroids[cat_i] = emb_all[indices].mean(axis=0)

    return centroids


def train_bert_slow_classifier(emb_train: np.ndarray,
                               df_train: pd.DataFrame):
    """
    slow path용 BERT embedding classifier.
    centroid보다 강한 supervised fallback 역할.
    """
    clf = make_pipeline(
        StandardScaler(),
        LinearSVC(random_state=SEED, max_iter=5000)
    )
    clf.fit(emb_train, df_train['cat_id'])
    return clf


def predict_slow_classifier(clf, emb_test: np.ndarray):
    t0 = time.time()
    preds = clf.predict(emb_test)
    elapsed = time.time() - t0
    ms_per = elapsed / len(emb_test) * 1000 if len(emb_test) else 0.0
    print(f'  [BERT Slow Classifier] 예측 속도={ms_per:.4f}ms/건')
    return preds, ms_per


def run_strategy_bert(strategy: str,
                      df_train: pd.DataFrame,
                      df_test: pd.DataFrame,
                      df_all: pd.DataFrame,
                      emb_all: np.ndarray,
                      emb_test: np.ndarray,
                      slow_preds,
                      slow_ms_per: float) -> dict:
    print(f'\n  ── 전략: {strategy} ──────────────────────')

    sample = get_sample(df_train, strategy, seed=SEED)
    centroids = compute_centroids_bert(sample, emb_all, df_all)

    sim_matrix = cosine_similarity(emb_test, centroids)

    theta_results = []
    for theta in THETA_LIST_BERT:
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
    stability = measure_stability(df_all, strategy, emb_all)

    return build_result_record('bert', strategy, theta_results, stability)


def run_bert_experiment(splits: dict) -> list:
    print('\n' + '=' * 60)
    print('  실험 B : BERT Centroid + Slow Classifier')
    print(f'  Device: {DEVICE}')
    print('=' * 60)

    df_train = splits['train'].reset_index(drop=True)
    df_test = splits['test'].reset_index(drop=True)
    df_all = pd.concat(
        [splits['train'], splits['val'], splits['test']],
        ignore_index=True
    ).reset_index(drop=True)

    print(f'  BERT 모델 로드: {BERT_MODEL_NAME}')
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)

    print(f'  전체 {len(df_all)}건 임베딩 추출 중...')
    texts_all = df_all['text'].fillna('').tolist()
    emb_all = extract_bert_embeddings(texts_all, tokenizer, model)
    print(f'  임베딩 완료: {emb_all.shape}')

    n_train = len(splits['train'])
    n_train_val = len(splits['train']) + len(splits['val'])

    emb_train = emb_all[:n_train]
    emb_test = emb_all[n_train_val:]

    print('  [BERT Slow Classifier] LinearSVC 학습...')
    slow_clf = train_bert_slow_classifier(emb_train, df_train)
    slow_preds, slow_ms_per = predict_slow_classifier(slow_clf, emb_test)

    all_results = []
    for strategy in STRATEGY_KEYS:
        result = run_strategy_bert(
            strategy,
            df_train,
            df_test,
            df_all,
            emb_all,
            emb_test,
            slow_preds,
            slow_ms_per
        )
        all_results.append(result)

    print('\n  ✅ BERT 실험 완료')
    return all_results