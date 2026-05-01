"""
bert_centroid.py
================
실험 B : BERT 임베딩 기반 centroid 계산 + fast/slow path 카테고리 배정.

GPU 권장 (RTX 3090) — CPU 에서도 동작하나 임베딩 추출에 수 시간 소요.
main.py 에서 GPU 없으면 자동 스킵 처리됨.

TF-IDF 방식(실험 A)과 동일한 흐름이지만 벡터 품질이 다르다.
  TF-IDF : 단어 빈도 기반 희소 벡터 (어휘 × 빈도 카운트)
  BERT   : 문맥 의미 기반 밀집 벡터 (768dim, 문장 전체 의미 압축)

두 방식의 차이가 centroid 품질 → fast path 정확도에 어떻게 반영되는지가
이 파일과 tfidf_centroid.py 를 비교하는 핵심 발견이다.

예상 가설
─────────
BERT centroid 가 TF-IDF centroid 보다 의미적으로 정교하므로
  → 동일 θ 에서 fast path Macro-F1 이 더 높을 것
  → 단, 임베딩 추출 비용이 크므로 "초기 1,000건 BERT 임베딩" 자체에
    시간이 소요됨 → 전체 시스템 속도 이점이 줄어들 수 있음
  → 이 트레이드오프가 보고서의 핵심 분석 포인트
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

from config import (CAT_CLASSES, NUM_CATS, BERT_MODEL_NAME,
                    BERT_MAX_LEN, BERT_BATCH_SIZE,
                    THETA_LIST, STRATEGY_KEYS, SEED)
from sampling_strategy import get_sample, measure_stability
from evaluator import run_assignment, evaluate, build_result_record

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ──────────────────────────────────────────────────────────────────────────────
# 1. BERT 임베딩 추출
# ──────────────────────────────────────────────────────────────────────────────

def extract_bert_embeddings(texts: list,
                             tokenizer, model,
                             max_len: int = BERT_MAX_LEN,
                             batch_size: int = BERT_BATCH_SIZE) -> np.ndarray:
    """
    텍스트 리스트를 BERT 로 인코딩해 pooler_output(768dim) 반환.

    pooler_output 을 쓰는 이유 :
      [CLS] 토큰의 hidden state 를 Linear+Tanh 로 변환한 벡터.
      BERT 가 문장 분류 태스크에서 사용하도록 사전학습한 표현이라
      카테고리 유사도 비교에 가장 적합하다.
      mean pooling(모든 토큰 평균) 대비 분류 목적 특화.

    배치 처리 이유 :
      전체 데이터(11,531건)를 한 번에 GPU 에 올리면 메모리 초과.
      batch_size=32 로 나눠 처리하고 결과를 누적.

    반환 : (len(texts), 768) numpy 배열
    """
    model.eval()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        enc = tokenizer(
            batch_texts,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids   = enc['input_ids'].to(DEVICE)
        attn_mask   = enc['attention_mask'].to(DEVICE)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask)
        # pooler_output : (batch, 768)
        embeddings = out.pooler_output.cpu().numpy()
        all_embeddings.append(embeddings)

        if (i // batch_size + 1) % 10 == 0:
            print(f'    임베딩 추출 {i + len(batch_texts)}/{len(texts)}건...')

    return np.vstack(all_embeddings)   # (N, 768)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Centroid 계산 (BERT 버전)
# ──────────────────────────────────────────────────────────────────────────────

def compute_centroids_bert(sample_df: pd.DataFrame,
                            emb_all: np.ndarray,
                            df_all: pd.DataFrame) -> np.ndarray:
    """
    샘플(sample_df) 의 BERT 임베딩으로 카테고리별 centroid 계산.

    tfidf_centroid.py 의 compute_centroids_tfidf() 와 동일한 인터페이스.
    emb_all : (len(df_all), 768) — 전체 데이터 임베딩 행렬
    sample_df 의 인덱스로 emb_all 에서 행 추출 후 평균.

    반환 : (NUM_CATS, 768) numpy 배열
    """
    centroids = np.zeros((NUM_CATS, emb_all.shape[1]))

    for cat_i, cat in enumerate(CAT_CLASSES):
        mask    = sample_df['category'] == cat
        indices = sample_df[mask].index.tolist()

        if len(indices) == 0:
            centroids[cat_i] = emb_all.mean(axis=0)
            print(f'    ⚠️  {cat}: 샘플 없음 → 전체 평균 대체')
        else:
            centroids[cat_i] = emb_all[indices].mean(axis=0)

    return centroids   # (NUM_CATS, 768)


# ──────────────────────────────────────────────────────────────────────────────
# 3. 단일 전략 실행
# ──────────────────────────────────────────────────────────────────────────────

def run_strategy_bert(strategy: str,
                      df_train: pd.DataFrame,
                      df_test: pd.DataFrame,
                      df_all: pd.DataFrame,
                      emb_all: np.ndarray,
                      emb_test: np.ndarray) -> dict:
    """
    tfidf_centroid.run_strategy_tfidf() 와 동일한 구조.
    벡터만 TF-IDF → BERT 로 교체됨.
    """
    print(f'\n  ── 전략: {strategy} ──────────────────────')

    sample    = get_sample(df_train, strategy, seed=SEED)
    centroids = compute_centroids_bert(sample, emb_all, df_all)

    # 유사도 : test 임베딩 × centroid
    sim_matrix = cosine_similarity(emb_test, centroids)  # (N_test, NUM_CATS)

    theta_results = []
    for theta in THETA_LIST:
        assignment = run_assignment(df_test, sim_matrix, theta)
        result     = evaluate(df_test, assignment)
        theta_results.append(result)

    # 안정성 측정 (BERT 임베딩 기반)
    print(f'  ── 안정성 측정 ({strategy}) ──')
    stability = measure_stability(df_all, strategy, emb_all)

    return build_result_record('bert', strategy, theta_results, stability)


# ──────────────────────────────────────────────────────────────────────────────
# 4. 전체 BERT 실험 실행 (main.py 에서 호출)
# ──────────────────────────────────────────────────────────────────────────────

def run_bert_experiment(splits: dict) -> list:
    """
    모든 전략 × 모든 θ 조합 실험 수행 (BERT 버전).

    핵심 병목 : 전체 데이터 BERT 임베딩 추출.
      11,531건 × batch_size=32 → 약 360 배치.
      RTX 3090 기준 약 5~10분 소요.
      이 시간은 "BERT centroid 방식의 초기 비용"으로 보고서에 기록한다.
    """
    print('\n' + '='*60)
    print('  실험 B : BERT Centroid 기반 카테고리 배정')
    print(f'  Device: {DEVICE}')
    print('='*60)

    df_train = splits['train'].reset_index(drop=True)
    df_test  = splits['test'].reset_index(drop=True)
    df_all   = pd.concat([splits['train'], splits['val'], splits['test']],
                          ignore_index=True).reset_index(drop=True)

    # BERT 모델 로드 (fine-tuning 아님 — 사전학습 가중치 그대로 사용)
    # 이유 : centroid 계산은 분류가 아니라 임베딩 추출이 목적이므로
    #        fine-tuning 없이 사전학습 표현만으로도 의미 있는 유사도 계산 가능.
    #        fine-tuning 모델이 있으면 더 좋지만 독립 실행을 위해 사전학습 사용.
    print(f'  BERT 모델 로드: {BERT_MODEL_NAME}')
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model     = AutoModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)

    # 전체 데이터 임베딩 추출 (가장 시간이 오래 걸리는 단계)
    print(f'  전체 {len(df_all)}건 임베딩 추출 중...')
    texts_all = df_all['text'].fillna('').tolist()
    emb_all   = extract_bert_embeddings(texts_all, tokenizer, model)
    print(f'  임베딩 완료: {emb_all.shape}')

    # test 임베딩 분리
    n_train_val = len(splits['train']) + len(splits['val'])
    emb_test    = emb_all[n_train_val:]

    all_results = []
    for strategy in STRATEGY_KEYS:
        result = run_strategy_bert(
            strategy, df_train, df_test, df_all, emb_all, emb_test)
        all_results.append(result)

    print('\n  ✅ BERT 실험 완료')
    return all_results
