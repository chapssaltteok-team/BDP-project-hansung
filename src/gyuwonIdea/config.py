"""
config.py
=========
gyuwonIdea 실험 전체에서 공유하는 상수/경로/하이퍼파라미터.

수치를 바꾸고 싶을 때 이 파일 하나만 수정하면
다른 모든 모듈에 자동 반영된다.
모듈 간 의존성 방향 : config ← 모든 모듈 (config 는 아무것도 import 하지 않음)
"""

import os

# ── 경로 ──────────────────────────────────────────────────────────────────────
# 프로젝트 루트(BDP-project-hansung/)에서 실행한다고 가정.
# RunPod Jupyter 터미널에서 `cd /workspace && python src/gyuwonIdea/main.py`
PROC_DIR   = 'data/processed'          # train/val/test.csv 위치
RESULT_DIR = 'results'
FIG_DIR    = os.path.join(RESULT_DIR, 'figures', 'gyuwon', 'idea')
JSON_PATH  = os.path.join(RESULT_DIR, 'gyuwon_idea_scores.json')
SCORES_PATH = os.path.join(RESULT_DIR, 'scores.json')  # 팀 공유 append 대상

os.makedirs(FIG_DIR, exist_ok=True)

# ── 샘플링 ────────────────────────────────────────────────────────────────────
SEED          = 42       # 재현성 고정
SAMPLE_N      = 1000     # centroid 계산에 쓸 초기 샘플 수
STABILITY_K   = 10       # centroid 안정성 측정을 위한 반복 횟수

# ── 임계값 후보 (θ) ───────────────────────────────────────────────────────────
# 낮을수록 fast path 비율 ↑ (속도 ↑, 정확도 ↓ 가능)
# 높을수록 slow path(BERT 재처리) 비율 ↑ (정확도 ↑, 속도 ↓)
THETA_LIST    = [0.3, 0.5, 0.7]

# ── 카테고리 정의 (v2 기준, prepro.py 와 동일 순서) ──────────────────────────
CAT_CLASSES  = ['타격', '투구', '선수이동', '부상', '감독·운영', '기타']
CAT_LABEL2ID = {c: i for i, c in enumerate(CAT_CLASSES)}
NUM_CATS     = len(CAT_CLASSES)   # 6

# ── TF-IDF 설정 ───────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 30_000   # 어휘 사전 크기
TFIDF_NGRAM_RANGE  = (1, 2)   # 단어 + 바이그램

# ── BERT 설정 ─────────────────────────────────────────────────────────────────
BERT_MODEL_NAME = 'klue/bert-base'
BERT_MAX_LEN    = 128     # centroid 계산용 — 전체 fine-tuning 보다 짧아도 충분
BERT_BATCH_SIZE = 32      # 인코딩 전용이라 학습보다 배치 크게 가능

# ── 샘플링 전략 레이블 (반복 처리용 키) ──────────────────────────────────────
STRATEGY_KEYS   = ['sequential', 'random', 'stratified']
STRATEGY_LABELS = {
    'sequential': 'Sequential\n(상위 1000건)',
    'random'    : 'Random\n(무작위 1000건)',
    'stratified': 'Stratified\n(카테고리 비율 샘플링)',
}
