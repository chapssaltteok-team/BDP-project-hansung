import os

PROC_DIR   = 'data/processed'
RESULT_DIR = 'results'
FIG_DIR    = os.path.join(RESULT_DIR, 'figures', 'gyuwon', 'idea')
JSON_PATH  = os.path.join(RESULT_DIR, 'gyuwon_idea_scores.json')
SCORES_PATH = os.path.join(RESULT_DIR, 'scores.json')

os.makedirs(FIG_DIR, exist_ok=True)

SEED          = 42
SAMPLE_N      = 1000
STABILITY_K   = 10

# ✅ 핵심 수정
THETA_LIST_TFIDF = [0.10, 0.15, 0.20, 0.25, 0.30]
THETA_LIST_BERT  = [0.50, 0.60, 0.70, 0.80, 0.90]

# ⚠️ 기존 코드 호환용 (절대 삭제하지 말 것)
THETA_LIST = THETA_LIST_TFIDF

CAT_CLASSES  = ['타격', '투구', '선수이동', '부상', '감독·운영', '기타']
CAT_LABEL2ID = {c: i for i, c in enumerate(CAT_CLASSES)}
NUM_CATS     = len(CAT_CLASSES)

TFIDF_MAX_FEATURES = 30_000
TFIDF_NGRAM_RANGE  = (1, 2)

BERT_MODEL_NAME = 'klue/bert-base'
BERT_MAX_LEN    = 128
BERT_BATCH_SIZE = 32

STRATEGY_KEYS   = ['sequential', 'random', 'stratified']
STRATEGY_LABELS = {
    'sequential': 'Sequential\n(상위 1000건)',
    'random'    : 'Random\n(무작위 1000건)',
    'stratified': 'Stratified\n(카테고리 비율 샘플링)',
}