"""
gyuwon_catFeature.py
====================
실험 ② : 카테고리 임베딩을 감성 분류의 보조 특징으로 추가 (CAT-02)
담당   : 김규원

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
연구 목적 및 가설
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"기사의 카테고리(투구/타격/부상 등)를 BERT 감성 분류기에 추가 정보로
주입하면 감성 판단 정확도가 오르는가?"

직관적 근거 :
  - 동일한 '부진' 표현이라도 투구 기사에서는 방어율 상승과 연결되고,
    타격 기사에서는 타율 하락과 연결된다.
  - 모델이 카테고리 맥락을 알면 동음이의어·중의적 표현의 감성 방향을
    더 정확히 추론할 수 있다는 가설.

실험 설계 (A/B 비교) :
  - 모델 A (기준선) : BERT 풀링 벡터(768dim) → 감성 분류
    → 기존 bert_sentiment.py 와 동일 구조, Macro-F1 0.7404 가 기준
  - 모델 B (제안)   : BERT 풀링 벡터(768dim) + 카테고리 임베딩(32dim)
                      → concat(800dim) → 감성 분류
  두 모델을 동일 하이퍼파라미터로 학습해 Macro-F1 차이를 측정한다.
  차이가 +0.01 이상이면 "카테고리가 감성 맥락 정보로 유효하다"는 결론.
  차이가 미미하거나 음수이면 "BERT 가 이미 카테고리 맥락을 내재화"한
  반증 자료로 분석에 활용한다. (어떤 결과든 보고서 서사가 성립함)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
아키텍처 상세
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [모델 A — 기준선]
  text → BERT Encoder → [CLS] pooler_output(768) → Linear(768,3) → 감성

  [모델 B — 카테고리 보조]
  text → BERT Encoder → [CLS] pooler_output(768) ─┐
                                                    concat(800) → Linear(800,3) → 감성
  category_id → Embedding(6, 32) ─────────────────┘

  카테고리 임베딩 차원(32)은 BERT 출력(768)의 약 4% 수준으로 설정.
  너무 크면 카테고리 신호가 텍스트 신호를 압도하고,
  너무 작으면 정보가 손실된다. 32dim 은 실험적으로 균형점.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실행 환경
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - RunPod RTX 3090 권장
  - 사전 조건 : data/processed/train/val/test.csv
                (sentiment_str, category 컬럼 모두 존재)
  - 실행 : python src/gyuwon_catFeature.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
출력 결과물
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  results/figures/gyuwon/catfeat_01_ab_comparison.png   A vs B 성능 비교
  results/figures/gyuwon/catfeat_02_sentiment_f1.png    감성 클래스별 F1 비교
  results/figures/gyuwon/catfeat_03_cat_impact.png      카테고리별 감성 정확도 히트맵
  results/figures/gyuwon/catfeat_04_embed_tsne.png      카테고리 임베딩 t-SNE 시각화
  results/gyuwon_catfeat_scores.json                    이번 실험 수치
  results/scores.json                                   기존 scores.json 에 append
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0. 임포트
# ──────────────────────────────────────────────────────────────────────────────
import os, json, time, warnings, re
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, accuracy_score,
                              f1_score, confusion_matrix)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import platform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, BertModel,
                          get_cosine_schedule_with_warmup)
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings('ignore')

# ── 한국어 폰트 ───────────────────────────────────────────────────────────────
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# ──────────────────────────────────────────────────────────────────────────────
# 1. 상수 및 하이퍼파라미터
# ──────────────────────────────────────────────────────────────────────────────
PROC_DIR   = 'data/processed'
RESULT_DIR = 'results'
FIG_DIR    = os.path.join(RESULT_DIR, 'figures', 'gyuwon')
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_NAME   = 'klue/bert-base'
MAX_LEN      = 256
BATCH_SIZE   = 16
EPOCHS       = 3
LR           = 2e-5
WARMUP_RATIO = 0.1
CAT_EMB_DIM  = 32      # 카테고리 임베딩 차원 (섹션 4 참고)
SEED         = 42
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f'[환경] Device={DEVICE}  Model={MODEL_NAME}  CAT_EMB_DIM={CAT_EMB_DIM}')

# ── 레이블 인코딩 ──────────────────────────────────────────────────────────────
# 감성 3클래스 : prepro.py 와 동일 순서 유지
SENT_LABEL2ID = {'긍정': 0, '중립': 1, '부정': 2}
SENT_ID2LABEL = {v: k for k, v in SENT_LABEL2ID.items()}
SENT_CLASSES  = ['긍정', '중립', '부정']

# 카테고리 6클래스 (v2 기준) — 반드시 prepro.py 의 CATEGORY_KEYWORDS_V2 키 순서와 일치
# '기타' 를 마지막에 추가해 총 6개
CAT_CLASSES  = ['타격', '투구', '선수이동', '부상', '감독·운영', '기타']
CAT_LABEL2ID = {c: i for i, c in enumerate(CAT_CLASSES)}
NUM_CATS     = len(CAT_CLASSES)   # 6

# ──────────────────────────────────────────────────────────────────────────────
# 2. 데이터 로드
# ──────────────────────────────────────────────────────────────────────────────
def load_splits() -> dict:
    """
    train/val/test.csv 를 읽어 반환한다.
    필수 컬럼 : text(또는 title_clean+body_clean), sentiment_str, category(v2)
    category 컬럼의 값이 CAT_CLASSES 에 없으면 '기타' 로 대체한다.
    """
    splits = {}
    for split in ['train', 'val', 'test']:
        path = os.path.join(PROC_DIR, f'{split}.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} 없음 — prepro.py 먼저 실행')
        df = pd.read_csv(path, encoding='utf-8-sig')

        # text 컬럼 보장
        if 'text' not in df.columns:
            df['text'] = (df['title_clean'].fillna('') + ' '
                          + df['body_clean'].fillna(''))

        # 카테고리 값 정제 : CAT_CLASSES 에 없는 값 → '기타'
        df['category'] = df['category'].apply(
            lambda c: c if c in CAT_LABEL2ID else '기타')

        splits[split] = df
        dist = df['sentiment_str'].value_counts().to_dict()
        print(f'  [{split}] {len(df)}건  감성={dist}')

    return splits

# ──────────────────────────────────────────────────────────────────────────────
# 3. Dataset
# ──────────────────────────────────────────────────────────────────────────────
class SentimentDataset(Dataset):
    """
    모델 A 와 B 를 모두 지원하는 Dataset.
    use_cat=True 일 때 category_id 를 추가로 반환한다.

    __getitem__ 반환값 :
      - input_ids      : (MAX_LEN,)  BERT 토큰 인덱스
      - attention_mask : (MAX_LEN,)  패딩 위치 마스크 (1=실제 토큰, 0=패딩)
      - sentiment_id   : scalar      감성 레이블 (0/1/2)
      - category_id    : scalar      카테고리 레이블 (0~5), use_cat=False 이면 -1
    """
    def __init__(self, df: pd.DataFrame, tokenizer,
                 max_len: int, use_cat: bool = False):
        self.texts       = df['text'].fillna('').tolist()
        self.sentiments  = (df['sentiment_str']
                            .map(SENT_LABEL2ID).fillna(1).astype(int).tolist())
        self.categories  = (df['category']
                            .map(CAT_LABEL2ID).fillna(5).astype(int).tolist()
                            if use_cat else [-1] * len(df))
        self.tokenizer   = tokenizer
        self.max_len     = max_len
        self.use_cat     = use_cat

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids'     : enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'sentiment_id'  : torch.tensor(self.sentiments[idx], dtype=torch.long),
            'category_id'   : torch.tensor(self.categories[idx], dtype=torch.long),
        }

# ──────────────────────────────────────────────────────────────────────────────
# 4. 모델 정의
# ──────────────────────────────────────────────────────────────────────────────

class ModelA_Baseline(nn.Module):
    """
    모델 A : 기존 BERT 감성 분류기 (기준선)

    구조 :
      BERT Encoder → pooler_output(768) → Dropout → Linear(768, 3)

    AutoModelForSequenceClassification 을 쓰지 않고 BertModel 을 직접
    사용하는 이유 : 모델 B 와 동일한 베이스 위에서 head 만 다르게 구성해야
    공정한 비교가 가능하기 때문이다.
    pooler_output 은 BERT 의 [CLS] 토큰을 추가 Linear+Tanh 로 변환한
    768dim 벡터로, 문장 전체의 의미를 압축한 표현이다.
    """
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.bert      = BertModel.from_pretrained(model_name)
        self.dropout   = nn.Dropout(dropout)
        # 분류 head : 768 → 3 (긍정/중립/부정)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask, **kwargs):
        # kwargs 로 category_id 를 받아도 무시 → A/B 를 동일 DataLoader 로 처리 가능
        out       = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled    = self.dropout(out.pooler_output)   # (B, 768)
        logits    = self.classifier(pooled)            # (B, 3)
        return logits


class ModelB_CatFeature(nn.Module):
    """
    모델 B : 카테고리 임베딩 보조 특징 추가

    구조 :
      BERT Encoder → pooler_output(768) ──────────────────────┐
                                                               concat → (800)
      category_id → Embedding(NUM_CATS, CAT_EMB_DIM=32) ─────┘
                                                    ↓
                                        Dropout → Linear(800, 3)

    핵심 설계 결정 :
      1) nn.Embedding(6, 32) : 카테고리 6개 각각을 32차원 밀집 벡터로 표현.
         one-hot(6dim) 대비 장점 — 카테고리 간 유사도를 학습할 수 있다.
         예를 들어 '투구'와 '타격' 임베딩이 '부상'보다 서로 가깝게 학습될 수 있다.

      2) concat 후 Linear : 두 표현을 단순히 이어붙인 뒤 분류 head 에 통과.
         Attention 기반 fusion 보다 단순하지만 이 실험의 목적은
         "카테고리 정보 자체의 유효성 측정"이므로 구조 복잡도는 최소화.

      3) 임베딩 차원 32 : BERT 출력(768)의 약 4%. 비율이 너무 크면
         카테고리 신호가 텍스트 신호를 압도해 BERT 의 문맥 표현이 희석됨.
         너무 작으면(예: 4dim) 카테고리 정보가 사실상 무의미해짐.
    """
    def __init__(self, model_name: str, num_cats: int,
                 cat_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.bert      = BertModel.from_pretrained(model_name)
        self.cat_embed = nn.Embedding(num_cats, cat_emb_dim)
        self.dropout   = nn.Dropout(dropout)
        # concat 후 차원 = 768 + cat_emb_dim
        fused_dim      = self.bert.config.hidden_size + cat_emb_dim
        self.classifier = nn.Linear(fused_dim, 3)

    def forward(self, input_ids, attention_mask, category_id, **kwargs):
        out       = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled    = out.pooler_output               # (B, 768)
        cat_vec   = self.cat_embed(category_id)     # (B, 32)
        fused     = torch.cat([pooled, cat_vec], dim=-1)  # (B, 800)
        fused     = self.dropout(fused)
        logits    = self.classifier(fused)           # (B, 3)
        return logits

# ──────────────────────────────────────────────────────────────────────────────
# 5. 학습 함수 (A/B 공통)
# ──────────────────────────────────────────────────────────────────────────────
def train_model(model: nn.Module, model_name_str: str,
                train_loader, val_loader, test_loader,
                use_cat: bool) -> dict:
    """
    단일 모델(A 또는 B)을 학습하고 test 평가 결과를 반환한다.

    use_cat=True 이면 forward() 에 category_id 를 전달한다.
    use_cat=False(모델 A)이면 category_id 를 전달하지 않는다.
    두 경로를 하나의 함수로 처리해 학습 로직 중복을 제거한다.

    Mixed Precision(fp16) : GPU 환경에서 자동 활성화.
    Cosine Warmup Scheduler : 전체 step 의 10% warm-up → cosine 감소.
    Gradient Clipping(max_norm=1.0) : BERT fine-tuning 표준 — 기울기 폭발 방지.
    Best Checkpoint : val Macro-F1 최고 시점의 state_dict 보존 →
      마지막 epoch 이 최고가 아니어도 안전하게 최적 모델 복원.
    """
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)
    criterion    = nn.CrossEntropyLoss()
    use_amp      = (DEVICE.type == 'cuda')
    scaler       = GradScaler(enabled=use_amp)

    best_val_f1  = 0.0
    best_state   = None
    val_history  = []

    print(f'\n  {"─"*50}')
    print(f'  모델 : {model_name_str}  (use_cat={use_cat})')
    print(f'  {"─"*50}')

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            input_ids   = batch['input_ids'].to(DEVICE)
            attn_mask   = batch['attention_mask'].to(DEVICE)
            sent_labels = batch['sentiment_id'].to(DEVICE)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                if use_cat:
                    cat_ids = batch['category_id'].to(DEVICE)
                    logits  = model(input_ids=input_ids,
                                   attention_mask=attn_mask,
                                   category_id=cat_ids)
                else:
                    logits  = model(input_ids=input_ids,
                                   attention_mask=attn_mask)
                loss = criterion(logits, sent_labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                if use_cat:
                    logits = model(
                        input_ids=batch['input_ids'].to(DEVICE),
                        attention_mask=batch['attention_mask'].to(DEVICE),
                        category_id=batch['category_id'].to(DEVICE))
                else:
                    logits = model(
                        input_ids=batch['input_ids'].to(DEVICE),
                        attention_mask=batch['attention_mask'].to(DEVICE))
                val_preds.extend(logits.argmax(-1).cpu().tolist())
                val_trues.extend(batch['sentiment_id'].tolist())

        val_f1 = f1_score(val_trues, val_preds, average='macro', zero_division=0)
        val_history.append(round(val_f1, 4))
        elapsed = time.time() - t0
        print(f'  Epoch {epoch}/{EPOCHS}  '
              f'loss={train_loss/len(train_loader):.4f}  '
              f'val_macro_f1={val_f1:.4f}  ({elapsed:.0f}s)')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}

    # ── Test 평가 ─────────────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_trues, test_cats = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            if use_cat:
                logits = model(
                    input_ids=batch['input_ids'].to(DEVICE),
                    attention_mask=batch['attention_mask'].to(DEVICE),
                    category_id=batch['category_id'].to(DEVICE))
            else:
                logits = model(
                    input_ids=batch['input_ids'].to(DEVICE),
                    attention_mask=batch['attention_mask'].to(DEVICE))
            test_preds.extend(logits.argmax(-1).cpu().tolist())
            test_trues.extend(batch['sentiment_id'].tolist())
            test_cats.extend(batch['category_id'].tolist())

    acc      = accuracy_score(test_trues, test_preds)
    macro_f1 = f1_score(test_trues, test_preds, average='macro', zero_division=0)
    report   = classification_report(test_trues, test_preds,
                                     target_names=SENT_CLASSES,
                                     output_dict=True, zero_division=0)
    print(f'\n  [TEST] Accuracy={acc:.4f}  Macro-F1={macro_f1:.4f}')
    print(classification_report(test_trues, test_preds,
                                target_names=SENT_CLASSES, zero_division=0))

    return {
        'model_name' : model_name_str,
        'use_cat'    : use_cat,
        'accuracy'   : acc,
        'macro_f1'   : macro_f1,
        'report'     : report,
        'val_history': val_history,
        'test_preds' : test_preds,
        'test_trues' : test_trues,
        'test_cats'  : test_cats,   # 카테고리별 분석에 사용
    }

# ──────────────────────────────────────────────────────────────────────────────
# 6. 시각화
# ──────────────────────────────────────────────────────────────────────────────

def plot_ab_comparison(res_a: dict, res_b: dict):
    """
    그래프 ① : A vs B 전체 성능 비교 (막대) + epoch 별 val Macro-F1 추이 (꺾은선)

    왼쪽 패널 — Accuracy / Macro-F1 병렬 막대.
      A(기준선)와 B(카테고리 보조)를 나란히 배치해 절대 수치 차이를 직관적으로 표시.
      수치 차이를 화살표+숫자로 명시해 발표 시 "몇 포인트 향상"을 한눈에 설명 가능.

    오른쪽 패널 — epoch 별 validation Macro-F1 꺾은선.
      수렴 속도 차이 확인. B가 처음부터 높게 시작하면 카테고리 임베딩이
      초기 수렴을 돕는다는 근거가 된다.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 왼쪽: 막대 비교
    metrics  = ['Accuracy', 'Macro-F1']
    vals_a   = [res_a['accuracy'], res_a['macro_f1']]
    vals_b   = [res_b['accuracy'], res_b['macro_f1']]
    x        = np.arange(len(metrics))
    w        = 0.3
    bars_a   = axes[0].bar(x - w/2, vals_a, w, label='A: BERT 기준선',
                           color='#95A5A6', edgecolor='black', alpha=0.85)
    bars_b   = axes[0].bar(x + w/2, vals_b, w, label='B: +카테고리 임베딩',
                           color='#2ECC71', edgecolor='black', alpha=0.85)

    for bar, val in zip(bars_a, vals_a):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.004,
                     f'{val:.4f}', ha='center', fontsize=9)
    for bar, val in zip(bars_b, vals_b):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.004,
                     f'{val:.4f}', ha='center', fontsize=9)

    # 차이 표시 (Macro-F1 기준)
    diff = res_b['macro_f1'] - res_a['macro_f1']
    color_diff = '#27AE60' if diff >= 0 else '#E74C3C'
    axes[0].annotate(f'Δ={diff:+.4f}',
                     xy=(x[1] + w/2, vals_b[1]),
                     xytext=(x[1] + w/2 + 0.25, vals_b[1] + 0.02),
                     fontsize=11, color=color_diff, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=color_diff))

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, fontsize=11)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel('Score')
    axes[0].set_title('모델 A vs B 전체 성능 비교', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)

    # 오른쪽: epoch 추이
    epochs = list(range(1, EPOCHS + 1))
    axes[1].plot(epochs, res_a['val_history'], marker='o',
                 label='A: 기준선', color='#95A5A6', linewidth=2)
    axes[1].plot(epochs, res_b['val_history'], marker='s',
                 label='B: +카테고리', color='#2ECC71', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val Macro-F1')
    axes[1].set_title('Epoch별 Validation Macro-F1 추이', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(epochs)

    plt.suptitle('카테고리 임베딩 보조 특징 추가 효과 (A vs B)', fontsize=13)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'catfeat_01_ab_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  → {path}')


def plot_sentiment_f1(res_a: dict, res_b: dict):
    """
    그래프 ② : 긍정/중립/부정 클래스별 F1 비교

    클래스별로 A 와 B 를 나란히 배치한다.
    "중립" 클래스는 일반적으로 가장 어려운 클래스(양방향 경계)인데,
    카테고리 임베딩이 중립 판단에도 도움이 되는지 확인하는 게 핵심.
    예를 들어 '부상' 카테고리 기사는 대부분 부정이므로 카테고리를 알면
    중립/부정 경계 케이스에서 부정으로 더 정확히 분류할 수 있다.
    """
    classes  = SENT_CLASSES
    f1s_a    = [res_a['report'][c]['f1-score'] for c in classes]
    f1s_b    = [res_b['report'][c]['f1-score'] for c in classes]
    x        = np.arange(len(classes))
    w        = 0.3

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_a  = ax.bar(x - w/2, f1s_a, w, label='A: 기준선',
                     color='#95A5A6', edgecolor='black', alpha=0.85)
    bars_b  = ax.bar(x + w/2, f1s_b, w, label='B: +카테고리',
                     color='#2ECC71', edgecolor='black', alpha=0.85)

    for bar, val in zip(bars_a, f1s_a):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005, f'{val:.3f}',
                ha='center', fontsize=9)
    for bar, val in zip(bars_b, f1s_b):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005, f'{val:.3f}',
                ha='center', fontsize=9)

    # 클래스별 차이 표시
    for i, (fa, fb, cls) in enumerate(zip(f1s_a, f1s_b, classes)):
        diff = fb - fa
        color = '#27AE60' if diff >= 0 else '#E74C3C'
        ax.text(i, max(fa, fb) + 0.03, f'{diff:+.3f}',
                ha='center', fontsize=9, color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('F1-score')
    ax.set_title('감성 클래스별 F1 비교 (A vs B)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'catfeat_02_sentiment_f1.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  → {path}')


def plot_cat_impact(res_a: dict, res_b: dict):
    """
    그래프 ③ : 카테고리별 감성 분류 정확도 히트맵 (A vs B)

    행 = 카테고리, 열 = 감성 클래스.
    각 셀 값 = 해당 카테고리의 해당 감성 기사 중 정분류 비율(recall).

    이 히트맵으로 확인하는 것 :
      - "부상" 카테고리에서 B가 부정 recall 이 더 높은가?
        (부상 기사는 부정 비율이 높아 카테고리 힌트가 가장 유효할 것으로 예상)
      - "선수이동" 카테고리에서 긍정/부정 혼동이 A보다 B에서 줄었는가?
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, res, title in zip(axes,
                               [res_a, res_b],
                               ['A: 기준선', 'B: +카테고리 임베딩']):
        # 카테고리 × 감성 정확도 행렬 구성
        # test_cats : 각 샘플의 카테고리 id (0~5)
        # test_trues: 실제 감성 id, test_preds: 예측 감성 id
        mat = np.zeros((NUM_CATS, 3))   # (6카테고리, 3감성)
        cnt = np.zeros((NUM_CATS, 3))

        for pred, true, cat in zip(res['test_preds'],
                                   res['test_trues'],
                                   res['test_cats']):
            cnt[cat, true] += 1
            if pred == true:
                mat[cat, true] += 1

        # recall = 정분류 / 전체 (0으로 나누기 방지)
        with np.errstate(divide='ignore', invalid='ignore'):
            recall = np.where(cnt > 0, mat / cnt, np.nan)

        im = ax.imshow(recall, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(3))
        ax.set_yticks(range(NUM_CATS))
        ax.set_xticklabels(SENT_CLASSES, fontsize=10)
        ax.set_yticklabels(CAT_CLASSES, fontsize=9)
        ax.set_xlabel('감성 (예측 대상)')
        ax.set_ylabel('카테고리')
        ax.set_title(f'{title}\n(셀 값 = 해당 그룹 내 정분류율)', fontsize=10)

        for i in range(NUM_CATS):
            for j in range(3):
                val = recall[i, j]
                n   = int(cnt[i, j])
                txt = f'{val:.2f}\n(n={n})' if not np.isnan(val) else 'N/A'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=7,
                        color='white' if (not np.isnan(val) and val > 0.6)
                              else 'black')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('카테고리별 감성 분류 정확도 (recall) 비교', fontsize=13)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'catfeat_03_cat_impact.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  → {path}')


def plot_embedding_tsne(model_b: nn.Module):
    """
    그래프 ④ : 학습 완료된 카테고리 임베딩 t-SNE 시각화

    모델 B 의 cat_embed 가중치(6 × 32)를 t-SNE 로 2D 로 축소해 산점도로 표시.

    해석 방법 :
      - '투구'와 '타격'이 서로 가까우면 → 모델이 두 카테고리를 유사한 감성 맥락으로 학습
      - '부상'이 다른 카테고리와 멀리 떨어지면 → 부상 기사의 감성 패턴이 독특하게 학습됨
      - 임베딩이 골고루 분산되면 → 카테고리마다 구별되는 표현을 학습했다는 증거

    t-SNE 는 6개 포인트처럼 매우 적은 데이터에서는 perplexity 를 데이터 수보다
    작게(여기서는 5) 설정해야 에러가 발생하지 않는다.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print('  ⚠️ sklearn 미설치 → t-SNE 시각화 스킵')
        return

    model_b.eval()
    with torch.no_grad():
        # (6, 32) → numpy
        emb = model_b.cat_embed.weight.cpu().numpy()

    # 6개 포인트이므로 PCA 축소 없이 바로 t-SNE (perplexity < n_samples)
    tsne   = TSNE(n_components=2, perplexity=5, random_state=SEED, n_iter=1000)
    emb_2d = tsne.fit_transform(emb)    # (6, 2)

    colors = ['#E74C3C','#3498DB','#2ECC71','#F39C12','#9B59B6','#95A5A6']
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (cat, color) in enumerate(zip(CAT_CLASSES, colors)):
        ax.scatter(emb_2d[i, 0], emb_2d[i, 1],
                   c=color, s=200, zorder=5, label=cat)
        ax.annotate(cat, (emb_2d[i, 0], emb_2d[i, 1]),
                    textcoords='offset points', xytext=(8, 4),
                    fontsize=10, color=color, fontweight='bold')

    ax.set_title('카테고리 임베딩 t-SNE (모델 B 학습 완료 후)', fontsize=12)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'catfeat_04_embed_tsne.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  → {path}')


# ──────────────────────────────────────────────────────────────────────────────
# 7. 결과 저장
# ──────────────────────────────────────────────────────────────────────────────
def save_results(res_a: dict, res_b: dict):
    """
    catfeat_scores.json 에 A/B 결과를 저장하고 scores.json 에 append.
    test_preds/test_trues/test_cats 는 대용량이므로 JSON 에서 제외.
    """
    def clean(r):
        return {k: v for k, v in r.items()
                if k not in ('test_preds', 'test_trues', 'test_cats')}

    catfeat_path = os.path.join(RESULT_DIR, 'gyuwon_catfeat_scores.json')
    with open(catfeat_path, 'w', encoding='utf-8') as f:
        json.dump([clean(res_a), clean(res_b)], f, ensure_ascii=False, indent=2)
    print(f'  → {catfeat_path}')

    scores_path = os.path.join(RESULT_DIR, 'scores.json')
    existing = []
    if os.path.exists(scores_path):
        with open(scores_path, encoding='utf-8') as f:
            existing = json.load(f)

    for res in [res_a, res_b]:
        existing.append({
            'model'        : res['model_name'],
            'experiment_id': 'CAT-02',
            'task'         : 'sentiment',
            'accuracy'     : round(res['accuracy'], 4),
            'macro_f1'     : round(res['macro_f1'], 4),
            'report'       : res['report'],
            'notes'        : f'use_cat={res["use_cat"]}  cat_emb_dim={CAT_EMB_DIM}',
        })
    with open(scores_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f'  → {scores_path} (append 완료)')


# ──────────────────────────────────────────────────────────────────────────────
# 8. 콘솔 요약
# ──────────────────────────────────────────────────────────────────────────────
def print_summary(res_a: dict, res_b: dict):
    diff_acc = res_b['accuracy']  - res_a['accuracy']
    diff_f1  = res_b['macro_f1'] - res_a['macro_f1']
    print(f'\n{"="*60}')
    print('  실험 ② 결과 요약 — 카테고리 임베딩 보조 특징')
    print(f'{"="*60}')
    print(f'  {"모델":<35} {"Accuracy":>9}  {"Macro-F1":>9}')
    print(f'  {"-"*57}')
    print(f'  {res_a["model_name"]:<35} {res_a["accuracy"]:>9.4f}  {res_a["macro_f1"]:>9.4f}')
    print(f'  {res_b["model_name"]:<35} {res_b["accuracy"]:>9.4f}  {res_b["macro_f1"]:>9.4f}')
    print(f'  {"-"*57}')
    print(f'  {"차이 (B - A)":<35} {diff_acc:>+9.4f}  {diff_f1:>+9.4f}')

    verdict = ('카테고리 임베딩이 감성 분류에 유효 (Macro-F1 향상)'
               if diff_f1 >= 0.005 else
               '카테고리 임베딩 효과 미미 → BERT가 이미 맥락 내재화'
               if -0.005 < diff_f1 < 0.005 else
               '카테고리 임베딩 추가가 오히려 감성 분류를 방해')
    print(f'\n  해석 : {verdict}')
    print(f'  → 결과에 관계없이 보고서 분석 서사로 활용 가능')


# ──────────────────────────────────────────────────────────────────────────────
# 9. 메인
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'\n{"="*60}')
    print('  gyuwon_catFeature.py  실험 시작')
    print('  모델 A (기준선) vs 모델 B (카테고리 임베딩 보조)')
    print(f'  BERT={MODEL_NAME}  Epochs={EPOCHS}  CAT_EMB_DIM={CAT_EMB_DIM}')
    print(f'{"="*60}')

    # 데이터 로드
    splits = load_splits()

    # Tokenizer 는 A/B 공유 (동일 입력 처리)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── DataLoader 생성 ────────────────────────────────────────────────────────
    # use_cat=True : 모델 B 용 — category_id 반환
    # use_cat=False: 모델 A 용 — category_id=-1 반환 (무시)
    # 두 버전의 DataLoader 를 별도로 만들지 않고 use_cat=True 로 통일해
    # A는 forward() 에서 category_id 를 받아도 무시하는 구조로 처리한다.
    def make_loader(split, shuffle):
        ds = SentimentDataset(splits[split], tokenizer,
                              max_len=MAX_LEN, use_cat=True)
        return DataLoader(ds, batch_size=BATCH_SIZE,
                          shuffle=shuffle, num_workers=0)

    train_loader = make_loader('train', shuffle=True)
    val_loader   = make_loader('val',   shuffle=False)
    test_loader  = make_loader('test',  shuffle=False)

    # ── 모델 A 학습 ───────────────────────────────────────────────────────────
    model_a = ModelA_Baseline(MODEL_NAME).to(DEVICE)
    res_a   = train_model(model_a, 'BERT_Sentiment_A_Baseline',
                          train_loader, val_loader, test_loader,
                          use_cat=False)

    # ── 모델 B 학습 ───────────────────────────────────────────────────────────
    model_b = ModelB_CatFeature(MODEL_NAME,
                                num_cats=NUM_CATS,
                                cat_emb_dim=CAT_EMB_DIM).to(DEVICE)
    res_b   = train_model(model_b, 'BERT_Sentiment_B_CatEmb32',
                          train_loader, val_loader, test_loader,
                          use_cat=True)

    # ── 시각화 ────────────────────────────────────────────────────────────────
    print(f'\n[시각화 생성]')
    plot_ab_comparison(res_a, res_b)
    plot_sentiment_f1(res_a, res_b)
    plot_cat_impact(res_a, res_b)
    plot_embedding_tsne(model_b)   # 모델 B 학습 완료 후 임베딩 시각화

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    print(f'\n[결과 저장]')
    save_results(res_a, res_b)

    # ── 요약 ──────────────────────────────────────────────────────────────────
    print_summary(res_a, res_b)

    print(f'\n✅ 실험 완료')
    print(f'   그래프 → {FIG_DIR}/catfeat_*.png')
    print(f'   수치   → {RESULT_DIR}/gyuwon_catfeat_scores.json')
