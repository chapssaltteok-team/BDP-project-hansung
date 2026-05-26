"""
gyuwon_categoryNums.py
======================
실험 ① : 카테고리 수(버전) 변화에 따른 BERT 분류 성능 ablation
담당   : 김규원

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
연구 목적
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
카테고리 설계(클래스 수 및 소수 클래스 처리 방식)가 BERT 분류 성능에
미치는 영향을 정량화한다.

  v1 (8개) : 타격 / 투구 / 트레이드 / 부상 / FA / 감독 / 심판 / 기타
             → 심판(71건), FA(82건) 가 독립 클래스로 존재 → 소수 클래스 문제
  v2 (6개) : 타격 / 투구 / 선수이동(FA+트레이드 통합) / 부상 / 감독·운영 / 기타
             → 심판 → 기타 편입, FA+트레이드 → 선수이동 통합

중간발표에서 이미 TF-IDF 기준 v2가 Macro-F1 +0.1317 높음이 확인됐다.
이 파일은 BERT fine-tuning 기준으로 동일 실험을 수행해 "BERT에서도
카테고리 설계가 성능에 직결되는가"를 검증하고, 추가로 v3(키워드 확장)
까지 비교해 v1 → v2 → v3 세 단계 추이 그래프를 생성한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실행 환경
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - RunPod RTX 3090 권장 (CPU로도 동작하나 epoch당 매우 느림)
  - 사전 조건 : data/processed/train.csv, val.csv, test.csv 존재
                (category_v1, category 컬럼 모두 포함된 버전 — prepro.py 실행 완료)
  - 실행 : python src/gyuwon_categoryNums.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
출력 결과물
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  results/figures/gyuwon/catnum_01_f1_trend.png      버전별 Macro-F1 추이
  results/figures/gyuwon/catnum_02_class_f1.png      각 버전 카테고리별 F1 상세
  results/figures/gyuwon/catnum_03_confusion.png     각 버전 Confusion Matrix
  results/gyuwon_catnum_scores.json                  이번 실험 전체 수치
  results/scores.json                                기존 scores.json 에 append
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0. 임포트
# ──────────────────────────────────────────────────────────────────────────────
# 기존 prepro.py 의 키워드·전처리 함수를 직접 복제해 단일 파일로 완결시킨다.
# (prepro.py 를 import 하면 __main__ 블록이 실행되는 부작용을 방지)

import os, sys, json, time, warnings
import re
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import platform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_cosine_schedule_with_warmup)
from torch.cuda.amp import autocast, GradScaler   # Mixed Precision (SPD-03)

warnings.filterwarnings('ignore')

# ── 한국어 폰트 설정 ──────────────────────────────────────────────────────────
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# ──────────────────────────────────────────────────────────────────────────────
# 1. 경로 및 하이퍼파라미터 상수
# ──────────────────────────────────────────────────────────────────────────────
PROC_DIR   = 'data/processed'
RESULT_DIR = 'results'
FIG_DIR    = os.path.join(RESULT_DIR, 'figures', 'gyuwon')
os.makedirs(FIG_DIR, exist_ok=True)

# BERT 모델 : klue/bert-base (기존 bert_category.py 와 동일 모델 사용)
MODEL_NAME  = 'klue/bert-base'
MAX_LEN     = 256      # 토큰 최대 길이 — 카테고리 분류는 감성보다 짧게 해도 충분
BATCH_SIZE  = 16
EPOCHS      = 3        # 중간발표와 동일 epoch 수 → 공정 비교
LR          = 2e-5
WARMUP_RATIO = 0.1     # 전체 step 의 10% 를 warm-up 으로 사용
SEED        = 42
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f'[환경] Device={DEVICE}  Model={MODEL_NAME}')

# ──────────────────────────────────────────────────────────────────────────────
# 2. 카테고리 키워드 정의 (prepro.py 복제 + v3 확장 추가)
# ──────────────────────────────────────────────────────────────────────────────
# prepro.py 의 label_category() 는 키워드 등장 횟수 합산 → 최다 카테고리 배정 방식.
# 동일 로직을 여기서도 사용해 v1/v2/v3 라벨을 재생성한다.
# (train.csv 에는 이미 category_v1, category(v2) 컬럼이 있으므로
#  v3 만 새로 계산하면 된다.)

_AD_PATTERNS = [
    r'<[^>]+>', r'\[.*?\]', r'▶.*', r'Copyright\s*©?.*',
    r'무단\s*전재.*금지', r'저작권자\s*©?.*?기자', r'기사\s*제보.*',
    r'[①②③④⑤⑥⑦⑧⑨⑩]',
    r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
    r'https?://\S+', r'\s{2,}',
]

def clean_text(text: str) -> str:
    """광고 패턴·HTML 제거 후 한글/영문/숫자/기본 문장부호만 남김 (prepro.py 동일)"""
    if not isinstance(text, str):
        return ''
    for pat in _AD_PATTERNS:
        text = re.sub(pat, ' ', text)
    text = re.sub(r'[^\w\s가-힣.,!?]', ' ', text)
    return text.strip()

def label_category(text: str, kw_dict: dict) -> str:
    """키워드 등장 횟수 합산 → 최다 카테고리 반환 (0이면 '기타')"""
    scores = {cat: sum(text.count(kw) for kw in kws)
              for cat, kws in kw_dict.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else '기타'

# ── v1 : 8개 카테고리 (계획서 원본) ──────────────────────────────────────────
KEYWORDS_V1 = {
    '타격'    : ['홈런','안타','타율','타점','출루율','장타','타격','배팅','4번타자','클린업'],
    '투구'    : ['삼진','방어율','선발','불펜','마무리','투구','완봉','완투','세이브','홀드','구위'],
    '트레이드': ['트레이드','이적','교환','방출','영입','계약','다년계약'],
    '부상'    : ['부상','이탈','등록 말소','수술','재활','회복','부상 우려'],
    'FA'      : ['FA','자유계약','다년 계약','연봉','계약 협상','옵트아웃','잔류','해외 진출'],
    '감독'    : ['감독','코치','벤치','작전','선발 로테이션','수뇌부','구단주','단장'],
    '심판'    : ['심판','오심','판정','비디오 판독','어필'],
    # '기타' 는 위 7개 중 점수 0 일 때 자동 배정
}

# ── v2 : 6개 카테고리 (소수 클래스 통합 — 중간발표 기준) ─────────────────────
KEYWORDS_V2 = {
    '타격'     : ['홈런','안타','타율','타점','출루율','장타율',
                  '타격','배팅','4번타자','클린업','득점','타선'],
    '투구'     : ['삼진','방어율','선발','불펜','마무리',
                  '투구','완봉','완투','세이브','홀드','구위',
                  '피안타','볼넷','탈삼진','이닝'],
    '선수이동' : ['트레이드','이적','교환','영입','계약',
                  '다년계약','다년 계약','FA','자유계약',
                  '계약 협상','연봉','옵트아웃','잔류','해외 진출','FA 시장','방출'],
    '부상'     : ['부상','등록 말소','수술','재활','회복',
                  '결장','부상 우려','통증','접질'],
    '감독·운영': ['감독','코치','벤치','작전',
                  '선발 로테이션','수뇌부','구단주','단장',
                  '구단','프런트','스프링캠프'],
    # '기타' 자동 배정
}

# ── v3 : 6개 카테고리 + 키워드 대폭 확장 (도전적 실험) ───────────────────────
# v2 에서 각 카테고리 키워드를 동의어·변형어·통계 용어까지 포함해 약 2배 확장.
# 목표 : 키워드 커버리지 향상 → '기타' 비율 감소 → 소수 클래스 recall 개선.
KEYWORDS_V3 = {
    '타격'     : ['홈런','안타','타율','타점','출루율','장타율',
                  '타격','배팅','4번타자','클린업','득점','타선',
                  # v3 추가
                  '2루타','3루타','내야안타','희생번트','희생플라이',
                  'OPS','WAR','wRC','타격왕','수위타자','연속안타',
                  '끝내기안타','끝내기홈런','만루홈런','역전타','결승타',
                  '장타','번트','도루','대주자','지명타자','DH'],
    '투구'     : ['삼진','방어율','선발','불펜','마무리',
                  '투구','완봉','완투','세이브','홀드','구위',
                  '피안타','볼넷','탈삼진','이닝',
                  # v3 추가
                  '선발투수','중간계투','마무리투수','셋업맨','원포인트',
                  '구속','슬라이더','커브','체인지업','포크볼','스플리터',
                  '투심','포심','싱커','커터','피홈런','자책점',
                  'ERA','WHIP','FIP','K/9','BB/9','피출루율',
                  '사이드암','언더핸드','오버핸드','노히트','퍼펙트'],
    '선수이동' : ['트레이드','이적','교환','영입','계약',
                  '다년계약','다년 계약','FA','자유계약',
                  '계약 협상','연봉','옵트아웃','잔류','해외 진출','FA 시장','방출',
                  # v3 추가
                  '외국인선수','외인','방출','웨이버','임의탈퇴',
                  '군입대','군제대','보류','옵션','인센티브',
                  '계약금','연봉조정','스플릿계약','마이너계약',
                  '메이저리그','일본리그','NPB','MLB','KBO복귀'],
    '부상'     : ['부상','등록 말소','수술','재활','회복',
                  '결장','부상 우려','통증','접질',
                  # v3 추가
                  '골절','인대','어깨','팔꿈치','무릎','발목',
                  '허리','근육','염좌','타박상','뇌진탕',
                  '토미존','수술','재수술','회복기간','복귀일정',
                  '1군말소','2군강등','부상자명단','DL','IL'],
    '감독·운영' : ['감독','코치','벤치','작전',
                   '선발 로테이션','수뇌부','구단주','단장',
                   '구단','프런트','스프링캠프',
                   # v3 추가
                   '수석코치','타격코치','투수코치','배터리코치',
                   '작전코치','주루코치','감독대행','임시감독',
                   '스카우트','드래프트','신인지명','2군감독',
                   '전력분석','전략','라인업','선발명단','엔트리',
                   '마케팅','구단운영','연고지','홈구장'],
    # '기타' 자동 배정
}

# 버전 메타데이터 (반복 처리를 위한 딕셔너리)
# col_name : train.csv 에서 읽어올 컬럼명 (v3 는 새로 계산)
VERSIONS = {
    'v1': {'keywords': KEYWORDS_V1, 'col_name': 'category_v1',  'n_cats': 8,
           'label': 'v1 (8개, 소수클래스 유지)'},
    'v2': {'keywords': KEYWORDS_V2, 'col_name': 'category',     'n_cats': 6,
           'label': 'v2 (6개, 소수클래스 통합)'},
    'v3': {'keywords': KEYWORDS_V3, 'col_name': 'category_v3',  'n_cats': 6,
           'label': 'v3 (6개, 키워드 확장)'},
}

# ──────────────────────────────────────────────────────────────────────────────
# 3. 데이터 로드 및 v3 라벨 생성
# ──────────────────────────────────────────────────────────────────────────────
def load_splits() -> dict:
    """
    train/val/test.csv 를 읽어 각 split 별 DataFrame 반환.
    v3 카테고리 컬럼이 없으면 text 컬럼으로 새로 계산해 추가한다.

    반환 형식 : {'train': df, 'val': df, 'test': df}
    """
    splits = {}
    for split in ['train', 'val', 'test']:
        path = os.path.join(PROC_DIR, f'{split}.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'{path} 없음 → prepro.py 를 먼저 실행하세요')
        df = pd.read_csv(path, encoding='utf-8-sig')

        # v3 라벨 : 아직 CSV 에 없으므로 직접 계산
        # text 컬럼이 없으면 title_clean + body_clean 으로 구성
        if 'text' not in df.columns:
            df['text'] = (df['title_clean'].fillna('') + ' '
                          + df['body_clean'].fillna(''))
        df['category_v3'] = df['text'].apply(
            lambda t: label_category(t, KEYWORDS_V3))

        splits[split] = df
        print(f'  로드: {split}.csv  {len(df)}건')

    return splits


# ──────────────────────────────────────────────────────────────────────────────
# 4. PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────
class CategoryDataset(Dataset):
    """
    BERT 입력용 Dataset.
    text 를 tokenizer 로 인코딩하고 label_id 를 반환한다.

    label2id 는 버전마다 다르므로 외부에서 주입한다.
    (v1 은 최대 8개 클래스, v2/v3 는 최대 6개 클래스)
    """
    def __init__(self, df: pd.DataFrame, col_name: str,
                 tokenizer, label2id: dict, max_len: int):
        self.texts    = df['text'].fillna('').tolist()
        self.labels   = df[col_name].map(label2id).fillna(0).astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

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
            'input_ids'     : enc['input_ids'].squeeze(0),       # (max_len,)
            'attention_mask': enc['attention_mask'].squeeze(0),   # (max_len,)
            'label'         : torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 5. 학습 함수
# ──────────────────────────────────────────────────────────────────────────────
def train_one_version(version_key: str, splits: dict) -> dict:
    """
    단일 버전(v1/v2/v3)에 대해 BERT fine-tuning 을 수행하고
    test set 평가 결과를 dict 로 반환한다.

    Mixed Precision(fp16) 을 GPU 환경에서 자동 적용해 학습 속도를 높인다.
    Cosine Warmup LR Scheduler 로 안정적 수렴을 유도한다.
    """
    cfg      = VERSIONS[version_key]
    col_name = cfg['col_name']
    kw_dict  = cfg['keywords']
    label    = cfg['label']

    print(f'\n{"="*60}')
    print(f'  버전: {label}')
    print(f'{"="*60}')

    # ── 4-1. 클래스 목록 및 인코딩 맵 ────────────────────────────────────────
    # 해당 버전의 키워드 키 + '기타' 로 클래스 목록 구성.
    # v1 은 8개(심판 포함), v2/v3 은 6개.
    classes   = list(kw_dict.keys()) + ['기타']
    label2id  = {c: i for i, c in enumerate(classes)}
    id2label  = {i: c for c, i in label2id.items()}
    num_labels = len(classes)
    print(f'  클래스({num_labels}개): {classes}')

    # ── 4-2. 데이터 분포 출력 ─────────────────────────────────────────────────
    for split_name in ['train', 'val', 'test']:
        dist = splits[split_name][col_name].value_counts().to_dict()
        total = sum(dist.values())
        print(f'  [{split_name}] 총 {total}건  |  ' +
              '  '.join(f'{k}:{v}' for k, v in sorted(dist.items())))

    # ── 4-3. Tokenizer & Model ────────────────────────────────────────────────
    # AutoTokenizer : klue/bert-base 의 WordPiece 토크나이저.
    # AutoModelForSequenceClassification : BERT 위에 Linear(hidden, num_labels)
    #   분류 head 가 자동으로 붙는 구조 → fine-tuning 시 전체 파라미터 업데이트.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels,
        ignore_mismatched_sizes=True   # head 크기가 버전마다 다르므로 경고 억제
    ).to(DEVICE)

    # ── 4-4. DataLoader ───────────────────────────────────────────────────────
    def make_loader(split_name, shuffle):
        ds = CategoryDataset(splits[split_name], col_name,
                             tokenizer, label2id, MAX_LEN)
        return DataLoader(ds, batch_size=BATCH_SIZE,
                          shuffle=shuffle, num_workers=0)

    train_loader = make_loader('train', shuffle=True)
    val_loader   = make_loader('val',   shuffle=False)
    test_loader  = make_loader('test',  shuffle=False)

    # ── 4-5. Optimizer & Scheduler ────────────────────────────────────────────
    # AdamW : weight decay 로 과적합 억제. BERT fine-tuning 표준 옵티마이저.
    # Cosine Warmup : 초기 warmup_steps 동안 LR 을 선형 증가 →
    #   이후 cosine 곡선으로 감소. 초기 불안정성 방지 + 빠른 수렴.
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)
    criterion    = nn.CrossEntropyLoss()

    # Mixed Precision scaler : GPU 에서만 활성화
    use_amp = (DEVICE.type == 'cuda')
    scaler  = GradScaler(enabled=use_amp)

    # ── 4-6. 학습 루프 ────────────────────────────────────────────────────────
    best_val_f1   = 0.0
    best_state    = None
    history       = []   # epoch 별 val Macro-F1 기록

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            input_ids  = batch['input_ids'].to(DEVICE)
            attn_mask  = batch['attention_mask'].to(DEVICE)
            labels_gpu = batch['label'].to(DEVICE)

            optimizer.zero_grad()

            # autocast : fp16 연산 (GPU 에서만 효과, CPU 에서는 무시됨)
            with autocast(enabled=use_amp):
                outputs = model(input_ids=input_ids,
                                attention_mask=attn_mask,
                                labels=labels_gpu)
                loss = outputs.loss   # CrossEntropy 내장

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # gradient clipping : 기울기 폭발 방지 (BERT fine-tuning 표준)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()

        # ── validation ────────────────────────────────────────────────────────
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(input_ids=batch['input_ids'].to(DEVICE),
                            attention_mask=batch['attention_mask'].to(DEVICE))
                val_preds.extend(out.logits.argmax(-1).cpu().tolist())
                val_labels.extend(batch['label'].tolist())

        val_f1 = f1_score(val_labels, val_preds, average='macro',
                          zero_division=0)
        history.append(val_f1)
        elapsed = time.time() - t0
        print(f'  Epoch {epoch}/{EPOCHS}  '
              f'train_loss={train_loss/len(train_loader):.4f}  '
              f'val_macro_f1={val_f1:.4f}  ({elapsed:.0f}s)')

        # best model 저장 (val Macro-F1 기준)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}

    # ── 4-7. Test 평가 ────────────────────────────────────────────────────────
    # best checkpoint 로 복원 후 test set 최종 평가
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(input_ids=batch['input_ids'].to(DEVICE),
                        attention_mask=batch['attention_mask'].to(DEVICE))
            test_preds.extend(out.logits.argmax(-1).cpu().tolist())
            test_labels.extend(batch['label'].tolist())

    acc      = accuracy_score(test_labels, test_preds)
    macro_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
    report   = classification_report(test_labels, test_preds,
                                     target_names=classes,
                                     output_dict=True, zero_division=0)
    cm       = confusion_matrix(test_labels, test_preds,
                                labels=list(range(num_labels)))

    print(f'\n  [TEST] Accuracy={acc:.4f}  Macro-F1={macro_f1:.4f}')
    print(classification_report(test_labels, test_preds,
                                target_names=classes, zero_division=0))

    return {
        'version'   : version_key,
        'label'     : label,
        'classes'   : classes,
        'num_labels': num_labels,
        'accuracy'  : acc,
        'macro_f1'  : macro_f1,
        'report'    : report,
        'cm'        : cm.tolist(),
        'val_history': history,      # epoch 별 val Macro-F1
        'id2label'  : id2label,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6. 시각화
# ──────────────────────────────────────────────────────────────────────────────
def plot_f1_trend(results: list):
    """
    그래프 ① : 버전별 Macro-F1 막대 + epoch 별 val Macro-F1 추이 (두 축)

    왼쪽 패널 — 최종 test Macro-F1 막대그래프 (v1 vs v2 vs v3 비교)
    오른쪽 패널 — 각 버전의 epoch별 validation Macro-F1 꺾은선 그래프
    이 두 패널을 나란히 보면 "최종 수치 차이"와 "수렴 속도 차이"를 동시에 확인 가능.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 왼쪽 : 최종 Macro-F1 막대
    labels   = [r['label'] for r in results]
    f1_vals  = [r['macro_f1'] for r in results]
    colors   = ['#E74C3C', '#2ECC71', '#3498DB'][:len(results)]
    bars = axes[0].bar(labels, f1_vals, color=colors, edgecolor='black', alpha=0.85)
    for bar, val in zip(bars, f1_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.005,
                     f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
    # 버전 간 차이 화살표 표시
    if len(results) >= 2:
        diff_v1v2 = results[1]['macro_f1'] - results[0]['macro_f1']
        axes[0].annotate('', xy=(1, results[1]['macro_f1']),
                         xytext=(0, results[0]['macro_f1']),
                         arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        axes[0].text(0.5, (results[0]['macro_f1'] + results[1]['macro_f1'])/2,
                     f'{diff_v1v2:+.4f}', ha='center', fontsize=10,
                     color='green' if diff_v1v2 > 0 else 'red')
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel('Macro-F1')
    axes[0].set_title('버전별 최종 Test Macro-F1 비교', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', labelsize=8)

    # 오른쪽 : epoch 별 validation Macro-F1 추이
    line_colors = ['#E74C3C', '#2ECC71', '#3498DB']
    for i, r in enumerate(results):
        epochs = list(range(1, len(r['val_history']) + 1))
        axes[1].plot(epochs, r['val_history'],
                     marker='o', label=r['label'],
                     color=line_colors[i], linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val Macro-F1')
    axes[1].set_title('Epoch별 Validation Macro-F1 추이', fontsize=12)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(list(range(1, EPOCHS + 1)))

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, 'catnum_01_f1_trend.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'  → {save_path}')


def plot_class_f1(results: list):
    """
    그래프 ② : 각 버전의 카테고리별 F1 상세 (서브플롯 1행 × 버전 수)

    카테고리별 F1 을 가로 막대로 표시하고 Macro-F1 기준선을 수직선으로 그린다.
    버전별로 나란히 보면 통합/확장이 어느 카테고리에 영향을 줬는지 직관적으로 파악.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        report  = r['report']
        classes = [c for c in r['classes']
                   if c not in ('accuracy', 'macro avg', 'weighted avg')
                   and c in report]
        f1s     = [report[c]['f1-score'] for c in classes]
        counts  = [int(report[c]['support']) for c in classes]

        bar_colors = ['#2ECC71' if f >= 0.7
                      else '#F39C12' if f >= 0.5
                      else '#E74C3C' for f in f1s]
        bars = ax.barh(classes, f1s, color=bar_colors,
                       edgecolor='black', alpha=0.85)
        for bar, f, cnt in zip(bars, f1s, counts):
            ax.text(bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'{f:.3f} (n={cnt})', va='center', fontsize=8)
        ax.axvline(r['macro_f1'], color='navy', linestyle='--', linewidth=1.5,
                   label=f"Macro-F1={r['macro_f1']:.3f}")
        ax.set_xlim(0, 1.15)
        ax.set_title(f"{r['label']}\nAcc={r['accuracy']:.3f}", fontsize=10)
        ax.set_xlabel('F1-score')
        ax.legend(fontsize=8)
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle('버전별 카테고리 F1 상세 비교', fontsize=13, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, 'catnum_02_class_f1.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → {save_path}')


def plot_confusion(results: list):
    """
    그래프 ③ : 각 버전의 Confusion Matrix (정규화 비율)

    행 = 실제 레이블, 열 = 예측 레이블. 대각선 = 정분류.
    소수 클래스(v1의 심판, FA)가 다른 클래스로 얼마나 혼동되는지 시각화.
    v2/v3 에서는 해당 클래스가 사라지므로 비교가 명확히 드러난다.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        cm  = np.array(r['cm'], dtype=float)
        # 행(실제) 기준 정규화 → 각 셀 = recall 의미
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm  = np.divide(cm, row_sums,
                             where=row_sums != 0,
                             out=np.zeros_like(cm))
        classes = r['classes']
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        for i in range(len(classes)):
            for j in range(len(classes)):
                val = cm_norm[i, j]
                ax.text(j, i, f'{val:.2f}',
                        ha='center', va='center', fontsize=7,
                        color='white' if val > 0.5 else 'black')
        ax.set_xlabel('예측')
        ax.set_ylabel('실제')
        ax.set_title(f'{r["label"]}\nMacro-F1={r["macro_f1"]:.3f}', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('버전별 Confusion Matrix (행 정규화)', fontsize=13, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, 'catnum_03_confusion.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → {save_path}')


# ──────────────────────────────────────────────────────────────────────────────
# 7. 결과 저장
# ──────────────────────────────────────────────────────────────────────────────
def save_results(results: list):
    """
    이번 실험 전체 수치를 catnum_scores.json 에 저장하고,
    기존 results/scores.json 에도 append 해 전체 실험 히스토리를 유지한다.
    cm(Confusion Matrix) 는 list 형태로 직렬화 가능하므로 그대로 저장.
    """
    # ── catnum 전용 저장 ──────────────────────────────────────────────────────
    catnum_path = os.path.join(RESULT_DIR, 'gyuwon_catnum_scores.json')
    output = []
    for r in results:
        output.append({
            'experiment'  : 'categoryNums_ablation',
            'version'     : r['version'],
            'label'       : r['label'],
            'num_labels'  : r['num_labels'],
            'accuracy'    : round(r['accuracy'], 4),
            'macro_f1'    : round(r['macro_f1'], 4),
            'report'      : r['report'],
            'cm'          : r['cm'],
            'val_history' : [round(v, 4) for v in r['val_history']],
        })
    with open(catnum_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'  → {catnum_path}')

    # ── 기존 scores.json append ───────────────────────────────────────────────
    scores_path = os.path.join(RESULT_DIR, 'scores.json')
    existing = []
    if os.path.exists(scores_path):
        with open(scores_path, encoding='utf-8') as f:
            existing = json.load(f)

    for r in results:
        existing.append({
            'model'      : f"BERT_Category_{r['version']}",
            'experiment_id': 'CAT-NUM-ABLATION',
            'task'       : 'category',
            'num_labels' : r['num_labels'],
            'accuracy'   : round(r['accuracy'], 4),
            'macro_f1'   : round(r['macro_f1'], 4),
            'report'     : r['report'],
            'notes'      : r['label'],
        })
    with open(scores_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f'  → {scores_path} (append 완료)')


# ──────────────────────────────────────────────────────────────────────────────
# 8. 콘솔 요약 출력
# ──────────────────────────────────────────────────────────────────────────────
def print_summary(results: list):
    print(f'\n{"="*60}')
    print('  카테고리 버전별 성능 요약')
    print(f'{"="*60}')
    print(f'  {"버전":<30} {"클래스수":>6}  {"Accuracy":>9}  {"Macro-F1":>9}')
    print(f'  {"-"*58}')
    for r in results:
        print(f'  {r["label"]:<30} {r["num_labels"]:>6}  '
              f'{r["accuracy"]:>9.4f}  {r["macro_f1"]:>9.4f}')

    if len(results) >= 2:
        diff = results[1]['macro_f1'] - results[0]['macro_f1']
        print(f'\n  v1 → v2 Macro-F1 변화 : {diff:+.4f}  '
              f'({"향상" if diff > 0 else "하락"})')
    if len(results) >= 3:
        diff2 = results[2]['macro_f1'] - results[1]['macro_f1']
        print(f'  v2 → v3 Macro-F1 변화 : {diff2:+.4f}  '
              f'({"향상" if diff2 > 0 else "하락"})')


# ──────────────────────────────────────────────────────────────────────────────
# 9. 메인 실행
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'\n{"="*60}')
    print('  gyuwon_categoryNums.py  실험 시작')
    print(f'  버전: v1(8개) → v2(6개) → v3(6개+키워드확장)')
    print(f'  BERT: {MODEL_NAME}  Epochs={EPOCHS}  LR={LR}')
    print(f'{"="*60}')

    # 데이터 로드 (v3 라벨 자동 생성 포함)
    splits = load_splits()

    # 각 버전 순서대로 fine-tuning
    # 버전 순서를 바꾸고 싶으면 아래 리스트만 수정
    results = []
    for vkey in ['v1', 'v2', 'v3']:
        result = train_one_version(vkey, splits)
        results.append(result)

    # 시각화 3종
    print(f'\n[시각화 생성]')
    plot_f1_trend(results)
    plot_class_f1(results)
    plot_confusion(results)

    # 결과 저장
    print(f'\n[결과 저장]')
    save_results(results)

    # 요약 출력
    print_summary(results)

    print(f'\n✅ 실험 완료')
    print(f'   그래프 → {FIG_DIR}/catnum_*.png')
    print(f'   수치   → {RESULT_DIR}/gyuwon_catnum_scores.json')