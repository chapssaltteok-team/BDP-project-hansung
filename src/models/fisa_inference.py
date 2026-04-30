"""
fisa_inference.py
=================
FISA-conclave/klue-roberta-news-sentiment 추론 전용
담당: 안성민

역할
────
- 뉴스 도메인 특화 사전학습 모델 추론
- GPU 불필요, 로컬 CPU에서 실행
- klue/bert-base fine-tuning 결과와 Macro-F1 비교
- 발표 포인트:
  "도메인 특화 사전학습 모델 vs 직접 구축 데이터셋으로 fine-tuning한 모델"

주의
────
- 규정 5조: 추론 목적 GPU 사용 금지 → CPU 전용
- 학습/파인튜닝 없이 추론만 수행

실행: python src/models/fisa_inference.py  (로컬 CPU, 약 5~10분)
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

PROC_DIR   = 'data/processed'
RESULT_DIR = 'results'
FIG_DIR    = os.path.join(RESULT_DIR, 'figures')
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)

# FISA 모델은 CPU 전용
DEVICE     = torch.device('cpu')
MODEL_NAME = 'FISA-conclave/klue-roberta-news-sentiment'

# ── FISA 모델 라벨 매핑 ───────────────────────────────────────────────────────
# FISA 모델 출력: negative(0) / neutral(1) / positive(2)
# 우리 라벨:      부정      / 중립        / 긍정
FISA_ID2LABEL = {0: '부정', 1: '중립', 2: '긍정'}
OUR_LABEL2ID  = {'긍정': 0, '중립': 1, '부정': 2}


class Timer:
    def __init__(self, log_path='results/time_log.json'):
        self.log_path = log_path
        self.records  = {}
        if os.path.exists(log_path):
            with open(log_path, encoding='utf-8') as f:
                self.records = json.load(f)

    def start(self, name):
        self.records[name] = {'start': time.time(), 'end': None,
                              'elapsed_sec': None, 'elapsed_min': None}

    def end(self, name):
        elapsed = time.time() - self.records[name]['start']
        self.records[name].update({
            'end': time.time(),
            'elapsed_sec': round(elapsed, 2),
            'elapsed_min': round(elapsed / 60, 2),
        })
        print(f"  ⏱ [{name}] 소요시간: {elapsed/60:.1f}분 ({elapsed:.1f}초)")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)


class TextDataset(Dataset):
    def __init__(self, texts: list, tokenizer, max_len=256):
        self.texts     = texts
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        return {
            'input_ids'     : enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
        }


def run_fisa_inference(n_samples: int = None) -> dict:
    """
    FISA 모델로 테스트셋 감성 추론
    n_samples: None이면 전체, 정수면 해당 건수만 (빠른 테스트용)
    """
    timer = Timer()
    timer.start('fisa_inference')

    print(f"\n{'='*55}")
    print(f"  FISA 뉴스 감성 모델 추론 (CPU 전용)")
    print(f"  모델: {MODEL_NAME}")
    print(f"{'='*55}")

    # 데이터 로드
    test_df = pd.read_csv(os.path.join(PROC_DIR, 'test.csv'), encoding='utf-8-sig')
    if n_samples:
        test_df = test_df.head(n_samples)
    test_df = test_df.dropna(subset=['text', 'sentiment_str'])

    print(f"  테스트 건수: {len(test_df)}건")
    print(f"  ※ CPU 추론 중... 시간이 걸릴 수 있습니다")

    # 모델 로드
    print(f"\n  모델 다운로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    # FISA 모델의 실제 라벨 확인
    print(f"  FISA 모델 id2label: {model.config.id2label}")

    # 추론
    dataset = TextDataset(test_df['text'].tolist(), tokenizer)
    loader  = DataLoader(dataset, batch_size=32)

    preds_fisa = []  # FISA 모델 출력 (우리 라벨로 변환됨)
    probs_all  = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            logits = model(ids, attention_mask=mask).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            preds  = logits.argmax(-1).cpu().tolist()

            # FISA 라벨 → 우리 라벨로 변환
            # FISA: config의 id2label 확인 후 매핑
            converted = []
            for p in preds:
                fisa_label = model.config.id2label.get(p, str(p))
                # 영어 라벨 → 한국어 변환
                label_map = {
                    'negative': '부정', 'neutral': '중립', 'positive': '긍정',
                    'NEGATIVE': '부정', 'NEUTRAL': '중립', 'POSITIVE': '긍정',
                    '부정': '부정', '중립': '중립', '긍정': '긍정',
                }
                converted.append(label_map.get(fisa_label, '중립'))

            preds_fisa.extend(converted)
            probs_all.append(probs)

            if (i+1) % 5 == 0:
                print(f"  진행: {min((i+1)*32, len(test_df))}/{len(test_df)}건")

    y_true = test_df['sentiment_str'].tolist()
    probs_all = np.concatenate(probs_all, axis=0)

    # 평가
    report = classification_report(
        y_true, preds_fisa,
        target_names=['긍정', '중립', '부정'],
        output_dict=True, zero_division=0,
    )
    acc = accuracy_score(y_true, preds_fisa)
    f1  = report['macro avg']['f1-score']

    print(f"\n  [FISA 추론 결과]")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro-F1  : {f1:.4f}")
    print(f"\n  클래스별 F1:")
    for label in ['긍정', '중립', '부정']:
        if label in report:
            print(f"    {label}: {report[label]['f1-score']:.4f}")

    # 혼동 행렬
    cm = confusion_matrix(y_true, preds_fisa, labels=['긍정','중립','부정'])
    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['긍정','중립','부정'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('FISA 모델 감성 분류 혼동 행렬')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fisa_sentiment_cm.png'), dpi=150)
    plt.close()
    print(f"  혼동 행렬 저장: {FIG_DIR}/fisa_sentiment_cm.png")

    # 확률 벡터 저장 (앙상블용)
    np.save(os.path.join(RESULT_DIR, 'fisa_probs_test.npy'), probs_all)
    print(f"  확률 벡터 저장: results/fisa_probs_test.npy")

    timer.end('fisa_inference')

    result = {
        'model'       : 'FISA_klue-roberta-news-sentiment',
        'type'        : 'inference_only (no training)',
        'accuracy'    : round(acc, 4),
        'macro_f1'    : round(f1, 4),
        'elapsed_min' : timer.records['fisa_inference']['elapsed_min'],
        'n_samples'   : len(test_df),
        'report'      : report,
        'note'        : '뉴스 도메인 특화 사전학습 모델 (HuggingFace). 비교 기준선용.',
    }

    path = os.path.join(RESULT_DIR, 'scores.json')
    data = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    data.append(result)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  결과 저장: {path}")

    return result


if __name__ == '__main__':
    # 전체 테스트셋 추론 (CPU에서 약 5~10분)
    # 빠른 테스트: n_samples=100
    result = run_fisa_inference(n_samples=None)

    print(f"\n{'='*55}")
    print(f"  FISA 추론 완료")
    print(f"  Macro-F1: {result['macro_f1']}")
    print(f"  → BERT fine-tuning 결과와 비교 예정")
    print(f"{'='*55}")
