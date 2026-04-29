"""
ensemble.py
===========
Late Fusion 앙상블: BERT 감성 분류 + ResNet 이미지 분류
담당: 분류모델 담당자

핵심 전략
──────────
1. BERT, ResNet 각각 학습된 모델에서 Softmax 확률 벡터 추출
2. 검증셋에서 α (0.0~1.0) 탐색 → val Macro-F1 최대화
   P_ensemble = α × P_text + (1-α) × P_img
3. 최적 α로 테스트셋 최종 평가
4. 단독 vs 앙상블 성능 비교표 출력

실행 순서
──────────
1. python src/models/bert_sentiment.py    (BERT 먼저)
2. python src/models/image_classifier.py (ResNet 먼저)
3. python src/models/ensemble.py          (마지막)

실행: python src/models/ensemble.py
"""
import os, sys, json, random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

PROC_DIR   = 'data/processed'
OUTPUT_DIR = 'outputs'
RESULT_DIR = 'results'

SEED      = 42
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABEL2ID  = {'긍정': 0, '중립': 1, '부정': 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

# ── 저장된 모델 경로 (bert_sentiment.py, image_classifier.py 실행 후) ────────
BERT_MODEL_DIR   = os.path.join(OUTPUT_DIR, 'bert_sentiment_klue_base')
RESNET_MODEL_DIR = os.path.join(OUTPUT_DIR, 'resnet_sentiment_Exp2', 'model.pth')


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════════
# 텍스트 확률 벡터 추출
# ══════════════════════════════════════════════════════════════════════════════

class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len=256):
        self.texts  = df['text'].tolist()
        self.labels = df['sentiment_str'].map(LABEL2ID).tolist()
        self.tok    = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx], max_length=self.max_len,
            truncation=True, padding='max_length', return_tensors='pt'
        )
        return {
            'input_ids'     : enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label'         : torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_text_probs(df: pd.DataFrame) -> np.ndarray:
    """BERT 모델로 확률 벡터 (N, 3) 반환"""
    print(f"  [BERT] 확률 추출 중 ({len(df)}건)...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_DIR).to(DEVICE)
    model.eval()

    loader = DataLoader(TextDataset(df, tokenizer), batch_size=32)
    probs_all = []
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            logits = model(ids, attention_mask=mask).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_all.append(probs)

    return np.concatenate(probs_all, axis=0)  # (N, 3)


# ══════════════════════════════════════════════════════════════════════════════
# 이미지 확률 벡터 추출
# ══════════════════════════════════════════════════════════════════════════════

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_map: pd.DataFrame):
        url2path = dict(zip(image_map['url'], image_map['local_path']))
        self.paths  = [url2path.get(u, '') for u in df['url'].tolist()]
        self.labels = df['sentiment_str'].map(LABEL2ID).tolist()

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (180, 180, 180))
        return EVAL_TRANSFORM(img), torch.tensor(self.labels[idx], dtype=torch.long)


def build_resnet_from_checkpoint() -> nn.Module:
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 3),
    )
    state = torch.load(RESNET_MODEL_DIR, map_location=DEVICE)
    m.load_state_dict(state)
    return m.to(DEVICE)


def get_image_probs(df: pd.DataFrame, image_map: pd.DataFrame) -> np.ndarray:
    """ResNet 모델로 확률 벡터 (N, 3) 반환"""
    print(f"  [ResNet] 확률 추출 중 ({len(df)}건)...")
    model = build_resnet_from_checkpoint()
    model.eval()

    loader = DataLoader(
        ImageDataset(df, image_map), batch_size=32, num_workers=2)
    probs_all = []
    with torch.no_grad():
        for imgs, _ in loader:
            logits = model(imgs.to(DEVICE))
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_all.append(probs)

    return np.concatenate(probs_all, axis=0)  # (N, 3)


# ══════════════════════════════════════════════════════════════════════════════
# 가중치 자동 탐색 (α 탐색)
# ══════════════════════════════════════════════════════════════════════════════

def search_alpha(p_text: np.ndarray,
                 p_img: np.ndarray,
                 y_true: np.ndarray,
                 n_steps: int = 21) -> tuple[float, float]:
    """
    α ∈ [0.0, 1.0] 범위를 n_steps로 탐색
    P = α × P_text + (1-α) × P_img

    Returns: (best_alpha, best_f1)
    """
    alphas   = np.linspace(0.0, 1.0, n_steps)
    best_f1  = -1.0
    best_alpha = 0.5

    print("\n  α 탐색 결과")
    print(f"  {'α':>6}  {'Macro-F1':>10}  {'Acc':>8}")
    print(f"  {'-'*28}")

    for alpha in alphas:
        p_ens  = alpha * p_text + (1 - alpha) * p_img
        preds  = p_ens.argmax(axis=1)
        report = classification_report(
            y_true, preds, target_names=list(LABEL2ID.keys()),
            output_dict=True, zero_division=0
        )
        f1  = report['macro avg']['f1-score']
        acc = accuracy_score(y_true, preds)
        marker = ' ← 최적' if f1 > best_f1 else ''
        print(f"  {alpha:>6.2f}  {f1:>10.4f}  {acc:>8.4f}{marker}")

        if f1 > best_f1:
            best_f1    = f1
            best_alpha = alpha

    print(f"\n  최적 α = {best_alpha:.2f}  (val Macro-F1 = {best_f1:.4f})")
    return best_alpha, best_f1


# ══════════════════════════════════════════════════════════════════════════════
# 메인 앙상블 실행
# ══════════════════════════════════════════════════════════════════════════════

def run_ensemble():
    set_seed(SEED)
    print(f"\n{'='*55}")
    print(f"  Late Fusion 앙상블  |  device={DEVICE}")
    print(f"{'='*55}")

    # ── 데이터 로드 ─────────────────────────────────────
    val_df    = pd.read_csv(os.path.join(PROC_DIR, 'val.csv'),   encoding='utf-8-sig')
    test_df   = pd.read_csv(os.path.join(PROC_DIR, 'test.csv'),  encoding='utf-8-sig')
    image_map = pd.read_csv(os.path.join(PROC_DIR, 'image_map.csv'), encoding='utf-8-sig')

    y_val  = val_df['sentiment_str'].map(LABEL2ID).values
    y_test = test_df['sentiment_str'].map(LABEL2ID).values

    # ── 확률 벡터 추출 ───────────────────────────────────
    print("\n[1/3] 텍스트 모델 확률 추출")
    val_text  = get_text_probs(val_df)
    test_text = get_text_probs(test_df)

    print("\n[2/3] 이미지 모델 확률 추출")
    val_img   = get_image_probs(val_df,  image_map)
    test_img  = get_image_probs(test_df, image_map)

    # ── 단독 성능 확인 ───────────────────────────────────
    print("\n── 단독 모델 성능 (val) ──")
    for name, probs in [('BERT (텍스트)', val_text), ('ResNet (이미지)', val_img)]:
        preds = probs.argmax(axis=1)
        r = classification_report(y_val, preds,
                                   target_names=list(LABEL2ID.keys()),
                                   output_dict=True, zero_division=0)
        print(f"  {name:<20} Acc={r['accuracy']:.4f}  Macro-F1={r['macro avg']['f1-score']:.4f}")

    # ── α 탐색 ───────────────────────────────────────────
    print("\n[3/3] α 가중치 탐색 (검증셋)")
    best_alpha, _ = search_alpha(val_text, val_img, y_val, n_steps=21)

    # ── 테스트셋 최종 평가 ───────────────────────────────
    p_ensemble = best_alpha * test_text + (1 - best_alpha) * test_img
    preds_ens  = p_ensemble.argmax(axis=1)

    report_ens = classification_report(
        y_test, preds_ens,
        target_names=list(LABEL2ID.keys()),
        output_dict=True, zero_division=0
    )

    # ── 최종 비교표 ──────────────────────────────────────
    print(f"\n{'='*55}")
    print("  최종 성능 비교 (테스트셋)")
    print(f"{'='*55}")
    print(f"  {'모델':<24} {'Acc':>8}  {'Macro-F1':>10}")
    print(f"  {'-'*45}")

    results = {}
    for name, probs in [('BERT 단독', test_text), ('ResNet 단독', test_img)]:
        preds = probs.argmax(axis=1)
        r = classification_report(y_test, preds,
                                   target_names=list(LABEL2ID.keys()),
                                   output_dict=True, zero_division=0)
        results[name] = r
        print(f"  {name:<24} {r['accuracy']:>8.4f}  {r['macro avg']['f1-score']:>10.4f}")

    results['Late Fusion 앙상블'] = report_ens
    best_mark = ''
    ens_f1 = report_ens['macro avg']['f1-score']
    all_f1 = {k: v['macro avg']['f1-score'] for k, v in results.items()}
    if ens_f1 == max(all_f1.values()):
        best_mark = ' ← 최고'
    print(f"  {'Late Fusion 앙상블':<24} {report_ens['accuracy']:>8.4f}"
          f"  {ens_f1:>10.4f}{best_mark}")
    print(f"\n  최적 가중치: α={best_alpha:.2f} (텍스트) / {1-best_alpha:.2f} (이미지)")

    # ── 결과 저장 ────────────────────────────────────────
    final_result = {
        'model'       : 'LateFusion_Ensemble',
        'best_alpha'  : round(best_alpha, 2),
        'accuracy'    : round(report_ens['accuracy'], 4),
        'macro_f1'    : round(ens_f1, 4),
        'report'      : report_ens,
        'bert_f1'     : round(all_f1['BERT 단독'], 4),
        'resnet_f1'   : round(all_f1['ResNet 단독'], 4),
    }
    path = os.path.join(RESULT_DIR, 'scores.json')
    data = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else []
    data.append(final_result)
    json.dump(data, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {path}")

    return final_result


if __name__ == '__main__':
    # 사전 조건: bert_sentiment.py, image_classifier.py 실행 완료 필요
    if not os.path.isdir(BERT_MODEL_DIR):
        print("❌ BERT 모델 없음 → python src/models/bert_sentiment.py 먼저 실행")
        exit(1)
    if not os.path.exists(RESNET_MODEL_DIR):
        print("❌ ResNet 모델 없음 → python src/models/image_classifier.py 먼저 실행")
        exit(1)

    run_ensemble()
    print('\nLate Fusion 앙상블 완료')
