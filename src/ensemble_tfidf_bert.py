"""
ensemble_tfidf_bert.py
======================
Late Fusion 앙상블: TF-IDF + BERT 감성 분류
담당: 이윤서

핵심 전략
──────────
1. TF-IDF, BERT 각각에서 Softmax 확률 벡터 추출
2. 검증셋에서 α (0.0~1.0) 탐색 → val Macro-F1 최대화
   P_ensemble = α × P_TF-IDF + (1-α) × P_BERT
3. 최적 α로 테스트셋 최종 평가
4. 단독 vs 앙상블 성능 비교표 출력

실행 순서
──────────
1. python src/models/bert_sentiment.py  (BERT 먼저)
2. python src/ensemble_tfidf_bert.py    (마지막)

실행: python src/ensemble_tfidf_bert.py
"""
import os, sys, json, random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

PROC_DIR   = 'data/processed'
OUTPUT_DIR = 'outputs'
RESULT_DIR = 'results'

SEED     = 42
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABEL2ID = {'긍정': 0, '중립': 1, '부정': 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
LABELS   = ['긍정', '중립', '부정']

BERT_MODEL_DIR = os.path.join(OUTPUT_DIR, 'bert_sentiment_klue_base')


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════════
# TF-IDF 확률 벡터 추출
# ══════════════════════════════════════════════════════════════════════════════

def get_tfidf_probs(train_df: pd.DataFrame,
                    val_df: pd.DataFrame,
                    test_df: pd.DataFrame):
    print("  [TF-IDF] 학습 및 확률 추출 중...")

    # train만으로 학습 (val/test 누수 방지)
    X_train = train_df['text'].fillna('').tolist()
    y_train = train_df['sentiment_str'].tolist()
    X_val   = val_df['text'].fillna('').tolist()
    X_test  = test_df['text'].fillna('').tolist()

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True,
            min_df=2,
            analyzer='char_wb',
        )),
        ('clf', CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, random_state=42),
            cv=3,
        )),
    ])

    pipe.fit(X_train, y_train)

    classes = pipe.named_steps['clf'].classes_
    label_order = [list(classes).index(l) for l in LABELS]

    val_probs  = pipe.predict_proba(X_val)[:, label_order]
    test_probs = pipe.predict_proba(X_test)[:, label_order]

    val_acc  = accuracy_score(val_df['sentiment_str'], pipe.predict(X_val))
    test_acc = accuracy_score(test_df['sentiment_str'], pipe.predict(X_test))
    print(f"  [TF-IDF] val Acc={val_acc:.4f}  test Acc={test_acc:.4f}")

    return val_probs, test_probs


# ══════════════════════════════════════════════════════════════════════════════
# BERT 확률 벡터 추출
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


def get_bert_probs(df: pd.DataFrame) -> np.ndarray:
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

    return np.concatenate(probs_all, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# α 탐색
# ══════════════════════════════════════════════════════════════════════════════

def search_alpha(p_tfidf, p_bert, y_true, n_steps=21):
    alphas     = np.linspace(0.0, 1.0, n_steps)
    best_f1    = -1.0
    best_alpha = 0.5

    print("\n  α 탐색 결과 (α=TF-IDF 비중)")
    print(f"  {'α':>6}  {'Macro-F1':>10}  {'Acc':>8}")
    print(f"  {'-'*28}")

    for alpha in alphas:
        p_ens  = alpha * p_tfidf + (1 - alpha) * p_bert
        preds  = p_ens.argmax(axis=1)
        report = classification_report(
            y_true, preds, target_names=LABELS,
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
    print(f"  TF-IDF + BERT Late Fusion 앙상블  |  device={DEVICE}")
    print(f"{'='*55}")

    train_df = pd.read_csv(os.path.join(PROC_DIR, 'train.csv'), encoding='utf-8-sig')
    val_df   = pd.read_csv(os.path.join(PROC_DIR, 'val.csv'),   encoding='utf-8-sig')
    test_df  = pd.read_csv(os.path.join(PROC_DIR, 'test.csv'),  encoding='utf-8-sig')

    y_val  = val_df['sentiment_str'].map(LABEL2ID).values
    y_test = test_df['sentiment_str'].map(LABEL2ID).values

    print("\n[1/3] TF-IDF 확률 추출")
    val_tfidf, test_tfidf = get_tfidf_probs(train_df, val_df, test_df)

    print("\n[2/3] BERT 확률 추출")
    val_bert  = get_bert_probs(val_df)
    test_bert = get_bert_probs(test_df)

    print("\n── 단독 모델 성능 (val) ──")
    for name, probs in [('TF-IDF', val_tfidf), ('BERT', val_bert)]:
        preds = probs.argmax(axis=1)
        r = classification_report(y_val, preds, target_names=LABELS,
                                   output_dict=True, zero_division=0)
        print(f"  {name:<20} Acc={r['accuracy']:.4f}  Macro-F1={r['macro avg']['f1-score']:.4f}")

    print("\n[3/3] α 가중치 탐색 (검증셋)")
    best_alpha, _ = search_alpha(val_tfidf, val_bert, y_val, n_steps=21)

    p_ensemble = best_alpha * test_tfidf + (1 - best_alpha) * test_bert
    preds_ens  = p_ensemble.argmax(axis=1)

    report_ens = classification_report(
        y_test, preds_ens, target_names=LABELS,
        output_dict=True, zero_division=0
    )

    print(f"\n{'='*55}")
    print("  최종 성능 비교 (테스트셋)")
    print(f"{'='*55}")
    print(f"  {'모델':<24} {'Acc':>8}  {'Macro-F1':>10}")
    print(f"  {'-'*45}")

    all_f1 = {}
    for name, probs in [('TF-IDF 단독', test_tfidf), ('BERT 단독', test_bert)]:
        preds = probs.argmax(axis=1)
        r = classification_report(y_test, preds, target_names=LABELS,
                                   output_dict=True, zero_division=0)
        all_f1[name] = r['macro avg']['f1-score']
        print(f"  {name:<24} {r['accuracy']:>8.4f}  {r['macro avg']['f1-score']:>10.4f}")

    ens_f1 = report_ens['macro avg']['f1-score']
    all_f1['TF-IDF+BERT 앙상블'] = ens_f1
    best_mark = ' ← 최고' if ens_f1 == max(all_f1.values()) else ''
    print(f"  {'TF-IDF+BERT 앙상블':<24} {report_ens['accuracy']:>8.4f}"
          f"  {ens_f1:>10.4f}{best_mark}")
    print(f"\n  최적 가중치: α={best_alpha:.2f} (TF-IDF) / {1-best_alpha:.2f} (BERT)")

    final_result = {
        'model'      : 'LateFusion_TFIDF_BERT',
        'best_alpha' : round(best_alpha, 2),
        'accuracy'   : round(report_ens['accuracy'], 4),
        'macro_f1'   : round(ens_f1, 4),
        'tfidf_f1'   : round(all_f1['TF-IDF 단독'], 4),
        'bert_f1'    : round(all_f1['BERT 단독'], 4),
        'report'     : report_ens,
    }

    path = os.path.join(RESULT_DIR, 'scores.json')
    data = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else []
    data.append(final_result)
    json.dump(data, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {path}")

    return final_result


if __name__ == '__main__':
    if not os.path.isdir(BERT_MODEL_DIR):
        print("❌ BERT 모델 없음 → python src/models/bert_sentiment.py 먼저 실행")
        exit(1)

    run_ensemble()
    print('\nTF-IDF + BERT 앙상블 완료')