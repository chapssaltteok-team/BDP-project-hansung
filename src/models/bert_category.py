"""
bert_category.py
================
BERT 기반 KBO 뉴스 카테고리 분류
담당: 분류모델 담당자

카테고리 (v2): 타격 / 투구 / 선수이동 / 부상 / 감독·운영 / 기타
베이스 모델: klue/bert-base

실험
────
Exp-1  klue/bert-base, max_len=256, epochs=5 (기본)
Exp-2  klue/roberta-base 비교 (RoBERTa 계열)

추가 (v2)
─────────
- 체크포인트 저장: outputs/checkpoint_{label}_ep{N}/ (에폭마다)
- Timer: results/time_log.json 소요 시간 기록
- 카테고리 v2 기준 (6개) 적용

실행: python src/models/bert_category.py
"""
import os, sys, json, time, random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

PROC_DIR   = 'data/processed'
OUTPUT_DIR = 'outputs'
RESULT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

SEED   = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# v2 카테고리 (6개) — prepro.py v2와 동일하게 유지
CATEGORIES = ['타격', '투구', '선수이동', '부상', '감독·운영', '기타']
LABEL2ID   = {cat: i for i, cat in enumerate(CATEGORIES)}
ID2LABEL   = {i: cat for cat, i in LABEL2ID.items()}
NUM_LABELS = len(CATEGORIES)


# ── 소요 시간 측정 ────────────────────────────────────────────────────────────
class Timer:
    """작업별 소요시간 기록 → results/time_log.json"""
    def __init__(self, log_path: str = 'results/time_log.json'):
        self.log_path = log_path
        self.records  = {}
        if os.path.exists(log_path):
            with open(log_path, encoding='utf-8') as f:
                self.records = json.load(f)

    def start(self, name: str):
        self.records[name] = {'start': time.time(), 'end': None,
                              'elapsed_sec': None, 'elapsed_min': None}

    def end(self, name: str):
        if name not in self.records:
            return
        elapsed = time.time() - self.records[name]['start']
        self.records[name]['end']         = time.time()
        self.records[name]['elapsed_sec'] = round(elapsed, 2)
        self.records[name]['elapsed_min'] = round(elapsed / 60, 2)
        print(f"  ⏱ [{name}] 소요시간: {elapsed / 60:.1f}분 ({elapsed:.1f}초)")
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Dataset ───────────────────────────────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 256):
        self.texts     = df['text'].tolist()
        self.labels    = df['category'].map(LABEL2ID).tolist()
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
            'label'         : torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── 평가 ──────────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds_all, labels_all = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            lbl  = batch['label'].to(DEVICE)
            out  = model(ids, attention_mask=mask)
            loss = criterion(out.logits, lbl)
            total_loss  += loss.item()
            preds_all.extend(out.logits.argmax(-1).cpu().tolist())
            labels_all.extend(lbl.cpu().tolist())

    report = classification_report(
        labels_all, preds_all,
        target_names=CATEGORIES, output_dict=True, zero_division=0
    )
    return total_loss / len(loader), report


# ── 학습 ──────────────────────────────────────────────────────────────────────
def train_bert_category(
    model_name : str   = 'klue/bert-base',
    max_len    : int   = 256,
    epochs     : int   = 5,
    batch_size : int   = 16,
    lr         : float = 2e-5,
    patience   : int   = 3,
    tag        : str   = '',
) -> dict:

    set_seed(SEED)
    label = tag or model_name.split('/')[-1]
    timer = Timer()
    timer.start(f'bert_category_{label}')

    print(f"\n{'='*55}")
    print(f"  BERT 카테고리 분류  [{label}]")
    print(f"  모델: {model_name}  |  device: {DEVICE}")
    print(f"  카테고리 ({NUM_LABELS}개): {CATEGORIES}")
    print(f"{'='*55}")

    # ── 데이터 ──────────────────────────────────────────
    train_df = pd.read_csv(os.path.join(PROC_DIR, 'train.csv'), encoding='utf-8-sig')
    val_df   = pd.read_csv(os.path.join(PROC_DIR, 'val.csv'),   encoding='utf-8-sig')
    test_df  = pd.read_csv(os.path.join(PROC_DIR, 'test.csv'),  encoding='utf-8-sig')

    for df in [train_df, val_df, test_df]:
        df['category'] = df['category'].fillna('기타')

    print(f"  데이터: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(f"  카테고리 분포 (train):\n{train_df['category'].value_counts().to_string()}")

    # ── 모델 ─────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS,
        id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(DEVICE)

    train_loader = DataLoader(
        NewsDataset(train_df, tokenizer, max_len),
        batch_size=batch_size, shuffle=True
    )
    val_loader  = DataLoader(NewsDataset(val_df,  tokenizer, max_len), batch_size=batch_size)
    test_loader = DataLoader(NewsDataset(test_df, tokenizer, max_len), batch_size=batch_size)

    # ── 클래스 불균형 대응: 가중치 ─────────────────────────
    label_counts = (
        train_df['category'].map(LABEL2ID)
        .value_counts()
        .reindex(range(NUM_LABELS), fill_value=1)
        .sort_index()
    )
    class_weights = torch.tensor(
        [1.0 / count for count in label_counts.values],
        dtype=torch.float32
    ).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ── 옵티마이저 ───────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    best_val_loss = float('inf')
    best_state    = None
    patience_cnt  = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            lbl  = batch['label'].to(DEVICE)
            optimizer.zero_grad()
            out  = model(ids, attention_mask=mask)
            loss = criterion(out.logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        val_loss, val_report = evaluate(model, val_loader, criterion)
        val_f1 = val_report['macro avg']['f1-score']
        print(f"  Epoch {epoch:2d} | train_loss={train_loss/len(train_loader):.4f} "
              f"| val_loss={val_loss:.4f} | val_F1(macro)={val_f1:.4f}")

        # ── 체크포인트 저장 (에폭마다) ───────────────────
        ckpt_dir = os.path.join(OUTPUT_DIR, f'checkpoint_bert_category_{label}_ep{epoch}')
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"  체크포인트 저장: {ckpt_dir}/")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early Stopping (epoch {epoch})")
                break

    # ── 테스트 평가 ───────────────────────────────────────
    model.load_state_dict(best_state)
    _, test_report = evaluate(model, test_loader, criterion)
    test_f1  = test_report['macro avg']['f1-score']
    test_acc = test_report['accuracy']

    print(f"\n  [Test] Accuracy={test_acc:.4f}  Macro-F1={test_f1:.4f}")

    # ── 최종 모델 저장 ────────────────────────────────────
    save_dir = os.path.join(OUTPUT_DIR, f'bert_category_{label}')
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"  최종 모델 저장: {save_dir}/")

    timer.end(f'bert_category_{label}')

    result = {
        'model'      : f'BERT_Category_{label}',
        'accuracy'   : round(test_acc, 4),
        'macro_f1'   : round(test_f1, 4),
        'elapsed_min': timer.records[f'bert_category_{label}']['elapsed_min'],
        'report'     : test_report,
    }
    result_path = os.path.join(RESULT_DIR, 'scores.json')
    data = []
    if os.path.exists(result_path):
        with open(result_path, encoding='utf-8') as f:
            data = json.load(f)
    data.append(result)
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return result


if __name__ == '__main__':
    # Exp-1: klue/bert-base (기본)
    train_bert_category(model_name='klue/bert-base', tag='klue_base')

    # Exp-2: klue/roberta-base 비교
    # train_bert_category(model_name='klue/roberta-base', tag='klue_roberta')

    print('\nBERT 카테고리 분류 완료')