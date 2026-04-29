"""
bert_sentiment.py (Final)
=========================
BERT 기반 KBO 뉴스 감성 분류 (긍정/중립/부정)
담당: 유범준

실험
────
Exp-1: klue/bert-base, epochs=5, lr=2e-5 (계획서 기준)
Exp-2: snunlp/KR-FinBert-SC (주석, 필요시 활성화)

추가
────
- 에폭마다 체크포인트 저장 (RunPod 세션 끊김 대비)
- Timer 소요시간 기록
- 추론 속도 측정

실행 환경: RunPod GPU | 예상: 25~35분
실행: python src/models/bert_sentiment.py
"""
import os, sys, json, time, random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

PROC_DIR   = 'data/processed'
OUTPUT_DIR = 'outputs'
RESULT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

SEED       = 42
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABEL2ID   = {'긍정': 0, '중립': 1, '부정': 2}
ID2LABEL   = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 3


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
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class NewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.texts     = df['text'].tolist()
        self.labels    = df['sentiment_str'].map(LABEL2ID).tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_len,
            truncation=True, padding='max_length', return_tensors='pt')
        return {
            'input_ids'     : enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label'         : torch.tensor(self.labels[idx], dtype=torch.long),
        }


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
            total_loss += loss.item()
            preds_all.extend(out.logits.argmax(-1).cpu().tolist())
            labels_all.extend(lbl.cpu().tolist())
    report = classification_report(
        labels_all, preds_all,
        target_names=list(LABEL2ID.keys()), output_dict=True, zero_division=0)
    return total_loss / len(loader), report


def train_bert_sentiment(
    model_name='klue/bert-base', max_len=256,
    epochs=5, batch_size=16, lr=2e-5, patience=3, tag='',
) -> dict:
    set_seed(SEED)
    label = tag or model_name.split('/')[-1]
    timer = Timer()
    timer.start(f'bert_sentiment_{label}')

    print(f"\n{'='*55}")
    print(f"  BERT 감성 분류 [{label}] | device: {DEVICE}")
    print(f"{'='*55}")

    train_df = pd.read_csv(os.path.join(PROC_DIR,'train.csv'), encoding='utf-8-sig')
    val_df   = pd.read_csv(os.path.join(PROC_DIR,'val.csv'),   encoding='utf-8-sig')
    test_df  = pd.read_csv(os.path.join(PROC_DIR,'test.csv'),  encoding='utf-8-sig')
    print(f"  데이터: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    tokenizer    = AutoTokenizer.from_pretrained(model_name)
    model        = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID).to(DEVICE)
    train_loader = DataLoader(NewsDataset(train_df,tokenizer,max_len), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(NewsDataset(val_df,  tokenizer,max_len), batch_size=batch_size)
    test_loader  = DataLoader(NewsDataset(test_df, tokenizer,max_len), batch_size=batch_size)

    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)
    criterion   = torch.nn.CrossEntropyLoss()
    best_val_loss, best_state, patience_cnt = float('inf'), None, 0

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            lbl  = batch['label'].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(ids, attention_mask=mask).logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            train_loss += loss.item()

        val_loss, val_report = evaluate(model, val_loader, criterion)
        print(f"  Epoch {epoch:2d} | train={train_loss/len(train_loader):.4f} "
              f"| val={val_loss:.4f} | F1={val_report['macro avg']['f1-score']:.4f}")

        # 체크포인트 저장
        ckpt = os.path.join(OUTPUT_DIR, f'ckpt_bert_sentiment_{label}_ep{epoch}')
        model.save_pretrained(ckpt); tokenizer.save_pretrained(ckpt)
        print(f"  체크포인트: {ckpt}/")

        if val_loss < best_val_loss:
            best_val_loss = val_loss; patience_cnt = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early Stopping (epoch {epoch})"); break

    model.load_state_dict(best_state)
    _, test_report = evaluate(model, test_loader, criterion)
    test_f1, test_acc = test_report['macro avg']['f1-score'], test_report['accuracy']

    model.eval()
    sample_batch = next(iter(test_loader))  # 한 번만 iter 생성
    s_ids  = sample_batch['input_ids'][:1].to(DEVICE)
    s_mask = sample_batch['attention_mask'][:1].to(DEVICE)
    with torch.no_grad():
        for _ in range(10): model(s_ids, attention_mask=s_mask)
        t0 = time.time()
        for _ in range(200): model(s_ids, attention_mask=s_mask)
    infer_ms = (time.time()-t0)/200*1000

    print(f"\n  [Test] Acc={test_acc:.4f} Macro-F1={test_f1:.4f} 추론={infer_ms:.2f}ms/sample")

    save_dir = os.path.join(OUTPUT_DIR, f'bert_sentiment_{label}')
    model.save_pretrained(save_dir); tokenizer.save_pretrained(save_dir)
    timer.end(f'bert_sentiment_{label}')

    result = {
        'model': f'BERT_Sentiment_{label}',
        'accuracy': round(test_acc,4), 'macro_f1': round(test_f1,4),
        'inference_ms': round(infer_ms,4),
        'elapsed_min': timer.records[f'bert_sentiment_{label}']['elapsed_min'],
        'report': test_report,
    }
    path = os.path.join(RESULT_DIR, 'scores.json')
    data = json.load(open(path,encoding='utf-8')) if os.path.exists(path) else []
    data.append(result)
    json.dump(data, open(path,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
    return result


if __name__ == '__main__':
    train_bert_sentiment(model_name='klue/bert-base', tag='klue_base')
    # train_bert_sentiment(model_name='snunlp/KR-FinBert-SC', tag='kr_finbert')
    print('\nBERT 감성 분류 완료')
