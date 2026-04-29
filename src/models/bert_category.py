"""
bert_category.py (Final)
========================
BERT 기반 KBO 뉴스 카테고리 분류
담당: 유범준

실험
────
Exp-1: klue/bert-base, v2(6개) 카테고리 (기본)
Exp-2: klue/bert-base, v1(8개) 카테고리 (ablation 비교용)
       → v1 vs v2 Macro-F1 비교로 카테고리화 효과 증명

추가
────
- 에폭마다 체크포인트 저장
- Timer 소요시간 기록
- 클래스 불균형 대응 가중치

실행 환경: RunPod GPU | 예상: 25~35분
실행: python src/models/bert_category.py
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

SEED   = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# v2 카테고리 (기본)
CATEGORIES_V2 = ['타격','투구','선수이동','부상','감독·운영','기타']
# v1 카테고리 (ablation)
CATEGORIES_V1 = ['타격','투구','트레이드','부상','FA','감독','심판','기타']


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
    def __init__(self, df, tokenizer, label2id, cat_col='category', max_len=256):
        self.texts     = df['text'].tolist()
        self.labels    = df[cat_col].map(label2id).fillna(label2id.get('기타',0)).astype(int).tolist()
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


def evaluate(model, loader, criterion, categories):
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
        target_names=categories, output_dict=True, zero_division=0)
    return total_loss / len(loader), report


def train_bert_category(
    model_name='klue/bert-base', max_len=256,
    epochs=5, batch_size=16, lr=2e-5, patience=3,
    version='v2', tag='',
) -> dict:
    """
    version: 'v2' (6개, 기본) 또는 'v1' (8개, ablation용)
    """
    set_seed(SEED)
    categories = CATEGORIES_V2 if version == 'v2' else CATEGORIES_V1
    cat_col    = 'category' if version == 'v2' else 'category_v1'
    label2id   = {cat: i for i, cat in enumerate(categories)}
    id2label   = {i: cat for cat, i in label2id.items()}
    num_labels = len(categories)
    label      = tag or f'{model_name.split("/")[-1]}_{version}'
    timer      = Timer()
    timer.start(f'bert_category_{label}')

    print(f"\n{'='*55}")
    print(f"  BERT 카테고리 분류 [{label}] | {version}({num_labels}개) | device: {DEVICE}")
    print(f"{'='*55}")

    train_df = pd.read_csv(os.path.join(PROC_DIR,'train.csv'), encoding='utf-8-sig')
    val_df   = pd.read_csv(os.path.join(PROC_DIR,'val.csv'),   encoding='utf-8-sig')
    test_df  = pd.read_csv(os.path.join(PROC_DIR,'test.csv'),  encoding='utf-8-sig')

    # category_v1 컬럼 없으면 에러 안내
    if cat_col not in train_df.columns:
        print(f"  ❌ '{cat_col}' 컬럼 없음 → prepro.py 재실행 필요")
        return {}

    for df in [train_df, val_df, test_df]:
        df[cat_col] = df[cat_col].fillna('기타')

    print(f"  데이터: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"  카테고리 분포:\n{train_df[cat_col].value_counts().to_string()}")

    tokenizer    = AutoTokenizer.from_pretrained(model_name)
    model        = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id).to(DEVICE)
    train_loader = DataLoader(NewsDataset(train_df,tokenizer,label2id,cat_col,max_len),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(NewsDataset(val_df,  tokenizer,label2id,cat_col,max_len), batch_size=batch_size)
    test_loader  = DataLoader(NewsDataset(test_df, tokenizer,label2id,cat_col,max_len), batch_size=batch_size)

    # 클래스 불균형 가중치
    counts  = train_df[cat_col].map(label2id).value_counts().reindex(range(num_labels), fill_value=1).sort_index()
    weights = torch.tensor([1.0/c for c in counts.values], dtype=torch.float32).to(DEVICE)
    criterion   = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)
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

        val_loss, val_report = evaluate(model, val_loader, criterion, categories)
        print(f"  Epoch {epoch:2d} | train={train_loss/len(train_loader):.4f} "
              f"| val={val_loss:.4f} | F1={val_report['macro avg']['f1-score']:.4f}")

        ckpt = os.path.join(OUTPUT_DIR, f'ckpt_bert_category_{label}_ep{epoch}')
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
    _, test_report = evaluate(model, test_loader, criterion, categories)
    test_f1, test_acc = test_report['macro avg']['f1-score'], test_report['accuracy']
    print(f"\n  [Test] Acc={test_acc:.4f} Macro-F1={test_f1:.4f}")

    save_dir = os.path.join(OUTPUT_DIR, f'bert_category_{label}')
    model.save_pretrained(save_dir); tokenizer.save_pretrained(save_dir)
    timer.end(f'bert_category_{label}')

    result = {
        'model'      : f'BERT_Category_{label}',
        'version'    : version,
        'n_categories': num_labels,
        'accuracy'   : round(test_acc,4),
        'macro_f1'   : round(test_f1,4),
        'elapsed_min': timer.records[f'bert_category_{label}']['elapsed_min'],
        'report'     : test_report,
    }
    path = os.path.join(RESULT_DIR, 'scores.json')
    data = json.load(open(path,encoding='utf-8')) if os.path.exists(path) else []
    data.append(result)
    json.dump(data, open(path,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
    return result


if __name__ == '__main__':
    # Exp-1: v2 (6개, 기본 - 계획서 기준)
    result_v2 = train_bert_category(model_name='klue/bert-base', version='v2', tag='klue_base_v2')

    # Exp-2: v1 (8개, ablation 비교용)
    result_v1 = train_bert_category(model_name='klue/bert-base', version='v1', tag='klue_base_v1')

    # 비교 출력
    if result_v1 and result_v2:
        diff = result_v2['macro_f1'] - result_v1['macro_f1']
        print(f"\n  카테고리 Ablation 결과:")
        print(f"  v1(8개) Macro-F1: {result_v1['macro_f1']}")
        print(f"  v2(6개) Macro-F1: {result_v2['macro_f1']}")
        print(f"  차이: {diff:+.4f} ({'v2 우세' if diff>0 else 'v1 우세'})")

    print('\nBERT 카테고리 분류 완료')
