"""
summarization.py
================
KoBART / T5 기반 KBO 뉴스 기사 요약
담당: 요약모델 담당자

모델 선택
─────────
- gogamza/kobart-summarization  (한국어 BART, 추천)
- eenzeenee/t5-base-korean-summarize (한국어 T5)

평가 지표: ROUGE-1, ROUGE-2, ROUGE-L

실험
────
Exp-1  KoBART, 추론 전용 (pretrained 그대로 사용)
Exp-2  KoBART Fine-tuning (제목을 요약 정답으로 사용)

실행: python src/models/summarization.py
"""
import os, sys, json, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

PROC_DIR   = 'data/processed'
OUTPUT_DIR = 'outputs'
RESULT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

SEED   = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── ROUGE 계산 ────────────────────────────────────────────────────────────────
def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """
    문자 n-gram 기반 ROUGE 계산
    (rouge_score 라이브러리 없을 경우 간이 계산)
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        r1, r2, rl = [], [], []
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            r1.append(scores['rouge1'].fmeasure)
            r2.append(scores['rouge2'].fmeasure)
            rl.append(scores['rougeL'].fmeasure)
        return {
            'rouge1': round(np.mean(r1), 4),
            'rouge2': round(np.mean(r2), 4),
            'rougeL': round(np.mean(rl), 4),
        }
    except ImportError:
        # 간이 ROUGE-1 (unigram overlap)
        r1_scores = []
        for pred, ref in zip(predictions, references):
            pred_set = set(pred.split())
            ref_set  = set(ref.split())
            if not ref_set:
                continue
            precision = len(pred_set & ref_set) / (len(pred_set) + 1e-8)
            recall    = len(pred_set & ref_set) / (len(ref_set)  + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            r1_scores.append(f1)
        score = round(np.mean(r1_scores), 4) if r1_scores else 0.0
        return {'rouge1': score, 'rouge2': 0.0, 'rougeL': score}


# ── Dataset ───────────────────────────────────────────────────────────────────
class SumDataset(Dataset):
    """
    본문(body_clean) → 제목(title_clean) 요약 학습
    """
    def __init__(self, df: pd.DataFrame, tokenizer,
                 src_max: int = 512, tgt_max: int = 64):
        self.sources = df['body_clean'].tolist()
        self.targets = df['title_clean'].tolist()
        self.tokenizer = tokenizer
        self.src_max   = src_max
        self.tgt_max   = tgt_max

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        src_enc = self.tokenizer(
            self.sources[idx],
            max_length=self.src_max,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        with self.tokenizer.as_target_tokenizer():
            tgt_enc = self.tokenizer(
                self.targets[idx],
                max_length=self.tgt_max,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )
        labels = tgt_enc['input_ids'].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100   # pad 무시

        return {
            'input_ids'     : src_enc['input_ids'].squeeze(0),
            'attention_mask': src_enc['attention_mask'].squeeze(0),
            'labels'        : labels,
        }


# ── Exp-1: 추론 전용 (Fine-tuning 없이) ──────────────────────────────────────
def run_inference_only(
    model_name: str = 'gogamza/kobart-summarization',
    n_samples : int = 100,
    max_input : int = 512,
    max_output: int = 64,
    tag       : str = 'kobart_inference',
) -> dict:
    """
    Fine-tuning 없이 pretrained 모델로 테스트셋 요약 후 ROUGE 측정
    """
    set_seed(SEED)
    print(f"\n{'='*55}")
    print(f"  요약 (추론 전용)  [{tag}]")
    print(f"  모델: {model_name}  |  device: {DEVICE}")
    print(f"{'='*55}")

    test_df  = pd.read_csv(os.path.join(PROC_DIR, 'test.csv'), encoding='utf-8-sig')
    test_df  = test_df.head(n_samples).dropna(subset=['body_clean', 'title_clean'])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    model.eval()

    predictions, references = [], []
    for _, row in test_df.iterrows():
        inputs = tokenizer(
            row['body_clean'],
            return_tensors='pt',
            max_length=max_input,
            truncation=True,
        ).to(DEVICE)
        with torch.no_grad():
            summary_ids = model.generate(
                **inputs,
                max_length=max_output,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
            )
        pred = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        predictions.append(pred)
        references.append(row['title_clean'])

    rouge = compute_rouge(predictions, references)
    print(f"  ROUGE-1: {rouge['rouge1']}  ROUGE-2: {rouge['rouge2']}  ROUGE-L: {rouge['rougeL']}")

    # 예시 출력
    print(f"\n  ── 예시 요약 (상위 3건) ──")
    for i in range(min(3, len(predictions))):
        print(f"  [원문 제목] {references[i]}")
        print(f"  [모델 요약] {predictions[i]}")
        print()

    result = {'model': f'SUM_{tag}', **rouge}
    _save_result(result)
    return result


# ── Exp-2: Fine-tuning ────────────────────────────────────────────────────────
def run_finetune(
    model_name : str   = 'gogamza/kobart-summarization',
    epochs     : int   = 3,
    batch_size : int   = 4,
    lr         : float = 2e-5,
    src_max    : int   = 512,
    tgt_max    : int   = 64,
    tag        : str   = 'kobart_ft',
) -> dict:
    """
    Seq2SeqTrainer를 이용한 KoBART Fine-tuning
    """
    set_seed(SEED)
    print(f"\n{'='*55}")
    print(f"  요약 Fine-tuning  [{tag}]")
    print(f"  모델: {model_name}  |  device: {DEVICE}")
    print(f"{'='*55}")

    train_df = pd.read_csv(os.path.join(PROC_DIR, 'train.csv'), encoding='utf-8-sig')
    val_df   = pd.read_csv(os.path.join(PROC_DIR, 'val.csv'),   encoding='utf-8-sig')
    test_df  = pd.read_csv(os.path.join(PROC_DIR, 'test.csv'),  encoding='utf-8-sig')

    for df in [train_df, val_df, test_df]:
        df.dropna(subset=['body_clean', 'title_clean'], inplace=True)

    print(f"  데이터: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)

    train_dataset = SumDataset(train_df, tokenizer, src_max, tgt_max)
    val_dataset   = SumDataset(val_df,   tokenizer, src_max, tgt_max)
    test_dataset  = SumDataset(test_df,  tokenizer, src_max, tgt_max)

    save_dir = os.path.join(OUTPUT_DIR, f'sum_{tag}')
    args = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        logging_steps=50,
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    trainer  = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train()

    # ── 테스트 ROUGE ─────────────────────────────────────
    pred_output = trainer.predict(test_dataset, max_length=tgt_max, num_beams=4)
    decoded_preds = tokenizer.batch_decode(
        pred_output.predictions, skip_special_tokens=True)
    decoded_refs  = test_df['title_clean'].tolist()

    rouge = compute_rouge(decoded_preds, decoded_refs)
    print(f"  [Test] ROUGE-1={rouge['rouge1']}  "
          f"ROUGE-2={rouge['rouge2']}  ROUGE-L={rouge['rougeL']}")

    result = {'model': f'SUM_{tag}', **rouge}
    _save_result(result)
    return result


def _save_result(result: dict):
    path = os.path.join(RESULT_DIR, 'scores.json')
    data = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    data.append(result)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  결과 저장: {path}")


if __name__ == '__main__':
    # Exp-1: 추론 전용
    run_inference_only(
        model_name='gogamza/kobart-summarization',
        n_samples=100,
        tag='kobart_inference',
    )

    # Exp-2: Fine-tuning (GPU 환경 권장)
    # run_finetune(
    #     model_name='gogamza/kobart-summarization',
    #     epochs=3,
    #     tag='kobart_ft',
    # )

    print('\n요약 모델 완료')
