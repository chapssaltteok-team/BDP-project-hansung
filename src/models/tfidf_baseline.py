"""
tfidf_baseline.py
=================
TF-IDF 기반 KBO 뉴스 감성/카테고리 분류 베이스라인
담당: 김동환

역할
────
- BERT 대비 전통적 ML 베이스라인 제공
- 임동훈 외(2021) 딥러닝 다중분류 선행연구 비교 기준
- CPU만으로 실행 가능 (GPU 불필요)

분류기
──────
- Logistic Regression (로지스틱 회귀)
- LinearSVC (선형 SVM)
- MultinomialNB (나이브 베이즈)
→ 3개 분류기 성능 비교 후 최적 선택

평가 지표
──────────
- Accuracy
- Macro-F1 (BERT 비교용 핵심 지표)
- 클래스별 F1

출력
────
- results/scores.json (BERT scores.json과 동일 형식)
- results/figures/tfidf_confusion_matrix.png
- results/time_log.json (소요 시간)

실행: python src/models/tfidf_baseline.py
      (로컬 CPU에서 실행 권장, 약 2~3분 소요)
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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

PROC_DIR   = 'data/processed'
RESULT_DIR = 'results'
FIG_DIR    = os.path.join(RESULT_DIR, 'figures')
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)

# 감성 레이블
SENTIMENT_LABELS  = ['긍정', '중립', '부정']
# 카테고리 레이블 (v2)
CATEGORY_LABELS   = ['타격', '투구', '선수이동', '부상', '감독·운영', '기타']


# ── 소요 시간 측정 ────────────────────────────────────────────────────────────
class Timer:
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
        print(f"  ⏱ [{name}] 소요시간: {elapsed/60:.1f}분 ({elapsed:.1f}초)")
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)


# ── TF-IDF 파이프라인 구성 ────────────────────────────────────────────────────
def build_pipelines() -> dict:
    """
    TF-IDF + 분류기 파이프라인 3종

    TfidfVectorizer 설정
    ─────────────────────
    - ngram_range=(1,2): unigram + bigram
      한국어는 2글자 이상 형태소 단위가 의미있음
    - max_features=50000: 상위 5만 개 단어만 사용
    - sublinear_tf=True: TF에 log 적용 (빈도 편향 완화)
    - min_df=2: 2개 미만 문서에 등장하는 단어 제거
    """
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
        min_df=2,
        analyzer='char_wb',   # 한국어: 문자 단위 n-gram이 형태소보다 효과적
    )

    return {
        'LogisticRegression': Pipeline([
            ('tfidf', tfidf),
            ('clf',   LogisticRegression(
                max_iter=1000,
                C=1.0,
                multi_class='multinomial',
                solver='lbfgs',
                random_state=42,
            )),
        ]),
        'LinearSVC': Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=50000,
                sublinear_tf=True,
                min_df=2,
                analyzer='char_wb',
            )),
            ('clf',   LinearSVC(
                C=1.0,
                max_iter=2000,
                random_state=42,
            )),
        ]),
        'NaiveBayes': Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=50000,
                sublinear_tf=False,  # NB는 log 미적용
                min_df=2,
                analyzer='char_wb',
            )),
            ('clf',   MultinomialNB(alpha=0.1)),
        ]),
    }


# ── 혼동 행렬 시각화 ──────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, labels: list,
                          title: str, save_path: str):
    cm  = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(title, fontsize=13)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  혼동 행렬 저장: {save_path}")


# ── 분류기 비교 표 출력 ──────────────────────────────────────────────────────
def print_comparison_table(results: dict, task: str):
    print(f"\n  {'─'*50}")
    print(f"  {task} 분류기 비교")
    print(f"  {'분류기':<22} {'Accuracy':>10}  {'Macro-F1':>10}")
    print(f"  {'─'*50}")
    best_f1  = -1
    best_clf = ''
    for name, r in results.items():
        f1  = r['macro_f1']
        acc = r['accuracy']
        marker = ''
        if f1 > best_f1:
            best_f1  = f1
            best_clf = name
        print(f"  {name:<22} {acc:>10.4f}  {f1:>10.4f}")
    print(f"  {'─'*50}")
    print(f"  최적 분류기: {best_clf}  (Macro-F1={best_f1:.4f})")
    return best_clf


# ── 감성 분류 실험 ────────────────────────────────────────────────────────────
def run_sentiment(timer: Timer) -> dict:
    """
    TF-IDF 감성 분류 (긍정/중립/부정)
    → BERT 감성 분류와 Macro-F1 직접 비교용
    """
    timer.start('tfidf_sentiment')
    print(f"\n{'='*55}")
    print(f"  TF-IDF 감성 분류 베이스라인")
    print(f"{'='*55}")

    # 데이터 로드
    train_df = pd.read_csv(os.path.join(PROC_DIR, 'train.csv'), encoding='utf-8-sig')
    val_df   = pd.read_csv(os.path.join(PROC_DIR, 'val.csv'),   encoding='utf-8-sig')
    test_df  = pd.read_csv(os.path.join(PROC_DIR, 'test.csv'),  encoding='utf-8-sig')

    # train + val 합쳐서 학습 (TF-IDF는 GPU 불필요라 가능)
    train_all = pd.concat([train_df, val_df], ignore_index=True)

    X_train = train_all['text'].fillna('').tolist()
    y_train = train_all['sentiment_str'].tolist()
    X_test  = test_df['text'].fillna('').tolist()
    y_test  = test_df['sentiment_str'].tolist()

    print(f"  학습: {len(X_train)}건  테스트: {len(X_test)}건")

    pipelines = build_pipelines()
    results   = {}
    best_pipeline = None
    best_f1       = -1
    best_name     = ''

    for name, pipe in pipelines.items():
        t0 = time.time()
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        elapsed = time.time() - t0

        report = classification_report(
            y_test, y_pred,
            target_names=SENTIMENT_LABELS,
            output_dict=True, zero_division=0,
        )
        acc  = accuracy_score(y_test, y_pred)
        f1   = report['macro avg']['f1-score']

        results[name] = {
            'accuracy' : round(acc, 4),
            'macro_f1' : round(f1, 4),
            'elapsed_sec': round(elapsed, 2),
            'report'   : report,
        }
        print(f"  [{name}] Acc={acc:.4f}  Macro-F1={f1:.4f}  ({elapsed:.1f}초)")

        if f1 > best_f1:
            best_f1      = f1
            best_name    = name
            best_pipeline = pipe

    # 최적 분류기 비교표
    print_comparison_table(results, '감성')

    # 혼동 행렬 저장 (최적 분류기)
    y_pred_best = best_pipeline.predict(X_test)
    plot_confusion_matrix(
        y_test, y_pred_best,
        labels=SENTIMENT_LABELS,
        title=f'TF-IDF 감성 분류 혼동 행렬 ({best_name})',
        save_path=os.path.join(FIG_DIR, 'tfidf_sentiment_cm.png'),
    )

    timer.end('tfidf_sentiment')

    # scores.json 저장
    final = {
        'model'      : 'TF-IDF_Sentiment',
        'best_clf'   : best_name,
        'accuracy'   : results[best_name]['accuracy'],
        'macro_f1'   : results[best_name]['macro_f1'],
        'elapsed_min': timer.records['tfidf_sentiment']['elapsed_min'],
        'all_classifiers': results,
        'report'     : results[best_name]['report'],
    }
    _save_result(final)
    return final


# ── 카테고리 분류 실험 ────────────────────────────────────────────────────────
def run_category(timer: Timer) -> dict:
    """
    TF-IDF 카테고리 분류 (6종 v2 기준)
    → BERT 카테고리 분류와 Macro-F1 직접 비교용
    """
    timer.start('tfidf_category')
    print(f"\n{'='*55}")
    print(f"  TF-IDF 카테고리 분류 베이스라인")
    print(f"{'='*55}")

    train_df = pd.read_csv(os.path.join(PROC_DIR, 'train.csv'), encoding='utf-8-sig')
    val_df   = pd.read_csv(os.path.join(PROC_DIR, 'val.csv'),   encoding='utf-8-sig')
    test_df  = pd.read_csv(os.path.join(PROC_DIR, 'test.csv'),  encoding='utf-8-sig')

    # 카테고리 컬럼 결측 처리
    for df in [train_df, val_df, test_df]:
        df['category'] = df['category'].fillna('기타')

    train_all = pd.concat([train_df, val_df], ignore_index=True)

    X_train = train_all['text'].fillna('').tolist()
    y_train = train_all['category'].tolist()
    X_test  = test_df['text'].fillna('').tolist()
    y_test  = test_df['category'].tolist()

    # 실제 존재하는 카테고리만 추출
    actual_labels = sorted(set(y_test))
    print(f"  학습: {len(X_train)}건  테스트: {len(X_test)}건")
    print(f"  카테고리: {actual_labels}")

    pipelines = build_pipelines()
    results   = {}
    best_pipeline = None
    best_f1       = -1
    best_name     = ''

    for name, pipe in pipelines.items():
        t0 = time.time()
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        elapsed = time.time() - t0

        report = classification_report(
            y_test, y_pred,
            target_names=actual_labels,
            output_dict=True, zero_division=0,
        )
        acc = accuracy_score(y_test, y_pred)
        f1  = report['macro avg']['f1-score']

        results[name] = {
            'accuracy'   : round(acc, 4),
            'macro_f1'   : round(f1, 4),
            'elapsed_sec': round(elapsed, 2),
            'report'     : report,
        }
        print(f"  [{name}] Acc={acc:.4f}  Macro-F1={f1:.4f}  ({elapsed:.1f}초)")

        if f1 > best_f1:
            best_f1       = f1
            best_name     = name
            best_pipeline = pipe

    print_comparison_table(results, '카테고리')

    # 혼동 행렬 저장
    y_pred_best = best_pipeline.predict(X_test)
    plot_confusion_matrix(
        y_test, y_pred_best,
        labels=actual_labels,
        title=f'TF-IDF 카테고리 분류 혼동 행렬 ({best_name})',
        save_path=os.path.join(FIG_DIR, 'tfidf_category_cm.png'),
    )

    timer.end('tfidf_category')

    final = {
        'model'      : 'TF-IDF_Category',
        'best_clf'   : best_name,
        'accuracy'   : results[best_name]['accuracy'],
        'macro_f1'   : results[best_name]['macro_f1'],
        'elapsed_min': timer.records['tfidf_category']['elapsed_min'],
        'all_classifiers': results,
        'report'     : results[best_name]['report'],
    }
    _save_result(final)
    return final


# ── v1 vs v2 카테고리 비교 실험 (ablation) ───────────────────────────────────
def run_category_ablation(timer: Timer) -> dict:
    """
    카테고리 v1(8개) vs v2(6개) TF-IDF 성능 비교
    → 카테고리 통합이 성능에 미치는 영향 정량적 증명
    → 발표 핵심 포인트: 전처리가 성능에 직결됨
    """
    timer.start('tfidf_category_ablation')
    print(f"\n{'='*55}")
    print(f"  TF-IDF 카테고리 Ablation (v1 vs v2)")
    print(f"{'='*55}")

    train_df = pd.read_csv(os.path.join(PROC_DIR, 'train.csv'), encoding='utf-8-sig')
    val_df   = pd.read_csv(os.path.join(PROC_DIR, 'val.csv'),   encoding='utf-8-sig')
    test_df  = pd.read_csv(os.path.join(PROC_DIR, 'test.csv'),  encoding='utf-8-sig')

    train_all = pd.concat([train_df, val_df], ignore_index=True)
    X_train   = train_all['text'].fillna('').tolist()
    X_test    = test_df['text'].fillna('').tolist()

    ablation_results = {}

    for version in ['v1', 'v2']:
        col = 'category_v1' if version == 'v1' else 'category'

        # v1 컬럼 없으면 스킵
        if col not in train_all.columns or col not in test_df.columns:
            print(f"  [{version}] '{col}' 컬럼 없음 → 스킵")
            print(f"  → prepro.py 재실행하면 category_v1 컬럼 생성됨")
            continue

        y_train = train_all[col].fillna('기타').tolist()
        y_test  = test_df[col].fillna('기타').tolist()

        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2), max_features=50000,
                sublinear_tf=True, min_df=2, analyzer='char_wb',
            )),
            ('clf', LogisticRegression(
                max_iter=1000, C=1.0,
                multi_class='multinomial', solver='lbfgs', random_state=42,
            )),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        actual_labels = sorted(set(y_test))
        report = classification_report(
            y_test, y_pred,
            target_names=actual_labels,
            output_dict=True, zero_division=0,
        )
        acc = accuracy_score(y_test, y_pred)
        f1  = report['macro avg']['f1-score']

        ablation_results[version] = {
            'n_categories': len(actual_labels),
            'categories'  : actual_labels,
            'accuracy'    : round(acc, 4),
            'macro_f1'    : round(f1, 4),
        }
        print(f"  [{version}] 카테고리={len(actual_labels)}개  "
              f"Acc={acc:.4f}  Macro-F1={f1:.4f}")

    # 비교 결과 출력
    if 'v1' in ablation_results and 'v2' in ablation_results:
        diff = ablation_results['v2']['macro_f1'] - ablation_results['v1']['macro_f1']
        print(f"\n  v2 - v1 Macro-F1 차이: {diff:+.4f}")
        if diff > 0:
            print(f"  → 카테고리 통합이 성능 향상에 기여")
        else:
            print(f"  → 카테고리 통합 효과 미미 (추가 분석 필요)")

    timer.end('tfidf_category_ablation')

    final = {
        'model'   : 'TF-IDF_Category_Ablation',
        'results' : ablation_results,
        'elapsed_min': timer.records['tfidf_category_ablation']['elapsed_min'],
    }
    _save_result(final)
    return final


# ── 결과 저장 ─────────────────────────────────────────────────────────────────
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


# ── 메인 ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"\n{'='*55}")
    print(f"  TF-IDF 베이스라인 (로컬 CPU 실행)")
    print(f"  GPU 불필요 | 예상 소요: 2~3분")
    print(f"{'='*55}")

    timer = Timer()

    # 1. 감성 분류
    sentiment_result = run_sentiment(timer)

    # 2. 카테고리 분류 (v2)
    category_result = run_category(timer)

    # 3. 카테고리 ablation (v1 vs v2)
    ablation_result = run_category_ablation(timer)

    # 최종 요약
    print(f"\n{'='*55}")
    print(f"  TF-IDF 베이스라인 완료")
    print(f"{'='*55}")
    print(f"  감성 Macro-F1 : {sentiment_result['macro_f1']}")
    print(f"  카테고리 Macro-F1: {category_result['macro_f1']}")
    print(f"\n  → BERT 학습 후 results/scores.json에서 비교 가능")
    print(f"  → results/figures/에 혼동 행렬 저장됨")
