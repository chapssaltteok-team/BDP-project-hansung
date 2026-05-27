"""
sentiment_analysis_validation.py
=================================
감성 분석 실용성 검증 실험
담당: 이윤서

실험 내용
─────────
1. 감성 라벨 품질 검증 - 자동 라벨 샘플 추출 및 통계
2. 카테고리별 감성 분포 - 카테고리와 감성의 연관성 분석
3. 중립 클래스 제거 실험 - 긍정/부정 2분류 성능 비교

실행: python src/sentiment_analysis_validation.py
"""

import os, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

PROC_DIR   = 'data/processed'
RESULT_DIR = 'results'
FIG_DIR    = os.path.join(RESULT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── 한글 폰트 설정 ─────────────────────────────────────────────────────────────
def set_korean_font():
    font_candidates = [
        '/System/Library/Fonts/AppleSDGothicNeo.ttc',
        '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
    ]
    for path in font_candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()


# ════════════════════════════════════════════════════════════════
# 1. 감성 라벨 품질 검증
# ════════════════════════════════════════════════════════════════

def validate_sentiment_labels():
    """
    자동 라벨링된 감성 레이블의 품질 분석
    - 클래스 분포 확인
    - 경계 사례(threshold 근처) 비율 확인
    - 샘플 기사 출력
    """
    print(f"\n{'='*55}")
    print("  1. 감성 라벨 품질 검증")
    print(f"{'='*55}")

    train_df = pd.read_csv(os.path.join(PROC_DIR, 'train.csv'), encoding='utf-8-sig')
    val_df   = pd.read_csv(os.path.join(PROC_DIR, 'val.csv'),   encoding='utf-8-sig')
    test_df  = pd.read_csv(os.path.join(PROC_DIR, 'test.csv'),  encoding='utf-8-sig')
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print(f"\n  전체 데이터: {len(df)}건")

    # 감성 분포
    dist = df['sentiment_str'].value_counts()
    print(f"\n  감성 분포:")
    for label, count in dist.items():
        print(f"    {label}: {count}건 ({count/len(df)*100:.1f}%)")

    # 클래스 불균형 확인
    max_count = dist.max()
    min_count = dist.min()
    imbalance_ratio = max_count / min_count
    print(f"\n  클래스 불균형 비율: {imbalance_ratio:.2f} (최다/최소)")
    if imbalance_ratio > 2:
        print("  ⚠️  클래스 불균형 심각 → 중립 클래스 과다 가능성")

    # 감성별 샘플 출력
    print(f"\n  감성별 샘플 기사 (각 3건):")
    for label in ['긍정', '중립', '부정']:
        samples = df[df['sentiment_str'] == label]['text'].head(3)
        print(f"\n  [{label}]")
        for i, text in enumerate(samples, 1):
            print(f"    {i}. {str(text)[:80]}...")

    # 파이 차트 저장
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ['#2196F3', '#9E9E9E', '#F44336']
    ax.pie(dist.values, labels=dist.index, autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax.set_title('감성 라벨 분포')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'sentiment_label_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  파이 차트 저장: {path}")

    return dist.to_dict()


# ════════════════════════════════════════════════════════════════
# 2. 카테고리별 감성 분포
# ════════════════════════════════════════════════════════════════

def analyze_category_sentiment():
    """
    카테고리별 감성 분포 분석
    - 카테고리와 감성이 얼마나 연관있는지 확인
    - 특정 카테고리가 특정 감성에 편중되는지 분석
    """
    print(f"\n{'='*55}")
    print("  2. 카테고리별 감성 분포 분석")
    print(f"{'='*55}")

    train_df = pd.read_csv(os.path.join(PROC_DIR, 'train.csv'), encoding='utf-8-sig')
    val_df   = pd.read_csv(os.path.join(PROC_DIR, 'val.csv'),   encoding='utf-8-sig')
    test_df  = pd.read_csv(os.path.join(PROC_DIR, 'test.csv'),  encoding='utf-8-sig')
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    if 'category' not in df.columns:
        print("  ❌ 'category' 컬럼 없음 → prepro.py 재실행 필요")
        return

    # 카테고리별 감성 분포
    cross = pd.crosstab(df['category'], df['sentiment_str'], normalize='index') * 100
    print(f"\n  카테고리별 감성 비율 (%):")
    print(cross.round(1).to_string())

    # 카테고리별 감성 분포 시각화
    fig, ax = plt.subplots(figsize=(12, 6))
    cross.plot(kind='bar', ax=ax, color=['#2196F3', '#9E9E9E', '#F44336'])
    ax.set_title('카테고리별 감성 분포')
    ax.set_xlabel('카테고리')
    ax.set_ylabel('비율 (%)')
    ax.legend(title='감성')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'category_sentiment_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  카테고리별 감성 분포 차트 저장: {path}")

    # 카테고리-감성 연관성 분석
    print(f"\n  카테고리별 주요 감성:")
    for cat in cross.index:
        dominant = cross.loc[cat].idxmax()
        ratio = cross.loc[cat].max()
        print(f"    {cat}: {dominant} ({ratio:.1f}%)")

    return cross.to_dict()


# ════════════════════════════════════════════════════════════════
# 3. 중립 클래스 제거 실험
# ════════════════════════════════════════════════════════════════

def experiment_binary_classification():
    """
    중립 클래스 제거 후 긍정/부정 2분류 성능 비교
    - 3분류(긍정/중립/부정) vs 2분류(긍정/부정) 성능 비교
    - 중립 제거 시 성능 향상 여부 확인
    """
    print(f"\n{'='*55}")
    print("  3. 중립 클래스 제거 실험 (3분류 vs 2분류)")
    print(f"{'='*55}")

    train_df = pd.read_csv(os.path.join(PROC_DIR, 'train.csv'), encoding='utf-8-sig')
    val_df   = pd.read_csv(os.path.join(PROC_DIR, 'val.csv'),   encoding='utf-8-sig')
    test_df  = pd.read_csv(os.path.join(PROC_DIR, 'test.csv'),  encoding='utf-8-sig')

    def build_pipe():
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2), max_features=50000,
                sublinear_tf=True, min_df=2, analyzer='char_wb')),
            ('clf', LinearSVC(C=1.0, max_iter=2000, random_state=42))
        ])

    results = {}

    # ── 3분류 (기존) ──────────────────────────────────────────
    print("\n  [3분류] 긍정/중립/부정")
    train_all = pd.concat([train_df, val_df], ignore_index=True)
    pipe3 = build_pipe()
    pipe3.fit(train_all['text'].fillna(''), train_all['sentiment_str'])
    y_pred3 = pipe3.predict(test_df['text'].fillna(''))
    report3 = classification_report(
        test_df['sentiment_str'], y_pred3,
        target_names=['긍정', '중립', '부정'],
        output_dict=True, zero_division=0
    )
    acc3 = accuracy_score(test_df['sentiment_str'], y_pred3)
    f1_3 = report3['macro avg']['f1-score']
    print(f"  Acc={acc3:.4f}  Macro-F1={f1_3:.4f}")
    print(f"  긍정 F1={report3['긍정']['f1-score']:.4f}")
    print(f"  중립 F1={report3['중립']['f1-score']:.4f}")
    print(f"  부정 F1={report3['부정']['f1-score']:.4f}")
    results['3분류'] = {'accuracy': round(acc3, 4), 'macro_f1': round(f1_3, 4), 'report': report3}

    # ── 2분류 (중립 제거) ──────────────────────────────────────
    print("\n  [2분류] 긍정/부정 (중립 제거)")
    train_bin = pd.concat([train_df, val_df], ignore_index=True)
    train_bin = train_bin[train_bin['sentiment_str'] != '중립']
    test_bin  = test_df[test_df['sentiment_str'] != '중립']

    print(f"  2분류 train: {len(train_bin)}건 / test: {len(test_bin)}건")

    pipe2 = build_pipe()
    pipe2.fit(train_bin['text'].fillna(''), train_bin['sentiment_str'])
    y_pred2 = pipe2.predict(test_bin['text'].fillna(''))
    report2 = classification_report(
        test_bin['sentiment_str'], y_pred2,
        target_names=['긍정', '부정'],
        output_dict=True, zero_division=0
    )
    acc2 = accuracy_score(test_bin['sentiment_str'], y_pred2)
    f1_2 = report2['macro avg']['f1-score']
    print(f"  Acc={acc2:.4f}  Macro-F1={f1_2:.4f}")
    print(f"  긍정 F1={report2['긍정']['f1-score']:.4f}")
    print(f"  부정 F1={report2['부정']['f1-score']:.4f}")
    results['2분류'] = {'accuracy': round(acc2, 4), 'macro_f1': round(f1_2, 4), 'report': report2}

    # ── 비교 결과 ──────────────────────────────────────────────
    diff = f1_2 - f1_3
    print(f"\n  {'='*45}")
    print(f"  비교 결과")
    print(f"  {'='*45}")
    print(f"  3분류 Macro-F1: {f1_3:.4f}")
    print(f"  2분류 Macro-F1: {f1_2:.4f}")
    print(f"  차이: {diff:+.4f} ({'2분류 우세' if diff > 0 else '3분류 우세'})")

    if diff > 0:
        print(f"\n  → 중립 제거 시 성능 향상: 감성 분석 활용 시 2분류 권장")
    else:
        print(f"\n  → 중립 포함해도 성능 차이 미미: 3분류 유지 또는 감성 분석 제외 검토")

    # ── 비교 차트 ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    labels  = ['3분류\n(긍정/중립/부정)', '2분류\n(긍정/부정)']
    f1_vals = [f1_3, f1_2]
    colors  = ['#2196F3', '#4CAF50']
    bars = ax.bar(labels, f1_vals, color=colors, width=0.4)
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylim(0.6, max(f1_vals) + 0.05)
    ax.set_ylabel('Macro-F1')
    ax.set_title('3분류 vs 2분류 성능 비교 (TF-IDF)')
    ax.axhline(y=f1_3, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'binary_vs_ternary_classification.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  비교 차트 저장: {path}")

    return results


# ════════════════════════════════════════════════════════════════
# 결과 저장
# ════════════════════════════════════════════════════════════════

def save_results(label_dist, category_sentiment, binary_results):
    result = {
        'experiment': 'sentiment_validation',
        'label_distribution': label_dist,
        'binary_vs_ternary': {
            '3분류_macro_f1': binary_results['3분류']['macro_f1'],
            '2분류_macro_f1': binary_results['2분류']['macro_f1'],
            'diff': round(binary_results['2분류']['macro_f1'] - binary_results['3분류']['macro_f1'], 4),
        }
    }
    path = os.path.join(RESULT_DIR, 'sentiment_validation_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {path}")


# ════════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"\n{'='*55}")
    print("  감성 분석 실용성 검증 실험")
    print(f"{'='*55}")

    # 1. 감성 라벨 품질 검증
    label_dist = validate_sentiment_labels()

    # 2. 카테고리별 감성 분포
    category_sentiment = analyze_category_sentiment()

    # 3. 중립 클래스 제거 실험
    binary_results = experiment_binary_classification()

    # 결과 저장
    save_results(label_dist, category_sentiment, binary_results)

    print(f"\n{'='*55}")
    print("  감성 분석 실용성 검증 완료")
    print(f"  그래프: results/figures/")
    print(f"  결과: results/sentiment_validation_results.json")
    print(f"{'='*55}")
