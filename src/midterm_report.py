"""
midterm_report.py
=================
중간발표용 결과 시각화
담당: 이윤서

생성 결과물
────────────
1. results/figures/midterm_01_model_comparison.png  모델별 성능 비교
2. results/figures/midterm_02_category_ablation.png 카테고리 v1 vs v2
3. results/figures/midterm_03_category_f1_detail.png 카테고리별 F1 상세
4. results/figures/midterm_04_sentiment_f1_detail.png 감성별 F1 상세
5. results/midterm_summary.json 최종 요약

실행: python src/midterm_report.py  (로컬 CPU 가능)
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import platform

# 한국어 폰트
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = 'results'
FIG_DIR    = os.path.join(RESULT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

COLORS = {
    'tfidf' : '#95A5A6',
    'fisa'  : '#3498DB',
    'bert'  : '#2ECC71',
    'resnet': '#E74C3C',
    'ensemble': '#9B59B6',
}


def load_scores() -> list:
    path = os.path.join(RESULT_DIR, 'scores.json')
    if not os.path.exists(path):
        print(f"  ❌ {path} 없음 → 실험 먼저 실행")
        return []
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def plot_model_comparison(scores: list):
    """모델별 Accuracy / Macro-F1 비교 막대그래프"""
    valid = [s for s in scores if 'accuracy' in s and 'macro_f1' in s
             and isinstance(s['accuracy'], float)]
    if not valid:
        print("  ⚠️ 유효한 결과 없음")
        return

    names = [s['model'] for s in valid]
    accs  = [s['accuracy'] for s in valid]
    f1s   = [s['macro_f1'] for s in valid]

    x   = np.arange(len(names))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(names)*2), 6))

    bars1 = ax.bar(x - w/2, accs, w, label='Accuracy', color='#3498DB', alpha=0.85, edgecolor='black')
    bars2 = ax.bar(x + w/2, f1s,  w, label='Macro-F1', color='#2ECC71', alpha=0.85, edgecolor='black')

    for bar in bars1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('_',' ') for n in names], rotation=30, ha='right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('모델별 성능 비교 (Accuracy / Macro-F1)', fontsize=13)
    ax.legend()
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.3, label='랜덤 기준선(0.5)')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'midterm_01_model_comparison.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✅ {path}")


def plot_category_ablation(scores: list):
    """카테고리 v1(8개) vs v2(6개) 비교"""
    v1_result = next((s for s in scores if 'v1' in s.get('model','').lower() and 'category' in s.get('model','').lower()), None)
    v2_result = next((s for s in scores if 'v2' in s.get('model','').lower() and 'category' in s.get('model','').lower()), None)

    # TF-IDF ablation에서 읽기
    ablation = next((s for s in scores if 'Ablation' in s.get('model','')), None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 왼쪽: Macro-F1 비교
    labels, values, colors = [], [], []
    if v1_result:
        labels.append('v1 (8개)'); values.append(v1_result['macro_f1']); colors.append('#E74C3C')
    if v2_result:
        labels.append('v2 (6개)'); values.append(v2_result['macro_f1']); colors.append('#2ECC71')

    if labels:
        bars = axes[0].bar(labels, values, color=colors, edgecolor='black', alpha=0.85)
        for bar in bars:
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                         f'{bar.get_height():.4f}', ha='center', fontsize=11)
        axes[0].set_ylim(0, 1.0)
        axes[0].set_title('카테고리 버전별 Macro-F1 비교', fontsize=12)
        axes[0].set_ylabel('Macro-F1')
        axes[0].grid(axis='y', alpha=0.3)
        if len(values) == 2:
            diff = values[1] - values[0]
            axes[0].annotate(f'차이: {diff:+.4f}',
                             xy=(0.5, max(values)+0.05), xycoords='data',
                             ha='center', fontsize=11,
                             color='green' if diff > 0 else 'red')
    else:
        axes[0].text(0.5, 0.5, '결과 없음\n(실험 완료 후 재실행)', ha='center', va='center',
                     transform=axes[0].transAxes, fontsize=12)
        axes[0].set_title('카테고리 버전별 Macro-F1 비교', fontsize=12)

    # 오른쪽: 카테고리 분포 비교
    try:
        comp_path = os.path.join(RESULT_DIR, 'category_version_comparison.json')
        if os.path.exists(comp_path):
            with open(comp_path, encoding='utf-8') as f:
                comp = json.load(f)
            v1_dist = comp['v1']['distribution']
            v2_dist = comp['v2']['distribution']
            all_cats = sorted(set(list(v1_dist.keys()) + list(v2_dist.keys())))
            v1_vals  = [v1_dist.get(c, 0) for c in all_cats]
            v2_vals  = [v2_dist.get(c, 0) for c in all_cats]
            x = np.arange(len(all_cats))
            axes[1].bar(x-0.2, v1_vals, 0.4, label='v1(8개)', color='#E74C3C', alpha=0.7, edgecolor='black')
            axes[1].bar(x+0.2, v2_vals, 0.4, label='v2(6개)', color='#2ECC71', alpha=0.7, edgecolor='black')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(all_cats, rotation=30, ha='right')
            axes[1].set_title('카테고리별 데이터 분포 비교', fontsize=12)
            axes[1].set_ylabel('건수')
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3)
    except Exception as e:
        axes[1].text(0.5, 0.5, f'분포 데이터 없음\n{e}', ha='center', va='center',
                     transform=axes[1].transAxes, fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'midterm_02_category_ablation.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✅ {path}")


def plot_category_f1_detail(scores: list):
    """BERT 카테고리 분류 - 카테고리별 F1 상세"""
    bert_cat = next((s for s in scores
                     if 'BERT_Category' in s.get('model','') and 'v2' in s.get('model','')
                     and 'report' in s), None)
    if not bert_cat:
        bert_cat = next((s for s in scores
                         if 'BERT_Category' in s.get('model','') and 'report' in s), None)
    if not bert_cat:
        print("  ⚠️ BERT 카테고리 결과 없음")
        return

    report  = bert_cat['report']
    cats    = [k for k in report.keys() if k not in ['accuracy','macro avg','weighted avg']]
    f1s     = [report[c]['f1-score'] for c in cats]
    counts  = [report[c]['support'] for c in cats]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#2ECC71' if f >= 0.7 else '#F39C12' if f >= 0.5 else '#E74C3C' for f in f1s]

    bars = axes[0].barh(cats, f1s, color=colors, edgecolor='black', alpha=0.85)
    for bar, f in zip(bars, f1s):
        axes[0].text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                     f'{f:.3f}', va='center', fontsize=10)
    axes[0].set_xlim(0, 1.1)
    axes[0].axvline(bert_cat['macro_f1'], color='blue', linestyle='--',
                    label=f"Macro-F1={bert_cat['macro_f1']:.3f}")
    axes[0].set_title('카테고리별 F1 (BERT v2)', fontsize=12)
    axes[0].set_xlabel('F1-score')
    axes[0].legend(); axes[0].grid(axis='x', alpha=0.3)

    axes[1].barh(cats, counts, color='#3498DB', edgecolor='black', alpha=0.85)
    for i, c in enumerate(counts):
        axes[1].text(c+5, i, str(int(c)), va='center', fontsize=10)
    axes[1].set_title('카테고리별 테스트 샘플 수', fontsize=12)
    axes[1].set_xlabel('건수')
    axes[1].grid(axis='x', alpha=0.3)

    plt.suptitle(f"BERT 카테고리 분류 상세 | Macro-F1={bert_cat['macro_f1']:.4f}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'midterm_03_category_f1_detail.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✅ {path}")


def plot_sentiment_f1_detail(scores: list):
    """감성 분류 모델별 클래스 F1 비교"""
    sentiment_models = [s for s in scores
                        if any(k in s.get('model','') for k in ['Sentiment','FISA','TF-IDF'])
                        and 'report' in s]
    if not sentiment_models:
        print("  ⚠️ 감성 분류 결과 없음")
        return

    classes = ['긍정', '중립', '부정']
    n_models = len(sentiment_models)
    x = np.arange(len(classes))
    w = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))
    color_list = ['#3498DB','#2ECC71','#E74C3C','#9B59B6','#F39C12']

    for i, s in enumerate(sentiment_models):
        report = s['report']
        f1s = []
        for cls in classes:
            if cls in report:
                f1s.append(report[cls]['f1-score'])
            else:
                f1s.append(0.0)
        bars = ax.bar(x + i*w - (n_models-1)*w/2, f1s, w,
                      label=s['model'].replace('_',' '),
                      color=color_list[i % len(color_list)], alpha=0.85, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('F1-score')
    ax.set_title('감성 분류 - 클래스별 F1 비교', fontsize=13)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'midterm_04_sentiment_f1_detail.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✅ {path}")


def save_summary(scores: list):
    """최종 요약 JSON 저장"""
    summary = {
        'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        'total_experiments': len(scores),
        'models': []
    }
    for s in scores:
        if 'accuracy' in s and isinstance(s['accuracy'], float):
            summary['models'].append({
                'model'    : s.get('model'),
                'accuracy' : s.get('accuracy'),
                'macro_f1' : s.get('macro_f1'),
                'elapsed_min': s.get('elapsed_min'),
            })

    path = os.path.join(RESULT_DIR, 'midterm_summary.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {path}")

    # 콘솔 출력
    print(f"\n  {'모델':<35} {'Macro-F1':>10}")
    print(f"  {'-'*48}")
    sorted_models = sorted(summary['models'], key=lambda x: x['macro_f1'] or 0, reverse=True)
    for m in sorted_models:
        print(f"  {m['model']:<35} {m['macro_f1']:>10.4f}")


if __name__ == '__main__':
    import pandas as pd

    print(f"\n{'='*55}")
    print(f"  중간발표 결과 시각화 생성")
    print(f"{'='*55}")

    scores = load_scores()
    if not scores:
        print("  scores.json 없음 → 실험 먼저 실행하세요")
        exit(1)

    print(f"  총 {len(scores)}개 실험 결과 로드")

    plot_model_comparison(scores)
    plot_category_ablation(scores)
    plot_category_f1_detail(scores)
    plot_sentiment_f1_detail(scores)
    save_summary(scores)

    print(f"\n✅ 시각화 완료 → {FIG_DIR}/")
