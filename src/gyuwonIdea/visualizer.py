"""
visualizer.py
=============
gyuwonIdea 실험 전체 시각화 모듈.

생성 그래프 6종
────────────────
idea_01_f1_heatmap_tfidf.png  : TF-IDF 전략 × θ Macro-F1 히트맵
idea_02_f1_heatmap_bert.png   : BERT 전략 × θ Macro-F1 히트맵
idea_03_fastpath_ratio.png    : θ별 fast path 비율 꺾은선 (TF-IDF vs BERT)
idea_04_stability.png         : centroid 안정성 비교 막대
idea_05_speed.png             : 처리 속도 비교 (ms/건)
idea_06_tfidf_vs_bert.png     : 두 방식 종합 비교 (θ=0.5 기준)

각 함수는 독립적으로 호출 가능하다.
main.py 에서 BERT 실험을 스킵한 경우 idea_02/06 은 자동으로 생략된다.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import platform

from config import (FIG_DIR, THETA_LIST, STRATEGY_KEYS, STRATEGY_LABELS,
                    CAT_CLASSES, NUM_CATS)
from evaluator import extract_matrix

# ── 한국어 폰트 ───────────────────────────────────────────────────────────────
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

import os


# ──────────────────────────────────────────────────────────────────────────────
# 내부 유틸
# ──────────────────────────────────────────────────────────────────────────────

def _save(fig, filename: str):
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {path}')


def _f1_heatmap(records: list, method_label: str, filename: str):
    """
    전략 × θ Macro-F1 히트맵.

    행 = 샘플링 전략 (sequential/random/stratified)
    열 = θ 값 (0.3/0.5/0.7)
    셀 = Macro-F1 (전체 기준)

    히트맵으로 보는 것 :
      - 오른쪽 열(θ 높음)로 갈수록 F1이 높아지는가?
        → 높으면 "θ 가 높을수록 fast path 기사가 분류하기 쉬운 기사"라는 증거
      - 아래 행(stratified)이 전반적으로 진한가?
        → stratified 가 centroid 품질이 가장 좋다는 증거
    """
    # (3전략, 3θ) 행렬 구성
    f1_mat = np.zeros((len(STRATEGY_KEYS), len(THETA_LIST)))
    for r_i, rec in enumerate(records):
        for t_i, tr in enumerate(rec['theta_results']):
            f1_mat[r_i, t_i] = tr['macro_f1_all']

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(f1_mat, cmap='YlGn', vmin=0.4, vmax=1.0, aspect='auto')

    ax.set_xticks(range(len(THETA_LIST)))
    ax.set_yticks(range(len(STRATEGY_KEYS)))
    ax.set_xticklabels([f'θ={t}' for t in THETA_LIST], fontsize=11)
    ax.set_yticklabels(
        [STRATEGY_LABELS[s].replace('\n', ' ') for s in STRATEGY_KEYS],
        fontsize=10)
    ax.set_xlabel('임계값 θ (높을수록 fast path 기준 엄격)', fontsize=10)
    ax.set_ylabel('샘플링 전략', fontsize=10)
    ax.set_title(f'{method_label}\n전략 × θ 별 Macro-F1 히트맵', fontsize=12)

    for i in range(len(STRATEGY_KEYS)):
        for j in range(len(THETA_LIST)):
            val = f1_mat[i, j]
            ax.text(j, i, f'{val:.4f}',
                    ha='center', va='center', fontsize=11,
                    color='white' if val > 0.75 else 'black',
                    fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Macro-F1')
    plt.tight_layout()
    _save(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# 공개 함수
# ──────────────────────────────────────────────────────────────────────────────

def plot_f1_heatmap_tfidf(tfidf_records: list):
    """그래프 ① : TF-IDF 전략 × θ Macro-F1 히트맵"""
    _f1_heatmap(tfidf_records, 'TF-IDF Centroid', 'idea_01_f1_heatmap_tfidf.png')


def plot_f1_heatmap_bert(bert_records: list):
    """그래프 ② : BERT 전략 × θ Macro-F1 히트맵"""
    _f1_heatmap(bert_records, 'BERT Centroid', 'idea_02_f1_heatmap_bert.png')


def plot_fastpath_ratio(tfidf_records: list, bert_records: list = None):
    """
    그래프 ③ : θ별 fast path 비율 꺾은선 (정확도-속도 트레이드오프)

    x축 = θ 값, y축 = fast path 비율(%).
    전략별로 선을 그린다. TF-IDF 와 BERT 를 같은 그래프에 겹쳐 비교.

    이 그래프가 핵심 트레이드오프를 가장 명확하게 보여준다 :
      θ 가 낮으면 fast path 비율 높음(빠름) but 정확도 히트맵에서 F1 낮음
      θ 가 높으면 fast path 비율 낮음(느림) but F1 높음
    → 두 그래프를 나란히 보면 "어느 θ 에서 균형이 맞는가"가 드러남
    """
    colors_tfidf = ['#E74C3C', '#F39C12', '#3498DB']
    colors_bert  = ['#922B21', '#B7950B', '#1A5276']
    line_styles  = ['-', '--', '-.']

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (rec, color, ls) in enumerate(
            zip(tfidf_records, colors_tfidf, line_styles)):
        ratios = [tr['fast_ratio'] * 100 for tr in rec['theta_results']]
        label  = f'TF-IDF / {STRATEGY_LABELS[rec["strategy"]].replace(chr(10), " ")}'
        ax.plot(THETA_LIST, ratios, marker='o', color=color,
                linestyle=ls, linewidth=2, label=label)

    if bert_records:
        for i, (rec, color, ls) in enumerate(
                zip(bert_records, colors_bert, line_styles)):
            ratios = [tr['fast_ratio'] * 100 for tr in rec['theta_results']]
            label  = f'BERT / {STRATEGY_LABELS[rec["strategy"]].replace(chr(10), " ")}'
            ax.plot(THETA_LIST, ratios, marker='s', color=color,
                    linestyle=ls, linewidth=2, label=label, alpha=0.7)

    ax.set_xlabel('임계값 θ', fontsize=11)
    ax.set_ylabel('Fast Path 비율 (%)', fontsize=11)
    ax.set_title('θ별 Fast Path 비율\n(높을수록 BERT 재처리 없이 빠르게 처리)', fontsize=12)
    ax.set_xticks(THETA_LIST)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(alpha=0.3)

    # 기준선 : 현재 전체 BERT 처리(100% slow) = fast path 0%
    ax.axhline(0, color='black', linestyle=':', alpha=0.5, label='현재(전체 BERT)')

    plt.tight_layout()
    _save(fig, 'idea_03_fastpath_ratio.png')


def plot_stability(tfidf_records: list, bert_records: list = None):
    """
    그래프 ④ : centroid 안정성 비교 (카테고리별 평균 분산)

    낮을수록 샘플 의존성이 낮고 안정적인 전략.

    왼쪽 패널 : 전략별 전체 평균 분산 (TF-IDF vs BERT 나란히)
    오른쪽 패널 : 카테고리별 분산 히트맵 (소수 카테고리 불안정 확인)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 왼쪽 : 전체 평균 분산 막대
    strategies = [rec['strategy'] for rec in tfidf_records]
    tfidf_vars = [rec['stability']['mean_var'] for rec in tfidf_records]
    x = np.arange(len(strategies))
    w = 0.35

    bars_t = axes[0].bar(x - w/2, tfidf_vars, w, label='TF-IDF',
                         color='#3498DB', edgecolor='black', alpha=0.85)
    for bar, val in zip(bars_t, tfidf_vars):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + max(tfidf_vars) * 0.01,
                     f'{val:.5f}', ha='center', fontsize=8)

    if bert_records:
        bert_vars = [rec['stability']['mean_var'] for rec in bert_records]
        bars_b = axes[0].bar(x + w/2, bert_vars, w, label='BERT',
                             color='#E74C3C', edgecolor='black', alpha=0.85)
        for bar, val in zip(bars_b, bert_vars):
            axes[0].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + max(bert_vars) * 0.01,
                         f'{val:.5f}', ha='center', fontsize=8)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        [STRATEGY_LABELS[s].replace('\n', '\n') for s in strategies], fontsize=9)
    axes[0].set_ylabel('평균 분산 (낮을수록 안정)')
    axes[0].set_title('샘플링 전략별 Centroid 안정성\n(10회 반복 분산)', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)

    # 오른쪽 : 카테고리별 분산 히트맵 (TF-IDF 기준)
    # (3전략, 6카테고리) 행렬
    cat_var_mat = np.zeros((len(STRATEGY_KEYS), NUM_CATS))
    for s_i, rec in enumerate(tfidf_records):
        for c_i, cat in enumerate(CAT_CLASSES):
            cat_var_mat[s_i, c_i] = rec['stability']['per_cat_var'].get(cat, 0)

    im = axes[1].imshow(cat_var_mat, cmap='Reds', aspect='auto')
    axes[1].set_xticks(range(NUM_CATS))
    axes[1].set_yticks(range(len(STRATEGY_KEYS)))
    axes[1].set_xticklabels(CAT_CLASSES, rotation=30, ha='right', fontsize=9)
    axes[1].set_yticklabels(
        [STRATEGY_LABELS[s].replace('\n', ' ') for s in STRATEGY_KEYS], fontsize=9)
    axes[1].set_title('카테고리별 Centroid 분산 (TF-IDF)\n(진할수록 불안정)', fontsize=11)
    for i in range(len(STRATEGY_KEYS)):
        for j in range(NUM_CATS):
            val = cat_var_mat[i, j]
            axes[1].text(j, i, f'{val:.4f}', ha='center', va='center',
                         fontsize=7,
                         color='white' if val > cat_var_mat.max() * 0.6 else 'black')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle('Centroid 안정성 분석 (10회 반복 샘플링)', fontsize=13)
    plt.tight_layout()
    _save(fig, 'idea_04_stability.png')


def plot_speed(tfidf_records: list, bert_records: list = None):
    """
    그래프 ⑤ : 처리 속도 비교 (ms/건)

    θ=0.5 기준 전략별 평균 처리 속도.
    BERT centroid 방식은 임베딩 추출 시간을 포함한 전체 시간도 별도 표시.

    참고 기준선 :
      전체 BERT fine-tuning 처리 ≈ gyuwon_categoryNums.py 에서 측정된 속도
      (보고서 작성 시 수동으로 기입)
    """
    theta_idx = THETA_LIST.index(0.5) if 0.5 in THETA_LIST else 1

    fig, ax = plt.subplots(figsize=(10, 5))

    strategies = [rec['strategy'] for rec in tfidf_records]
    x = np.arange(len(strategies))
    w = 0.3

    tfidf_speeds = [rec['theta_results'][theta_idx]['elapsed_ms_per']
                    for rec in tfidf_records]
    bars_t = ax.bar(x - w/2 if bert_records else x, tfidf_speeds, w,
                    label='TF-IDF Centroid', color='#3498DB',
                    edgecolor='black', alpha=0.85)
    for bar, val in zip(bars_t, tfidf_speeds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001,
                f'{val:.3f}ms', ha='center', fontsize=9)

    if bert_records:
        bert_speeds = [rec['theta_results'][theta_idx]['elapsed_ms_per']
                       for rec in bert_records]
        bars_b = ax.bar(x + w/2, bert_speeds, w,
                        label='BERT Centroid (유사도만)', color='#E74C3C',
                        edgecolor='black', alpha=0.85)
        for bar, val in zip(bars_b, bert_speeds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001,
                    f'{val:.3f}ms', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [STRATEGY_LABELS[s].replace('\n', '\n') for s in strategies], fontsize=9)
    ax.set_ylabel('평균 처리 속도 (ms/건)')
    ax.set_title(f'전략별 처리 속도 비교 (θ=0.5 기준)\n'
                 f'(유사도 계산만 측정 — BERT 임베딩 추출 시간 별도)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save(fig, 'idea_05_speed.png')


def plot_tfidf_vs_bert(tfidf_records: list, bert_records: list):
    """
    그래프 ⑥ : TF-IDF vs BERT 종합 비교 (θ=0.5, stratified 기준)

    4개 지표를 나란히 막대로 비교 :
      Macro-F1 (전체) / Fast Path 비율 / Centroid 안정성 / 처리 속도
    스케일이 달라서 각 지표를 정규화 후 비교하는 레이더 차트도 추가.

    이 그래프가 "두 방식을 종합적으로 어떻게 볼 것인가"를
    발표에서 한 장으로 설명하는 핵심 슬라이드 재료가 된다.
    """
    theta_idx = THETA_LIST.index(0.5) if 0.5 in THETA_LIST else 1
    strat_idx = STRATEGY_KEYS.index('stratified')

    tr_tfidf = tfidf_records[strat_idx]['theta_results'][theta_idx]
    tr_bert  = bert_records[strat_idx]['theta_results'][theta_idx]
    st_tfidf = tfidf_records[strat_idx]['stability']['mean_var']
    st_bert  = bert_records[strat_idx]['stability']['mean_var']

    metrics = ['Macro-F1\n(전체)', 'Fast Path\n비율(%)', 'Centroid\n안정성(역수)', '속도\n(역수, ms/건)']
    # 안정성과 속도는 낮을수록 좋으므로 역수로 변환해 "높을수록 좋음"으로 통일
    vals_tfidf = [
        tr_tfidf['macro_f1_all'],
        tr_tfidf['fast_ratio'],
        1 / (st_tfidf + 1e-9),            # 안정성 역수
        1 / (tr_tfidf['elapsed_ms_per'] + 1e-9),  # 속도 역수
    ]
    vals_bert = [
        tr_bert['macro_f1_all'],
        tr_bert['fast_ratio'],
        1 / (st_bert + 1e-9),
        1 / (tr_bert['elapsed_ms_per'] + 1e-9),
    ]

    # 정규화 (각 지표 최대값 기준)
    max_vals = [max(a, b) for a, b in zip(vals_tfidf, vals_bert)]
    norm_t   = [v / m if m > 0 else 0 for v, m in zip(vals_tfidf, max_vals)]
    norm_b   = [v / m if m > 0 else 0 for v, m in zip(vals_bert,  max_vals)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 왼쪽 : 원본 수치 막대 (정규화 전, 각 지표 독립 y축)
    x = np.arange(4)
    w = 0.3
    axes[0].bar(x - w/2, vals_tfidf, w, label='TF-IDF / Stratified',
                color='#3498DB', edgecolor='black', alpha=0.85)
    axes[0].bar(x + w/2, vals_bert,  w, label='BERT / Stratified',
                color='#E74C3C', edgecolor='black', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, fontsize=9)
    axes[0].set_title('TF-IDF vs BERT 지표 비교 (θ=0.5, Stratified)', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('원본 값 (스케일 혼재 — 참고용)')

    # 오른쪽 : 정규화 후 레이더 대용 막대
    axes[1].bar(x - w/2, norm_t, w, label='TF-IDF (정규화)',
                color='#3498DB', edgecolor='black', alpha=0.85)
    axes[1].bar(x + w/2, norm_b, w, label='BERT (정규화)',
                color='#E74C3C', edgecolor='black', alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics, fontsize=9)
    axes[1].set_ylim(0, 1.2)
    axes[1].set_title('정규화 비교 (1.0 = 해당 지표 최고값)\n높을수록 좋음', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylabel('정규화 점수')

    plt.suptitle('TF-IDF vs BERT Centroid 종합 비교', fontsize=13)
    plt.tight_layout()
    _save(fig, 'idea_06_tfidf_vs_bert.png')


# ──────────────────────────────────────────────────────────────────────────────
# 일괄 실행 (main.py 에서 호출)
# ──────────────────────────────────────────────────────────────────────────────

def run_all_plots(tfidf_records: list, bert_records: list = None):
    """
    모든 시각화를 순서대로 생성한다.
    bert_records=None 이면 BERT 관련 그래프(②⑥)를 스킵한다.
    """
    print('\n[시각화 생성]')
    plot_f1_heatmap_tfidf(tfidf_records)

    if bert_records:
        plot_f1_heatmap_bert(bert_records)

    plot_fastpath_ratio(tfidf_records, bert_records)
    plot_stability(tfidf_records, bert_records)
    plot_speed(tfidf_records, bert_records)

    if bert_records:
        plot_tfidf_vs_bert(tfidf_records, bert_records)

    print('  시각화 완료')
