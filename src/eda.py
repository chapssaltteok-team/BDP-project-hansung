"""
eda.py
======
탐색적 데이터 분석
담당: 전처리+EDA 담당자

분석 항목
─────────
1. 데이터 규모 요약 (출처별, 날짜별)
2. 감성 레이블 분포
3. 카테고리 레이블 분포
4. 기사 길이 분포
5. 날짜별 뉴스 건수 추이
6. 워드클라우드 (감성별)
7. 키워드 Top-30 빈도

실행: python src/eda.py
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter
import re



import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'   # 맑은 고딕 (Windows 기본 한글 폰트)
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'     # Mac
else:
    plt.rcParams['font.family'] = 'NanumGothic'     # Linux
plt.rcParams['axes.unicode_minus'] = False

PROC_DIR = 'data/processed'
FIG_DIR  = 'results/figures'
os.makedirs(FIG_DIR, exist_ok=True)


# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def load_all() -> pd.DataFrame:
    dfs = []
    for split in ['train', 'val', 'test']:
        path = os.path.join(PROC_DIR, f'{split}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, encoding='utf-8-sig')
            df['split'] = split
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError("전처리된 CSV가 없습니다. preprocessing.py를 먼저 실행하세요.")
    return pd.concat(dfs, ignore_index=True)


# ── 1. 기본 요약 ──────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    print('=' * 55)
    print('  KBO 뉴스 EDA 요약')
    print('=' * 55)
    print(f'  총 기사 수   : {len(df):,}건')
    print(f'  출처별 분포:\n{df["source"].value_counts().to_string()}')
    print(f'\n  분할 분포:\n{df["split"].value_counts().to_string()}')
    print(f'\n  날짜 범위: {df["date"].min()} ~ {df["date"].max()}')
    print(f'  평균 본문 길이: {df["body_clean"].str.len().mean():.0f}자')


# ── 2. 감성/카테고리 분포 ─────────────────────────────────────────────────────
def plot_label_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 감성 분포
    sent_counts = df['sentiment_str'].value_counts()
    colors_sent = ['#4CAF50', '#9E9E9E', '#F44336']
    axes[0].bar(sent_counts.index, sent_counts.values,
                color=colors_sent[:len(sent_counts)], edgecolor='black', alpha=0.85)
    axes[0].set_title('감성 레이블 분포', fontsize=13)
    axes[0].set_xlabel('감성')
    axes[0].set_ylabel('기사 수')
    for i, v in enumerate(sent_counts.values):
        axes[0].text(i, v + 5, str(v), ha='center', fontsize=10)

    # 카테고리 분포
    cat_counts = df['category'].value_counts()
    axes[1].barh(cat_counts.index, cat_counts.values,
                 color='steelblue', edgecolor='black', alpha=0.85)
    axes[1].set_title('카테고리 레이블 분포', fontsize=13)
    axes[1].set_xlabel('기사 수')
    for i, v in enumerate(cat_counts.values):
        axes[1].text(v + 1, i, str(v), va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'eda_01_label_distribution.png'), dpi=150)
    plt.close()
    print('  → eda_01_label_distribution.png 저장')


# ── 3. 기사 길이 분포 ─────────────────────────────────────────────────────────
def plot_text_length(df: pd.DataFrame):
    df['body_len'] = df['body_clean'].str.len()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # 전체 분포
    axes[0].hist(df['body_len'], bins=40, color='steelblue',
                 edgecolor='black', alpha=0.8)
    axes[0].axvline(df['body_len'].mean(), color='red', linestyle='--',
                    label=f'평균 {df["body_len"].mean():.0f}자')
    axes[0].set_title('기사 본문 길이 분포 (전체)')
    axes[0].set_xlabel('글자 수')
    axes[0].set_ylabel('기사 수')
    axes[0].legend()

    # 감성별 박스플롯
    sentiments = ['긍정', '중립', '부정']
    data_by_sent = [df[df['sentiment_str'] == s]['body_len'].values
                    for s in sentiments if s in df['sentiment_str'].values]
    axes[1].boxplot(data_by_sent,
                    labels=[s for s in sentiments if s in df['sentiment_str'].values])
    axes[1].set_title('감성별 기사 길이 분포')
    axes[1].set_xlabel('감성')
    axes[1].set_ylabel('글자 수')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'eda_02_text_length.png'), dpi=150)
    plt.close()
    print('  → eda_02_text_length.png 저장')


# ── 4. 날짜별 뉴스 건수 추이 ──────────────────────────────────────────────────
def plot_daily_count(df: pd.DataFrame):
    df['date_str'] = df['date'].astype(str)
    daily = df.groupby('date_str').size().reset_index(name='count')
    daily = daily.sort_values('date_str')

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(range(len(daily)), daily['count'],
            color='steelblue', linewidth=1.5)
    ax.fill_between(range(len(daily)), daily['count'],
                    alpha=0.2, color='steelblue')
    # x축 레이블 간격 조정
    step = max(1, len(daily) // 20)
    ax.set_xticks(range(0, len(daily), step))
    ax.set_xticklabels(daily['date_str'].iloc[::step], rotation=45, ha='right')
    ax.set_title('날짜별 KBO 뉴스 수집 건수')
    ax.set_xlabel('날짜')
    ax.set_ylabel('기사 수')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'eda_03_daily_count.png'), dpi=150)
    plt.close()
    print('  → eda_03_daily_count.png 저장')


# ── 5. 키워드 Top-30 ──────────────────────────────────────────────────────────
def plot_top_keywords(df: pd.DataFrame):
    stop_words = {
        '기자', '뉴스', '스포츠', '야구', 'KBO', '리그', '시즌',
        '경기', '팀', '이날', '이번', '지난', '올해', '지난해',
        '이후', '통해', '이상', '이하', '있다', '했다', '하고',
        '하는', '위해', '대해', '대한', '으로', '에서', '에게',
    }
    # 형태소 대신 단순 공백 분리 (morpheme 미사용)
    all_words = []
    for text in df['text'].dropna():
        words = re.findall(r'[가-힣]{2,6}', text)
        all_words.extend([w for w in words if w not in stop_words])

    counter = Counter(all_words)
    top30 = counter.most_common(30)

    fig, ax = plt.subplots(figsize=(14, 6))
    words_, counts_ = zip(*top30)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top30)))[::-1]
    ax.barh(words_[::-1], counts_[::-1], color=colors, edgecolor='black')
    ax.set_title('전체 기사 Top-30 키워드')
    ax.set_xlabel('빈도')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'eda_04_top_keywords.png'), dpi=150)
    plt.close()
    print('  → eda_04_top_keywords.png 저장')


# ── 6. 출처별 카테고리 히트맵 ────────────────────────────────────────────────
def plot_source_category_heatmap(df: pd.DataFrame):
    pivot = df.pivot_table(index='source', columns='category',
                           aggfunc='size', fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(pivot.values, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, pivot.values[i, j], ha='center', va='center', fontsize=9)
    ax.set_title('출처별 × 카테고리별 기사 수')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'eda_05_source_category.png'), dpi=150)
    plt.close()
    print('  → eda_05_source_category.png 저장')


# ── 메인 ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df = load_all()
    print_summary(df)
    plot_label_distribution(df)
    plot_text_length(df)
    plot_daily_count(df)
    plot_top_keywords(df)
    plot_source_category_heatmap(df)
    print(f'\nEDA 완료 → {FIG_DIR}/')
