"""
auto_label_v2.py
================
articles_v2.csv (전처리 v2 완료본)에 라벨링만 다시 수행
- 텍스트 정제는 v2에서 이미 완료되었으므로 스킵
- 자동 감성 라벨링 + 카테고리 라벨링만 수행
- train/val/test 재분할

실행: python src/auto_label_v2.py
"""
import re
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ── 입력/출력 경로 (v2용으로 변경) ──────────────────────────────────────
INPUT_PATH = 'data/processed/articles_v2.csv'      # v2 전처리 완료본
PROC_DIR   = 'data/processed_v2'                   # 분리 폴더 (v1과 충돌 방지)
os.makedirs(PROC_DIR, exist_ok=True)

# ── 자동 감성 라벨 키워드 (기존 그대로 유지) ───────────────────────
POSITIVE_KEYWORDS = [
    '우승', '완봉', '완투', '역전', '끝내기', '만루홈런', '퍼펙트',
    '최다', '신기록', '호투', '쾌투', '대승', '압승', '연승',
    '복귀', '활약', '맹타', '폭발', '화려', '기대', '선발 예정',
    '계약 연장', '영입 성공', '훈련 순항', '부활',
]
NEGATIVE_KEYWORDS = [
    '패배', '부상', '이탈', '방출', '퇴출', '논란', '갈등',
    '실망', '최저', '최하', '연패', '폭투', '난타', '무너',
    '충돌', '위기', '경고', '징계', '사과', '은퇴 고려',
    '방출 위기', '부진', '실책', '실패', '탈락',
]

CATEGORY_KEYWORDS = {
    '타격': ['홈런', '안타', '타율', '타점', '출루율', '장타', '타격', '배팅', '4번타자', '클린업'],
    '투구': ['삼진', '방어율', '선발', '불펜', '마무리', '투구', '완봉', '완투', '세이브', '홀드', '구위'],
    '트레이드': ['트레이드', '이적', '교환', '방출', '영입', '계약', 'FA', '자유계약', '다년계약'],
    '부상': ['부상', '이탈', '등록 말소', '수술', '재활', '회복', 'DL', '부상 우려'],
    'FA': ['FA', '자유계약', '다년 계약', '연봉', '계약 협상', '옵트아웃', '잔류', '해외 진출'],
    '감독': ['감독', '코치', '벤치', '작전', '선발 로테이션', '수뇌부', '구단주', '단장'],
    '심판': ['심판', '오심', '판정', '비디오 판독', 'VAR', '어필'],
}


def label_sentiment(text: str) -> int:
    pos = sum(text.count(kw) for kw in POSITIVE_KEYWORDS)
    neg = sum(text.count(kw) for kw in NEGATIVE_KEYWORDS)
    if pos > neg:
        return 1
    elif neg > pos:
        return -1
    return 0


def label_category(text: str) -> str:
    scores = {cat: 0 for cat in CATEGORY_KEYWORDS}
    for cat, kws in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(text.count(kw) for kw in kws)
    best_cat = max(scores, key=scores.get)
    return best_cat if scores[best_cat] > 0 else '기타'


def relabel_v2(val_ratio: float = 0.1,
               test_ratio: float = 0.1,
               seed: int = 42) -> dict:
    """v2 정제 데이터에 라벨링만 새로 수행"""

    # ── 1. v2 데이터 로드 ─────────────────────────────────
    print(f"  로드: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig')
    print(f"  총 기사: {len(df):,}건")

    # ── 2. 결측 제거 (정제는 이미 완료, 스킵) ─────────────
    df = df.dropna(subset=['title', 'body']).reset_index(drop=True)

    # title + body 결합 (v1과 동일한 입력 형식)
    df['text'] = df['title'].astype(str) + ' ' + df['body'].astype(str)

    # 본문 너무 짧은 것 제거 (v2 정제 후에도 일관성 위해 유지)
    df = df[df['body'].str.len() >= 100].reset_index(drop=True)
    print(f"  본문 100자 미만 제거 후: {len(df):,}건")

    # ── 3. 라벨링 ───────────────────────────────────────
    df['sentiment'] = df['text'].apply(label_sentiment)
    df['category']  = df['text'].apply(label_category)
    df['sentiment_str'] = df['sentiment'].map({1: '긍정', 0: '중립', -1: '부정'})

    print(f"\n  v2 감성 분포:")
    print(df['sentiment_str'].value_counts().to_string())
    print(f"\n  v2 카테고리 분포:")
    print(df['category'].value_counts().to_string())

    # ── 4. 분할 (random_state=42 유지 — v1과 동일) ─────
    train_df, temp_df = train_test_split(
        df, test_size=val_ratio + test_ratio,
        random_state=seed, stratify=df['sentiment_str']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed, stratify=temp_df['sentiment_str']
    )

    for split, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        path = os.path.join(PROC_DIR, f'{split}.csv')
        split_df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"  저장: {path}  ({len(split_df):,}건)")

    # ── 5. 메타 저장 ────────────────────────────────────
    meta = {
        'version'   : 'v2',
        'input'     : INPUT_PATH,
        'total'     : len(df),
        'train'     : len(train_df),
        'val'       : len(val_df),
        'test'      : len(test_df),
        'sentiment_dist': df['sentiment_str'].value_counts().to_dict(),
        'category_dist' : df['category'].value_counts().to_dict(),
        'val_ratio' : val_ratio,
        'test_ratio': test_ratio,
        'seed'      : seed,
    }
    with open(os.path.join(PROC_DIR, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print('\nv2 라벨링 완료')
    return {'train': train_df, 'val': val_df, 'test': test_df}


if __name__ == '__main__':
    relabel_v2()