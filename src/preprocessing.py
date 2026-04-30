"""
preprocessing.py
================
전처리 파이프라인
담당: 전처리+EDA 담당자

기능
────
1. 3개 크롤러 CSV 병합 → 중복 제거
2. 텍스트 정제 (HTML태그, 특수문자, 광고문구 등 제거)
3. 자동 라벨링
   - 감성 레이블 (긍정 1 / 중립 0 / 부정 -1)
   - 카테고리 레이블 (타격 / 투구 / 트레이드 / 부상 / FA / 감독 / 심판 / 기타)
4. 학습/검증/테스트 분할 → data/processed/ 저장

실행: python src/preprocessing.py
"""
import re
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RAW_DIR  = 'data/raw'
PROC_DIR = 'data/processed'
os.makedirs(PROC_DIR, exist_ok=True)

# ── 자동 감성 라벨 키워드 ─────────────────────────────────────────────────────
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

# ── 카테고리 키워드 ───────────────────────────────────────────────────────────
CATEGORY_KEYWORDS = {
    '타격': ['홈런', '안타', '타율', '타점', '출루율', '장타', '타격', '배팅', '4번타자', '클린업'],
    '투구': ['삼진', '방어율', '선발', '불펜', '마무리', '투구', '완봉', '완투', '세이브', '홀드', '구위'],
    '트레이드': ['트레이드', '이적', '교환', '방출', '영입', '계약', 'FA', '자유계약', '다년계약'],
    '부상': ['부상', '이탈', '등록 말소', '수술', '재활', '회복', 'DL', '부상 우려'],
    'FA': ['FA', '자유계약', '다년 계약', '연봉', '계약 협상', '옵트아웃', '잔류', '해외 진출'],
    '감독': ['감독', '코치', '벤치', '작전', '선발 로테이션', '수뇌부', '구단주', '단장'],
    '심판': ['심판', '오심', '판정', '비디오 판독', 'VAR', '어필'],
}


# ── 텍스트 정제 ───────────────────────────────────────────────────────────────
_AD_PATTERNS = [
    r'<[^>]+>',                          # HTML 태그
    r'\[.*?\]',                          # [광고], [사진] 등
    r'▶.*',                              # 화살표 이하 광고/링크
    r'Copyright.*',                      # 저작권 문구
    r'무단.*금지',
    r'저작권자.*?기자',
    r'기사제보.*',
    r'[①②③④⑤⑥⑦⑧⑨⑩]',
    r'\s{2,}',                           # 연속 공백
]

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    for pat in _AD_PATTERNS:
        text = re.sub(pat, ' ', text)
    text = re.sub(r'[^\w\s가-힣.,!?]', ' ', text)
    return text.strip()


# ── 감성 자동 라벨링 ──────────────────────────────────────────────────────────
def label_sentiment(text: str) -> int:
    """
    긍정 1 / 중립 0 / 부정 -1
    제목과 본문의 키워드 빈도 차이로 판정
    """
    pos = sum(text.count(kw) for kw in POSITIVE_KEYWORDS)
    neg = sum(text.count(kw) for kw in NEGATIVE_KEYWORDS)
    if pos > neg:
        return 1
    elif neg > pos:
        return -1
    return 0


# ── 카테고리 자동 라벨링 ──────────────────────────────────────────────────────
def label_category(text: str) -> str:
    """가장 많이 등장한 카테고리 키워드로 분류"""
    scores = {cat: 0 for cat in CATEGORY_KEYWORDS}
    for cat, kws in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(text.count(kw) for kw in kws)
    best_cat = max(scores, key=scores.get)
    return best_cat if scores[best_cat] > 0 else '기타'


# ── 메인 전처리 ───────────────────────────────────────────────────────────────
def preprocess(val_ratio: float = 0.1,
               test_ratio: float = 0.1,
               seed: int = 42) -> dict:
    """
    1. CSV 병합
    2. 정제 + 라벨링
    3. 분할 저장

    Returns
    -------
    dict: {'train': df, 'val': df, 'test': df}
    """
    # ── 1. 병합 ─────────────────────────────────────────
    csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"'{RAW_DIR}/'에 CSV 파일이 없습니다. 크롤러를 먼저 실행하세요.")

    dfs = []
    for fname in csv_files:
        path = os.path.join(RAW_DIR, fname)
        df = pd.read_csv(path, encoding='utf-8-sig')
        dfs.append(df)
        print(f"  로드: {fname}  ({len(df)}건)")

    df = pd.concat(dfs, ignore_index=True)
    print(f"\n  병합 전 총계: {len(df)}건")

    # ── 2. 중복 제거 ─────────────────────────────────────
    df.drop_duplicates(subset='url', inplace=True)
    df.dropna(subset=['title', 'body'], inplace=True)
    print(f"  중복/결측 제거 후: {len(df)}건")

    # ── 3. 텍스트 정제 ───────────────────────────────────
    df['title_clean'] = df['title'].apply(clean_text)
    df['body_clean']  = df['body'].apply(clean_text)
    df['text']        = df['title_clean'] + ' ' + df['body_clean']

    # 본문 너무 짧은 것 제거 (100자 미만)
    df = df[df['body_clean'].str.len() >= 100].reset_index(drop=True)
    print(f"  본문 100자 미만 제거 후: {len(df)}건")

    # ── 4. 라벨링 ────────────────────────────────────────
    df['sentiment'] = df['text'].apply(label_sentiment)
    df['category']  = df['text'].apply(label_category)

    # 감성 레이블 → 문자열 매핑
    df['sentiment_str'] = df['sentiment'].map({1: '긍정', 0: '중립', -1: '부정'})

    print(f"\n  감성 분포:\n{df['sentiment_str'].value_counts().to_string()}")
    print(f"\n  카테고리 분포:\n{df['category'].value_counts().to_string()}")

    # ── 5. 분할 ──────────────────────────────────────────
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
        print(f"  저장: {path}  ({len(split_df)}건)")

    # ── 6. 메타 저장 ─────────────────────────────────────
    meta = {
        'total'     : len(df),
        'train'     : len(train_df),
        'val'       : len(val_df),
        'test'      : len(test_df),
        'sentiment_dist': df['sentiment_str'].value_counts().to_dict(),
        'category_dist' : df['category'].value_counts().to_dict(),
        'sources'   : df['source'].value_counts().to_dict(),
        'val_ratio' : val_ratio,
        'test_ratio': test_ratio,
        'seed'      : seed,
    }
    with open(os.path.join(PROC_DIR, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print('\n전처리 완료')
    return {'train': train_df, 'val': val_df, 'test': test_df}


if __name__ == '__main__':
    preprocess()
