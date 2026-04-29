"""
prepro.py (v2 Final)
====================
전처리 파이프라인
담당: 김규원

변경사항
────────
- category_v1 (8개) + category (v2, 6개) 동시 생성
- v1 vs v2 비교 JSON 저장
- Timer 소요시간 기록

실행: python src/prepro.py  (로컬 CPU, 약 10초)
"""
import re, os, json, time
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR    = 'data/raw'
PROC_DIR   = 'data/processed'
RESULT_DIR = 'results'
os.makedirs(PROC_DIR,   exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


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
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)


POSITIVE_KEYWORDS = [
    '우승','역전','끝내기','대승','압승','연승','완봉','완투',
    '퍼펙트게임','노히트','만루홈런','신기록','최다','호투','쾌투',
    '맹타','폭발적','영입 성공','계약 연장','부활','전성기','최고',
]
NEGATIVE_KEYWORDS = [
    '패배','연패','대패','완패','부상','수술','재활','결장','말소',
    '방출','퇴출','논란','갈등','충돌','징계','경고','부진','실망',
    '최저','최하','폭투','난타','무너','실책','실패','탈락','사과',
]

CATEGORY_KEYWORDS_V1 = {
    '타격'   : ['홈런','안타','타율','타점','출루율','장타','타격','배팅','4번타자','클린업'],
    '투구'   : ['삼진','방어율','선발','불펜','마무리','투구','완봉','완투','세이브','홀드','구위'],
    '트레이드': ['트레이드','이적','교환','방출','영입','계약','다년계약'],
    '부상'   : ['부상','이탈','등록 말소','수술','재활','회복','부상 우려'],
    'FA'     : ['FA','자유계약','다년 계약','연봉','계약 협상','옵트아웃','잔류','해외 진출'],
    '감독'   : ['감독','코치','벤치','작전','선발 로테이션','수뇌부','구단주','단장'],
    '심판'   : ['심판','오심','판정','비디오 판독','어필'],
}

CATEGORY_KEYWORDS_V2 = {
    '타격'    : ['홈런','안타','타율','타점','출루율','장타율',
                '타격','배팅','4번타자','클린업','득점','타선'],
    '투구'    : ['삼진','방어율','선발','불펜','마무리',
                '투구','완봉','완투','세이브','홀드','구위',
                '피안타','볼넷','탈삼진','이닝'],
    '선수이동': ['트레이드','이적','교환','영입','계약',
                '다년계약','다년 계약','FA','자유계약',
                '계약 협상','연봉','옵트아웃','잔류','해외 진출','FA 시장','방출'],
    '부상'    : ['부상','등록 말소','수술','재활','회복',
                '결장','부상 우려','통증','접질'],
    '감독·운영': ['감독','코치','벤치','작전',
                 '선발 로테이션','수뇌부','구단주','단장',
                 '구단','프런트','스프링캠프'],
}

_AD_PATTERNS = [
    r'<[^>]+>', r'\[.*?\]', r'▶.*', r'Copyright\s*©?.*',
    r'무단\s*전재.*금지', r'저작권자\s*©?.*?기자', r'기사\s*제보.*',
    r'[①②③④⑤⑥⑦⑧⑨⑩]',
    r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
    r'https?://\S+', r'\s{2,}',
]

def clean_text(text):
    if not isinstance(text, str): return ''
    for pat in _AD_PATTERNS:
        text = re.sub(pat, ' ', text)
    text = re.sub(r'[^\w\s가-힣.,!?]', ' ', text)
    return text.strip()

def label_sentiment(text, threshold=2):
    pos  = sum(text.count(kw) for kw in POSITIVE_KEYWORDS)
    neg  = sum(text.count(kw) for kw in NEGATIVE_KEYWORDS)
    diff = pos - neg
    if diff >= threshold:    return '긍정'
    elif diff <= -threshold: return '부정'
    return '중립'

def label_category(text, kw_dict):
    scores   = {cat: sum(text.count(kw) for kw in kws) for cat, kws in kw_dict.items()}
    best_cat = max(scores, key=scores.get)
    return best_cat if scores[best_cat] > 0 else '기타'

def save_category_comparison(df):
    comparison = {
        'v1': {
            'n_categories' : len(CATEGORY_KEYWORDS_V1) + 1,
            'categories'   : list(CATEGORY_KEYWORDS_V1.keys()) + ['기타'],
            'distribution' : df['category_v1'].value_counts().to_dict(),
            'description'  : '원본 8개 카테고리 (계획서 기준)',
        },
        'v2': {
            'n_categories' : len(CATEGORY_KEYWORDS_V2) + 1,
            'categories'   : list(CATEGORY_KEYWORDS_V2.keys()) + ['기타'],
            'distribution' : df['category'].value_counts().to_dict(),
            'description'  : '개선 6개 카테고리 (소수클래스 통합)',
        },
        'changes': {
            '심판(약 71건)': '기타로 편입',
            'FA+트레이드'  : '선수이동으로 통합',
            '근거'         : '계획서 조항: 50건 미만 소수 클래스 기타 병합 원칙 준용',
        },
    }
    path = os.path.join(RESULT_DIR, 'category_version_comparison.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"  카테고리 비교 저장: {path}")


def preprocess(val_ratio=0.1, test_ratio=0.1, seed=42):
    timer = Timer()
    timer.start('preprocessing')

    csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"'{RAW_DIR}/'에 CSV 없음")

    dfs = []
    for fname in sorted(csv_files):
        df = pd.read_csv(os.path.join(RAW_DIR, fname), encoding='utf-8-sig')
        dfs.append(df)
        print(f"  로드: {fname} ({len(df)}건)")

    df = pd.concat(dfs, ignore_index=True)
    print(f"\n  병합 전 총계: {len(df)}건")

    before = len(df)
    df.drop_duplicates(subset='url', inplace=True)
    print(f"  URL 중복 제거: {before-len(df)}건 → {len(df)}건")

    before = len(df)
    df.drop_duplicates(subset='title', keep='first', inplace=True)
    print(f"  제목 중복 제거: {before-len(df)}건 → {len(df)}건")

    df.dropna(subset=['title','body'], inplace=True)
    print(f"  결측값 제거 후: {len(df)}건")

    df['date']        = df['date'].astype(str)
    df['title_clean'] = df['title'].apply(clean_text)
    df['body_clean']  = df['body'].apply(clean_text)
    df['text']        = df['title_clean'] + ' ' + df['body_clean']
    df = df[df['title_clean'].str.len() >= 5].reset_index(drop=True)
    df = df[df['body_clean'].str.len()  >= 100].reset_index(drop=True)
    print(f"  길이 필터 후: {len(df)}건")

    df['sentiment_str'] = df['text'].apply(label_sentiment)
    df['category_v1']   = df['text'].apply(lambda t: label_category(t, CATEGORY_KEYWORDS_V1))
    df['category']      = df['text'].apply(lambda t: label_category(t, CATEGORY_KEYWORDS_V2))

    print(f"\n  감성 분포:\n{df['sentiment_str'].value_counts().to_string()}")
    print(f"\n  카테고리 v1:\n{df['category_v1'].value_counts().to_string()}")
    print(f"\n  카테고리 v2:\n{df['category'].value_counts().to_string()}")

    save_category_comparison(df)

    train_df, temp_df = train_test_split(
        df, test_size=val_ratio+test_ratio,
        random_state=seed, stratify=df['sentiment_str'])
    val_df, test_df = train_test_split(
        temp_df, test_size=test_ratio/(val_ratio+test_ratio),
        random_state=seed, stratify=temp_df['sentiment_str'])

    for split, sdf in [('train',train_df),('val',val_df),('test',test_df)]:
        path = os.path.join(PROC_DIR, f'{split}.csv')
        sdf.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"  저장: {path} ({len(sdf)}건)")

    meta = {
        'total'            : len(df),
        'train'            : len(train_df),
        'val'              : len(val_df),
        'test'             : len(test_df),
        'sentiment_dist'   : df['sentiment_str'].value_counts().to_dict(),
        'category_dist'    : df['category'].value_counts().to_dict(),
        'category_v1_dist' : df['category_v1'].value_counts().to_dict(),
        'sources'          : df['source'].value_counts().to_dict(),
        'val_ratio'        : val_ratio,
        'test_ratio'       : test_ratio,
        'seed'             : seed,
        'label_threshold'  : 2,
        'categories_v2'    : list(CATEGORY_KEYWORDS_V2.keys()) + ['기타'],
        'categories_v1'    : list(CATEGORY_KEYWORDS_V1.keys()) + ['기타'],
    }
    with open(os.path.join(PROC_DIR, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    timer.end('preprocessing')
    print('\n전처리 완료')
    return {'train': train_df, 'val': val_df, 'test': test_df}


if __name__ == '__main__':
    preprocess()
