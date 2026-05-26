"""
eda_noise.py
============
전처리 v2 설계를 위한 노이즈 진단 EDA
담당: 김규원

목적
────
1. 본문 내 사진 설명(캡션) 문구 탐지
2. 영문 전용 기사 탐지 (한글 비율 기반)
3. 짧은 본문 / 광고성 잔여 패턴 탐지
4. 출처별 노이즈 분포 파악
→ 전처리 코드(prepro_v2.py) 규칙 설계 근거 확보

실행: python src/eda_noise.py
"""
import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import platform
from collections import Counter

# ── 한글 폰트 ────────────────────────────────────────────────────────────────
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

RAW_DIR = 'data/원시데이터'
FIG_DIR = 'results/figures'
REPORT_DIR = 'results'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ── 데이터 로드 (원시 CSV 그대로) ─────────────────────────────────────────────
def load_raw() -> pd.DataFrame:
    files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"{RAW_DIR}/ 에 원시 CSV가 없습니다.")
    dfs = []
    for f in sorted(files):
        df = pd.read_csv(os.path.join(RAW_DIR, f), encoding='utf-8-sig')
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=['body']).reset_index(drop=True)
    print(f"  원시 데이터 로드: {len(df):,}건")
    return df


# ── 1. 한글/영문 비율 분석 ───────────────────────────────────────────────────
HANGUL_RE = re.compile(r'[가-힣]')
ALPHA_RE  = re.compile(r'[A-Za-z]')

def hangul_ratio(text: str) -> float:
    """전체 문자 대비 한글 비율"""
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    total = len(text)
    hangul = len(HANGUL_RE.findall(text))
    return hangul / total

def has_any_hangul(text: str) -> bool:
    """한글이 단 1개라도 있으면 True"""
    return bool(HANGUL_RE.search(text)) if isinstance(text, str) else False

def analyze_language(df: pd.DataFrame):
    print("\n[1] 언어 분포 분석")
    df['hangul_ratio'] = df['body'].apply(hangul_ratio)
    df['has_hangul']   = df['body'].apply(has_any_hangul)

    # 한글 1글자도 없는 기사
    no_hangul = df[~df['has_hangul']]
    print(f"  한글 0개 기사 : {len(no_hangul):,}건  ({len(no_hangul)/len(df)*100:.2f}%)")

    # 한글 비율 구간
    bins = [0, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0]
    labels = ['0%', '0~1%', '1~10%', '10~30%', '30~50%', '50~70%', '70%↑']
    # 0% 구간은 별도 처리
    df['hangul_bin'] = pd.cut(df['hangul_ratio'],
                              bins=[-0.001, 0.001, 0.1, 0.3, 0.5, 0.7, 1.01],
                              labels=['0%(영문전용)', '0~10%', '10~30%',
                                      '30~50%', '50~70%', '70%↑'])
    bin_counts = df['hangul_bin'].value_counts().sort_index()
    print(f"\n  한글 비율 구간별:\n{bin_counts.to_string()}")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(df['hangul_ratio'], bins=50, color='steelblue',
                 edgecolor='black', alpha=0.8)
    axes[0].axvline(0.3, color='red', linestyle='--', label='추천 임계값 0.3')
    axes[0].set_title('기사 본문 한글 비율 분포')
    axes[0].set_xlabel('한글 비율')
    axes[0].set_ylabel('기사 수')
    axes[0].legend()

    bin_counts.plot(kind='bar', ax=axes[1], color='coral',
                    edgecolor='black', alpha=0.85)
    axes[1].set_title('한글 비율 구간별 기사 수')
    axes[1].set_xlabel('한글 비율 구간')
    axes[1].set_ylabel('기사 수')
    axes[1].tick_params(axis='x', rotation=30)
    for i, v in enumerate(bin_counts.values):
        axes[1].text(i, v + 3, str(v), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'noise_01_language.png'), dpi=150)
    plt.close()
    print('  → noise_01_language.png 저장')
    return df, no_hangul


# ── 2. 사진 캡션 패턴 탐지 ───────────────────────────────────────────────────
# 흔한 캡션 시그니처
CAPTION_PATTERNS = {
    '사진_제공'   : re.compile(r'사진\s*[=:]\s*'),
    '사진제공'    : re.compile(r'사진\s*제공'),
    '뉴스1제공'   : re.compile(r'뉴스1\s*제공'),
    '연합뉴스제공': re.compile(r'연합뉴스\s*(제공|자료사진)'),
    '캡션_괄호'   : re.compile(r'\[\s*사진\s*[^\]]*\]'),
    '괄호기자'    : re.compile(r'\([^)]{0,15}기자\)'),
    '게티이미지'  : re.compile(r'게티이미지|GettyImages', re.I),
    '자료사진'    : re.compile(r'자료\s*사진'),
    '그래픽'      : re.compile(r'그래픽\s*[=:]'),
    'OSEN'        : re.compile(r'OSEN\s*[=:]'),
}

def analyze_captions(df: pd.DataFrame):
    print("\n[2] 사진 캡션 패턴 탐지")
    results = {}
    for name, pat in CAPTION_PATTERNS.items():
        hit = df['body'].str.contains(pat, na=False).sum()
        results[name] = int(hit)
        print(f"  {name:15s}: {hit:,}건")

    # 시각화
    fig, ax = plt.subplots(figsize=(10, 5))
    names  = list(results.keys())
    values = list(results.values())
    ax.barh(names[::-1], values[::-1], color='teal',
            edgecolor='black', alpha=0.85)
    for i, v in enumerate(values[::-1]):
        ax.text(v + max(values)*0.01, i, str(v), va='center', fontsize=9)
    ax.set_title('본문 내 사진 캡션 추정 패턴 빈도')
    ax.set_xlabel('해당 패턴 포함 기사 수')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'noise_02_caption_patterns.png'), dpi=150)
    plt.close()
    print('  → noise_02_caption_patterns.png 저장')
    return results


# ── 3. 본문 첫 줄/마지막 줄 샘플 (캡션·저작권 위치 파악) ─────────────────────
def sample_head_tail(df: pd.DataFrame, n=20):
    print("\n[3] 본문 첫/마지막 50자 샘플")
    samples = {'head': [], 'tail': []}
    for i, body in enumerate(df['body'].sample(min(n, len(df)),
                                                random_state=42)):
        head = str(body)[:60].replace('\n', ' ')
        tail = str(body)[-60:].replace('\n', ' ')
        samples['head'].append(head)
        samples['tail'].append(tail)

    # 가장 빈번한 본문 시작 토큰(첫 단어) 20개
    first_tokens = df['body'].astype(str).str.strip().str.split().str[0]
    top_starts   = first_tokens.value_counts().head(20)
    print("\n  본문 첫 단어 Top-20:")
    print(top_starts.to_string())
    return samples, top_starts.to_dict()


# ── 4. 본문 길이 분포 + 짧은 기사 검출 ───────────────────────────────────────
def analyze_length(df: pd.DataFrame):
    print("\n[4] 본문 길이 분포")
    df['body_len'] = df['body'].str.len()
    print(f"  평균: {df['body_len'].mean():.0f}자")
    print(f"  중앙: {df['body_len'].median():.0f}자")
    print(f"  최소: {df['body_len'].min()}자, 최대: {df['body_len'].max()}자")
    print(f"  100자 미만 : {(df['body_len'] < 100).sum()}건")
    print(f"  200자 미만 : {(df['body_len'] < 200).sum()}건")
    print(f"  300자 미만 : {(df['body_len'] < 300).sum()}건")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df['body_len'].clip(upper=5000), bins=60,
            color='slateblue', edgecolor='black', alpha=0.8)
    for thr, c in [(100, 'red'), (200, 'orange'), (300, 'green')]:
        ax.axvline(thr, linestyle='--', color=c, label=f'{thr}자')
    ax.set_title('본문 길이 분포 (상한 5000자 클리핑)')
    ax.set_xlabel('글자 수')
    ax.set_ylabel('기사 수')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'noise_03_body_length.png'), dpi=150)
    plt.close()
    print('  → noise_03_body_length.png 저장')


# ── 5. 출처별 노이즈 비율 ────────────────────────────────────────────────────
def analyze_by_source(df: pd.DataFrame):
    print("\n[5] 출처별 노이즈 비율")
    grp = df.groupby('source').agg(
        n            =('body', 'size'),
        no_hangul    =('has_hangul', lambda s: (~s).sum()),
        low_hangul   =('hangul_ratio', lambda s: (s < 0.3).sum()),
        avg_len      =('body_len', 'mean'),
        short_count  =('body_len', lambda s: (s < 200).sum()),
    )
    grp['no_hangul_pct']  = (grp['no_hangul']  / grp['n'] * 100).round(2)
    grp['low_hangul_pct'] = (grp['low_hangul'] / grp['n'] * 100).round(2)
    grp['short_pct']      = (grp['short_count']/ grp['n'] * 100).round(2)
    print(grp.to_string())
    grp.to_csv(os.path.join(REPORT_DIR, 'noise_by_source.csv'),
               encoding='utf-8-sig')
    print('  → noise_by_source.csv 저장')
    return grp


# ── 6. 영문 전용/저한글 기사 실제 샘플 저장 ──────────────────────────────────
def save_noise_samples(df: pd.DataFrame, n=15):
    print("\n[6] 노이즈 의심 기사 샘플 저장")
    suspects = df[df['hangul_ratio'] < 0.3].copy()
    suspects = suspects.sort_values('hangul_ratio').head(n)
    cols = ['source', 'title', 'hangul_ratio', 'body']
    suspects['body'] = suspects['body'].str[:300]
    out = suspects[cols]
    out.to_csv(os.path.join(REPORT_DIR, 'noise_suspect_samples.csv'),
               index=False, encoding='utf-8-sig')
    print(f'  → noise_suspect_samples.csv 저장 ({len(out)}건)')


# ── 7. 종합 리포트 JSON ──────────────────────────────────────────────────────
def save_report(df, no_hangul, caption_stats, by_source):
    report = {
        'total_articles'        : int(len(df)),
        'no_hangul_articles'    : int(len(no_hangul)),
        'no_hangul_pct'         : round(len(no_hangul)/len(df)*100, 2),
        'low_hangul_lt_30pct'   : int((df['hangul_ratio'] < 0.3).sum()),
        'short_body_lt_200'     : int((df['body_len'] < 200).sum()),
        'caption_pattern_hits'  : caption_stats,
        'source_summary'        : by_source.reset_index().to_dict(orient='records'),
        'recommendation': {
            'language_filter'  : '한글 비율 ≥ 0.3 인 기사만 유지 (영문 전용 제거)',
            'caption_removal'  : '사진=, 사진제공, [사진 …], (○○ 기자), 게티이미지 등 정규식 제거',
            'min_body_length'  : '본문 200자 이상',
            'rationale'        : '본 EDA 결과 기반 임계값'
        }
    }
    path = os.path.join(REPORT_DIR, 'noise_eda_report.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  종합 리포트 저장 → {path}")


# ── 7. 캡션 실제 문장 추출 (마커 + 뒤따르는 문장) ──────────────────────────
def extract_caption_samples(df: pd.DataFrame, n=30):
    """
    사진 마커 뒤에 어떤 문장이 따라오는지 실제로 뽑아서 본다.
    → 캡션 종료 지점(마침표/줄바꿈/문장부호) 패턴 파악
    """
    print("\n[7] 캡션 실제 문장 샘플 추출")

    # 마커별로 뒤따르는 텍스트 100자까지 캡쳐
    patterns = {
        '사진=뒤'   : re.compile(r'(사진\s*[=:]\s*[^\n]{0,150})'),
        '사진제공뒤': re.compile(r'(사진\s*제공[^\n]{0,150})'),
        '▲뒤'      : re.compile(r'(▲[^\n]{0,150})'),
        '괄호기자'  : re.compile(r'(\([^)]{0,30}기자\))'),
        '게티뒤'   : re.compile(r'(게티이미지[^\n]{0,100})'),
    }

    all_samples = {}
    for name, pat in patterns.items():
        hits = []
        for body in df['body'].dropna():
            matches = pat.findall(str(body))
            hits.extend(matches)
        all_samples[name] = hits

        # 통계
        if hits:
            lengths = [len(h) for h in hits]
            print(f"\n  [{name}] 총 {len(hits):,}건")
            print(f"    평균 길이: {np.mean(lengths):.1f}자 / "
                  f"중앙: {np.median(lengths):.0f}자 / "
                  f"최대: {max(lengths)}자")
            print(f"    샘플 5개:")
            for s in hits[:5]:
                preview = s.replace('\n', ' ')[:120]
                print(f"      · {preview}")

    # CSV 저장
    rows = []
    for name, hits in all_samples.items():
        for h in hits[:50]:  # 패턴당 50개씩
            rows.append({'pattern': name,
                         'length': len(h),
                         'text': h.replace('\n', ' ')[:200]})
    pd.DataFrame(rows).to_csv(
        os.path.join(REPORT_DIR, 'caption_samples.csv'),
        index=False, encoding='utf-8-sig')
    print(f'\n  → caption_samples.csv 저장')
    return all_samples


# ── 8. 캡션 길이 분포 (정규식 max 길이 결정용) ──────────────────────────────
def analyze_caption_length(df: pd.DataFrame):
    """
    '사진=' 마커 뒤 문장이 보통 어디서 끝나는지 (마침표/줄바꿈/공백)
    → 정규식의 종결 조건 설계 근거
    """
    print("\n[8] '사진=' 뒤 캡션 종료 지점 분석")
    
    # 사진= 뒤 ~ 다음 마침표/줄바꿈까지 캡쳐
    pat = re.compile(r'사진\s*[=:]\s*([^.\n]{0,200})[.\n]')
    
    lengths = []
    enders  = Counter()
    for body in df['body'].dropna():
        for m in pat.finditer(str(body)):
            caption = m.group(1)
            lengths.append(len(caption))
            # 다음 문자가 . 인지 \n 인지
            end_pos = m.end()
            if end_pos <= len(body):
                enders[body[end_pos-1]] += 1
    
    if lengths:
        print(f"  '사진=' 캡션 길이 분포:")
        print(f"    평균: {np.mean(lengths):.1f}자")
        print(f"    중앙: {np.median(lengths):.0f}자")
        print(f"    90%: {np.percentile(lengths, 90):.0f}자")
        print(f"    최대: {max(lengths)}자")
        print(f"  종결 문자 분포: {dict(enders.most_common(5))}")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(lengths, bins=40, color='salmon', edgecolor='black', alpha=0.8)
        ax.axvline(np.percentile(lengths, 90), color='red',
                   linestyle='--', label=f'90% 지점 {np.percentile(lengths,90):.0f}자')
        ax.set_title("'사진=' 뒤 캡션 길이 분포")
        ax.set_xlabel('캡션 길이 (글자)')
        ax.set_ylabel('빈도')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'noise_04_caption_length.png'), dpi=150)
        plt.close()
        print('  → noise_04_caption_length.png 저장')


# ── 메인에 추가 ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print('  전처리 v2 설계용 노이즈 진단 EDA')
    print('=' * 60)

    df = load_raw()
    df, no_hangul   = analyze_language(df)
    caption_stats   = analyze_captions(df)
    sample_head_tail(df, n=20)
    analyze_length(df)
    by_source       = analyze_by_source(df)
    save_noise_samples(df, n=15)
    extract_caption_samples(df, n=30)      # ✨ 추가
    analyze_caption_length(df)             # ✨ 추가
    save_report(df, no_hangul, caption_stats, by_source)

    print('\n노이즈 EDA 완료')