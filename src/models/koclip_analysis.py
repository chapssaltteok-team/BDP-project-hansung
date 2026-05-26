"""
koclip_analysis.py
==================
KoCLIP 기반 제목-이미지 유사도 분석
목적: 멀티모달 성능 저조가 도메인 특성 때문인지 정량 증명

분석 내용
──────────
1. 전체 유사도 분포 → "KBO 뉴스 썸네일의 X%가 자료사진"을 수치로 증명
2. 카테고리별 유사도 분포 비교
3. 감성별 유사도 분포 비교
4. 불일치 유형 분류 (방향 A): 의도적 vs 불가피한 불일치

실행: python src/models/koclip_analysis.py
주의: MPS(Apple Silicon) / CUDA / CPU 자동 감지
"""

import os, sys, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, VisionTextDualEncoderModel

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

PROC_DIR   = 'data/processed'
IMG_DIR    = 'data/images'
RESULT_DIR = 'results'
FIG_DIR    = os.path.join(RESULT_DIR, 'figures', 'koclip')
os.makedirs(FIG_DIR, exist_ok=True)

# ── 디바이스 자동 감지 ────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("  ✅ Apple MPS 사용")
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("  ✅ CUDA 사용")
else:
    DEVICE = torch.device('cpu')
    print("  ⚠️  CPU 사용")

# ── 한글 폰트 ─────────────────────────────────────────────────────────────────
def set_korean_font():
    candidates = [
        '/System/Library/Fonts/AppleSDGothicNeo.ttc',
        '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        'C:/Windows/Fonts/malgun.ttf',
    ]
    for p in candidates:
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            prop = fm.FontProperties(fname=p)
            plt.rcParams['font.family'] = prop.get_name()
            break
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

LABEL2ID          = {'긍정': 0, '중립': 1, '부정': 2}
UNAVOIDABLE_CATS  = ['부상', '선수이동']
MISMATCH_THRESHOLD = 0.25


# ── Dataset ───────────────────────────────────────────────────────────────────
class KoCLIPDataset(Dataset):
    def __init__(self, df, image_map, processor):
        url2path       = dict(zip(image_map['url'], image_map['local_path']))
        self.titles    = df['title'].tolist() if 'title' in df.columns else df['text'].str[:50].tolist()
        self.img_paths = [url2path.get(u, '') for u in df['url'].tolist()]
        self.processor = processor

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.img_paths[idx]).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (180, 180, 180))
        inputs = self.processor(
            text=[self.titles[idx]], images=img,
            return_tensors='pt', padding='max_length',
            truncation=True, max_length=64,
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}


# ── 유사도 계산 ───────────────────────────────────────────────────────────────
def compute_similarities(df, image_map, batch_size=32):
    print("\n  KoCLIP 모델 로드 중...")
    model     = VisionTextDualEncoderModel.from_pretrained('koclip/koclip-base-pt').to(DEVICE)
    processor = CLIPProcessor.from_pretrained('koclip/koclip-base-pt')
    model.eval()

    loader = DataLoader(KoCLIPDataset(df, image_map, processor),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    sims = []
    print(f"  유사도 계산 중 ({len(df)}건)...")
    with torch.no_grad():
        for batch in tqdm(loader, desc='  KoCLIP'):
            pv   = batch['pixel_values'].to(DEVICE)
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            img_emb  = model.get_image_features(pixel_values=pv)
            txt_emb  = model.get_text_features(input_ids=ids, attention_mask=mask)
            img_emb  = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb  = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            sims.extend((img_emb * txt_emb).sum(dim=-1).cpu().numpy().tolist())
    return np.array(sims)


# ── 분석 1: 전체 분포 ─────────────────────────────────────────────────────────
def plot_distribution(sims, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pct_below = (sims < MISMATCH_THRESHOLD).mean() * 100

    ax = axes[0]
    ax.hist(sims, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(MISMATCH_THRESHOLD, color='red', linestyle='--', lw=2,
               label=f'임계값 {MISMATCH_THRESHOLD}')
    ax.axvline(np.median(sims), color='orange', linestyle='--', lw=1.5,
               label=f'중앙값 {np.median(sims):.3f}')
    ax.set_xlabel('코사인 유사도'); ax.set_ylabel('기사 수')
    ax.set_title('KBO 뉴스 제목-이미지 유사도 분포'); ax.legend()

    ax = axes[1]
    sorted_sims = np.sort(sims)
    ax.plot(sorted_sims, np.arange(1, len(sorted_sims)+1)/len(sorted_sims),
            color='steelblue', lw=2)
    ax.axvline(MISMATCH_THRESHOLD, color='red', linestyle='--', lw=2,
               label=f'임계값 이하 {pct_below:.1f}%')
    ax.set_xlabel('코사인 유사도'); ax.set_ylabel('누적 비율')
    ax.set_title('유사도 누적 분포 (CDF)'); ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  [유사도 통계]")
    print(f"  평균: {sims.mean():.4f} | 중앙값: {np.median(sims):.4f} | 표준편차: {sims.std():.4f}")
    print(f"  임계값({MISMATCH_THRESHOLD}) 이하: {pct_below:.1f}% → 자료사진 추정")
    print(f"  → 저장: {save_path}")


# ── 분석 2: 카테고리별 ────────────────────────────────────────────────────────
def plot_by_category(df, sims, save_path):
    if 'category' not in df.columns:
        print("  ⚠️  카테고리 컬럼 없음 — 생략")
        return
    cats   = df['category'].unique()
    data   = [sims[df['category'] == c] for c in cats]
    colors = ['#E74C3C' if c in UNAVOIDABLE_CATS else '#3498DB' for c in cats]

    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(data, labels=cats, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.axhline(MISMATCH_THRESHOLD, color='red', linestyle='--', lw=1.5,
               label=f'임계값 {MISMATCH_THRESHOLD}')
    ax.set_ylabel('코사인 유사도')
    ax.set_title('카테고리별 유사도 (빨강=불가피한 불일치 예상)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  [카테고리별 불일치 비율]")
    for c, d in zip(cats, data):
        pct = (d < MISMATCH_THRESHOLD).mean() * 100
        tag = ' ← 불가피한 불일치 예상' if c in UNAVOIDABLE_CATS else ''
        print(f"  {c:<12} {pct:>6.1f}%{tag}")
    print(f"  → 저장: {save_path}")


# ── 분석 3: 감성별 ───────────────────────────────────────────────────────────
def plot_by_sentiment(df, sims, save_path):
    labels = ['긍정', '중립', '부정']
    colors = ['#2ECC71', '#95A5A6', '#E74C3C']
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color in zip(labels, colors):
        mask = df['sentiment_str'] == label
        ax.hist(sims[mask], bins=40, alpha=0.6,
                label=f'{label} ({mask.sum()}건)', color=color, edgecolor='white')
    ax.axvline(MISMATCH_THRESHOLD, color='black', linestyle='--', lw=1.5,
               label=f'임계값 {MISMATCH_THRESHOLD}')
    ax.set_xlabel('코사인 유사도'); ax.set_ylabel('기사 수')
    ax.set_title('감성별 유사도 분포'); ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  [감성별 평균 유사도]")
    for label in labels:
        mask = df['sentiment_str'] == label
        print(f"  {label}: 평균 {sims[mask].mean():.4f} | 중앙값 {np.median(sims[mask]):.4f}")
    print(f"  → 저장: {save_path}")


# ── 방향 A: 불일치 유형 분류 ──────────────────────────────────────────────────
def classify_mismatch(df, sims, threshold=MISMATCH_THRESHOLD):
    result = df.copy()
    result['koclip_similarity'] = sims

    def label(row):
        if row['koclip_similarity'] >= threshold:
            return '일치'
        if 'category' in row and row['category'] in UNAVOIDABLE_CATS:
            return '불가피한_불일치'
        return '의도적_불일치_후보'

    result['mismatch_type'] = result.apply(label, axis=1)
    return result


def plot_mismatch_types(result_df, save_path):
    counts = result_df['mismatch_type'].value_counts()
    color_map = {'일치': '#2ECC71', '불가피한_불일치': '#F39C12',
                 '의도적_불일치_후보': '#E74C3C', '불일치': '#E74C3C'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    wedge_colors = [color_map.get(k, '#95A5A6') for k in counts.index]
    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
           colors=wedge_colors, startangle=90, pctdistance=0.8)
    ax.set_title('불일치 유형 분포')

    if 'category' in result_df.columns:
        ax = axes[1]
        pivot = result_df.groupby(['category', 'mismatch_type']).size().unstack(fill_value=0)
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        pivot_pct.plot(kind='bar', ax=ax, stacked=True,
                       color=[color_map.get(c, '#95A5A6') for c in pivot_pct.columns],
                       edgecolor='white')
        ax.set_ylabel('비율 (%)'); ax.set_title('카테고리별 불일치 유형 비율')
        ax.legend(loc='upper right', fontsize=9)
        ax.tick_params(axis='x', rotation=45)
    else:
        axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  [불일치 유형 분류 결과]")
    total = len(result_df)
    for mtype, cnt in counts.items():
        print(f"  {mtype:<22} {cnt:>6}건 ({cnt/total*100:.1f}%)")
    print(f"  → 저장: {save_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def run_koclip_analysis():
    print(f"\n{'='*60}")
    print(f"  KoCLIP 유사도 분석  |  device={DEVICE}")
    print(f"{'='*60}")

    val_df    = pd.read_csv(os.path.join(PROC_DIR, 'val.csv'),       encoding='utf-8-sig')
    test_df   = pd.read_csv(os.path.join(PROC_DIR, 'test.csv'),      encoding='utf-8-sig')
    image_map = pd.read_csv(os.path.join(PROC_DIR, 'image_map.csv'), encoding='utf-8-sig')

    # val + test 통합 처리
    val_df['split']  = 'val'
    test_df['split'] = 'test'
    combined_df = pd.concat([val_df, test_df], ignore_index=True)
    print(f"  분석 데이터: val {len(val_df)}건 + test {len(test_df)}건 = 총 {len(combined_df)}건")

    sims = compute_similarities(combined_df, image_map)

    print(f"\n{'─'*50}\n  [분석 1] 전체 유사도 분포")
    plot_distribution(sims, os.path.join(FIG_DIR, 'koclip_01_distribution.png'))

    print(f"\n{'─'*50}\n  [분석 2] 카테고리별")
    plot_by_category(test_df, sims, os.path.join(FIG_DIR, 'koclip_02_by_category.png'))

    print(f"\n{'─'*50}\n  [분석 3] 감성별")
    plot_by_sentiment(test_df, sims, os.path.join(FIG_DIR, 'koclip_03_by_sentiment.png'))

    print(f"\n{'─'*50}\n  [방향 A] 불일치 유형 분류")
    result_df = classify_mismatch(test_df, sims)
    plot_mismatch_types(result_df, os.path.join(FIG_DIR, 'koclip_04_mismatch_types.png'))

    result_df.to_csv(os.path.join(RESULT_DIR, 'koclip_similarity_results.csv'),
                     index=False, encoding='utf-8-sig')

    summary = {
        'model'              : 'KoCLIP_Analysis',
        'n_samples'          : int(len(test_df)),
        'mean_similarity'    : round(float(sims.mean()), 4),
        'median_similarity'  : round(float(np.median(sims)), 4),
        'std_similarity'     : round(float(sims.std()), 4),
        'threshold'          : MISMATCH_THRESHOLD,
        'pct_below_threshold': round(float((sims < MISMATCH_THRESHOLD).mean() * 100), 2),
        'mismatch_type_counts': result_df['mismatch_type'].value_counts().to_dict(),
    }
    scores_path = os.path.join(RESULT_DIR, 'scores.json')
    data = json.load(open(scores_path, encoding='utf-8')) if os.path.exists(scores_path) else []
    data.append(summary)
    json.dump(data, open(scores_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  완료 | 그래프: {FIG_DIR}/ | CSV: results/koclip_similarity_results.csv")
    print(f"{'='*60}")
    return summary


if __name__ == '__main__':
    if not os.path.exists(os.path.join(PROC_DIR, 'test.csv')):
        print("❌ test.csv 없음 → python src/prepro.py 먼저 실행")
        sys.exit(1)
    if not os.path.exists(os.path.join(PROC_DIR, 'image_map.csv')):
        print("❌ image_map.csv 없음 → 이미지 다운로드 확인")
        sys.exit(1)
    run_koclip_analysis()
