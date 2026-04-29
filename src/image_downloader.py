"""
image_downloader.py
===================
크롤링된 기사의 썸네일 이미지 일괄 다운로드
담당: 크롤링 담당자

입력: data/raw/*.csv (image_url 컬럼)
출력: data/images/{article_id}.jpg
      data/processed/image_map.csv  (article_url ↔ local_path 매핑)
      results/time_log.json         (소요 시간 기록)

실행: python src/image_downloader.py
"""
import os
import time
import random
import hashlib
import json
import pandas as pd
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO

RAW_DIR    = 'data/raw'
IMG_DIR    = 'data/images'
PROC_DIR   = 'data/processed'
RESULT_DIR = 'results'
MAP_PATH   = os.path.join(PROC_DIR, 'image_map.csv')

os.makedirs(IMG_DIR,    exist_ok=True)
os.makedirs(PROC_DIR,   exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}

TARGET_SIZE = (224, 224)   # ResNet 입력 크기


# ── 소요 시간 측정 ────────────────────────────────────────────────────────────
class Timer:
    """작업별 소요시간 기록 → results/time_log.json"""
    def __init__(self, log_path: str = 'results/time_log.json'):
        self.log_path = log_path
        self.records = {}
        # 기존 로그 불러오기
        if os.path.exists(log_path):
            with open(log_path, encoding='utf-8') as f:
                self.records = json.load(f)

    def start(self, name: str):
        self.records[name] = {'start': time.time(), 'end': None,
                              'elapsed_sec': None, 'elapsed_min': None}

    def end(self, name: str):
        if name not in self.records:
            print(f"[Timer] '{name}' start() 먼저 호출 필요")
            return
        elapsed = time.time() - self.records[name]['start']
        self.records[name]['end']         = time.time()
        self.records[name]['elapsed_sec'] = round(elapsed, 2)
        self.records[name]['elapsed_min'] = round(elapsed / 60, 2)
        print(f"  ⏱ [{name}] 소요시간: {elapsed / 60:.1f}분 ({elapsed:.1f}초)")
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)


# ── URL → 파일명 변환 ─────────────────────────────────────────────────────────
def url_to_filename(url: str) -> str:
    """URL MD5 해시 → 고유 파일명"""
    h = hashlib.md5(url.encode()).hexdigest()[:12]
    return f"{h}.jpg"


# ── 이미지 단건 다운로드 + 리사이즈 ──────────────────────────────────────────
def download_image(url: str, save_path: str) -> bool:
    """
    이미지 다운로드 → 224×224 JPEG 저장
    Returns: True(성공) / False(실패)
    """
    if not url or not url.startswith('http'):
        return False
    if os.path.exists(save_path):
        return True  # 이미 다운로드됨

    try:
        resp = requests.get(url, headers=HEADERS, timeout=10, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get('Content-Type', '')
        if 'image' not in content_type:
            return False

        img = Image.open(BytesIO(resp.content)).convert('RGB')
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
        img.save(save_path, 'JPEG', quality=90)
        return True

    except Exception as e:
        print(f"  [이미지 실패] {url[:60]}... → {e}")
        return False


# ── 대체 이미지 생성 ──────────────────────────────────────────────────────────
def create_placeholder(save_path: str):
    """이미지 없는 기사용 회색 224×224 플레이스홀더"""
    img = Image.new('RGB', TARGET_SIZE, color=(180, 180, 180))
    img.save(save_path, 'JPEG')


# ── 메인 다운로드 ─────────────────────────────────────────────────────────────
def download_all(delay: float = 0.3) -> pd.DataFrame:
    """
    모든 raw CSV에서 image_url 읽어 다운로드
    image_url 없는 기사 → 플레이스홀더 생성

    Returns
    -------
    DataFrame: url, image_url, local_path, has_image
    """
    timer = Timer()
    timer.start('image_download')

    # ── CSV 로드 ─────────────────────────────────────────
    csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"'{RAW_DIR}/'에 CSV 없음")

    dfs = []
    for fname in csv_files:
        df = pd.read_csv(os.path.join(RAW_DIR, fname), encoding='utf-8-sig')
        if 'image_url' not in df.columns:
            df['image_url'] = ''
        dfs.append(df[['url', 'image_url']].copy())

    df = pd.concat(dfs, ignore_index=True)
    df.drop_duplicates(subset='url', inplace=True)
    df['image_url'] = df['image_url'].fillna('')
    total = len(df)
    print(f"[이미지 다운로더] 총 {total}건")

    # ── 다운로드 루프 ─────────────────────────────────────
    local_paths = []
    has_images  = []
    success = fail = placeholder = 0

    for i, row in df.iterrows():
        fname     = url_to_filename(row['url'])
        save_path = os.path.join(IMG_DIR, fname)

        if row['image_url']:
            ok = download_image(row['image_url'], save_path)
            if ok:
                success += 1
                has_images.append(True)
            else:
                create_placeholder(save_path)
                fail += 1
                has_images.append(False)
        else:
            create_placeholder(save_path)
            placeholder += 1
            has_images.append(False)

        local_paths.append(save_path)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - timer.records['image_download']['start']
            speed   = (i + 1) / elapsed * 60  # 건/분
            remain  = (total - i - 1) / speed if speed > 0 else 0
            print(f"  {i+1}/{total} | 성공={success} 실패={fail} "
                  f"플레이스홀더={placeholder} | "
                  f"경과={elapsed/60:.1f}분 | 잔여≈{remain:.1f}분")

        time.sleep(random.uniform(delay * 0.5, delay * 1.5))

    df['local_path'] = local_paths
    df['has_image']  = has_images

    df.to_csv(MAP_PATH, index=False, encoding='utf-8-sig')
    print(f"\n[완료] 성공={success}  실패={fail}  플레이스홀더={placeholder}")
    print(f"매핑 저장: {MAP_PATH}")

    timer.end('image_download')

    # ── 다운로드 통계 저장 ────────────────────────────────
    stats = {
        'total'      : total,
        'success'    : success,
        'fail'       : fail,
        'placeholder': placeholder,
        'success_rate': round(success / total * 100, 2) if total else 0,
        'elapsed_min' : timer.records['image_download']['elapsed_min'],
    }
    stats_path = os.path.join(RESULT_DIR, 'image_download_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"통계 저장: {stats_path}")

    return df


if __name__ == '__main__':
    download_all(delay=0.3)