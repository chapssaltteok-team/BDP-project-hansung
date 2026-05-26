"""
data_loader.py
==============
train/val/test.csv 로드 및 공통 전처리.

두 실험(TF-IDF / BERT) 이 동일한 데이터를 공유하므로
로드 로직을 한 곳에 모아 중복 코드를 제거한다.

반환 구조
─────────
load_splits() → {'train': df, 'val': df, 'test': df}
load_full()   → df (train+val+test 합본, 전체 분포 분석용)

각 df 에 보장되는 컬럼
  text       : title_clean + ' ' + body_clean (없으면 자동 생성)
  category   : v2 카테고리 (CAT_CLASSES 에 없으면 '기타' 로 대체)
  cat_id     : category 의 정수 인코딩 (0~5)
"""

import os
import pandas as pd
from config import PROC_DIR, CAT_LABEL2ID, CAT_CLASSES


def _clean_category(cat: str) -> str:
    """CAT_CLASSES 에 없는 카테고리 값을 '기타' 로 대체한다."""
    return cat if cat in CAT_LABEL2ID else '기타'


def _ensure_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    'text' 컬럼이 없으면 title_clean + body_clean 으로 생성한다.
    prepro.py 가 이미 text 컬럼을 만들어두지만,
    혹시 없을 경우를 대비한 방어 코드.
    """
    if 'text' not in df.columns:
        df = df.copy()
        df['text'] = (df.get('title_clean', pd.Series([''] * len(df))).fillna('')
                      + ' '
                      + df.get('body_clean', pd.Series([''] * len(df))).fillna(''))
    return df


def _add_cat_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    category 컬럼 정제 + cat_id(정수) 컬럼 추가.
    cat_id 는 evaluator 와 visualizer 에서 행렬 인덱싱에 사용한다.
    """
    df = df.copy()
    df['category'] = df['category'].apply(_clean_category)
    df['cat_id']   = df['category'].map(CAT_LABEL2ID).fillna(5).astype(int)
    return df


def load_splits() -> dict:
    """
    train/val/test.csv 를 각각 읽어 dict 로 반환한다.

    반환 : {'train': df, 'val': df, 'test': df}

    이 실험은 카테고리 배정 속도/정확도 측정이 목적이므로
    split 구분이 크게 중요하지 않지만, 기존 파이프라인과
    일관성을 유지하기 위해 split 구조를 그대로 유지한다.
    실제 centroid 계산은 train 에서만, 평가는 test 에서 수행한다.
    """
    splits = {}
    for split in ['train', 'val', 'test']:
        path = os.path.join(PROC_DIR, f'{split}.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'{path} 없음 — prepro.py 를 먼저 실행하세요')
        df = pd.read_csv(path, encoding='utf-8-sig')
        df = _ensure_text(df)
        df = _add_cat_id(df)
        splits[split] = df

        dist = df['category'].value_counts().to_dict()
        total = len(df)
        print(f'  [{split}] {total}건  카테고리={dist}')

    return splits


def load_full() -> pd.DataFrame:
    """
    train+val+test 를 합쳐 전체 데이터프레임 반환.
    centroid 안정성 측정처럼 전체 분포를 대상으로
    샘플링을 반복할 때 사용한다.
    """
    splits = load_splits()
    df = pd.concat(splits.values(), ignore_index=True)
    print(f'  [전체] {len(df)}건')
    return df
