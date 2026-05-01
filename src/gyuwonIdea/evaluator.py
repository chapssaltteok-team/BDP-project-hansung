"""
evaluator.py
============
TF-IDF / BERT 두 실험에서 공통으로 사용하는 평가 지표 계산 모듈.

평가 항목
──────────
1. Macro-F1          : fast+slow path 최종 배정 결과 vs 정답(category 컬럼)
2. θ 통과율          : fast path 로 처리된 비율 (전체 대비)
3. 처리 속도(ms/건)  : 전체 배정 소요시간 / 전체 건수
4. Centroid 안정성   : sampling_strategy.measure_stability() 결과 활용

정답 기준
──────────
prepro.py 가 키워드 기반으로 생성한 category(v2) 컬럼을 정답으로 사용한다.
BERT fine-tuning 결과(gyuwon_categoryNums.py)가 있다면 그것을 정답으로
대체해도 되지만 이 파일은 독립 실행을 위해 CSV 컬럼만 사용한다.

"키워드 기반 정답이 완벽하지 않다"는 한계는 보고서 한계 절에 명시한다.
이 실험의 목적이 절대적 정확도가 아니라 "전략 간 상대 비교"이므로
동일 기준으로 모든 전략을 평가하면 공정성이 보장된다.
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from config import CAT_CLASSES, CAT_LABEL2ID, THETA_LIST


# ──────────────────────────────────────────────────────────────────────────────
# 1. fast / slow path 분류 실행 및 속도 측정
# ──────────────────────────────────────────────────────────────────────────────

def run_assignment(df_target: pd.DataFrame,
                   sim_matrix: np.ndarray,
                   theta: float) -> dict:
    """
    전체 대상 데이터(df_target)에 대해 fast/slow path 카테고리 배정 수행.

    sim_matrix : (len(df_target), NUM_CATS) 코사인 유사도 행렬.
                 각 행 = 해당 기사와 카테고리별 centroid 의 유사도 벡터.
                 tfidf_centroid.py 또는 bert_centroid.py 에서 계산해 전달.

    배정 로직
    ──────────
    각 기사에 대해 가장 높은 유사도(max_sim)를 찾는다.
      max_sim >= theta → fast path : 해당 카테고리로 즉시 배정
      max_sim <  theta → slow path : 실제로는 BERT 로 재처리해야 하지만
                                     이 평가 모듈에서는 "max_sim 기준 최고 카테고리"
                                     를 배정해 정확도 하한을 추정한다.
                                     (slow path 대상 수를 별도 집계)

    slow path 를 실제 BERT 로 처리하지 않는 이유 :
      이 실험의 목적은 "θ 값과 전략에 따른 정확도-속도 트레이드오프 측정"이다.
      slow path 에서 BERT 로 재처리하면 결국 BERT 전체 처리와 같아져
      실험 의미가 사라진다. 대신 slow path 비율을 별도 지표로 보고해
      "θ 가 낮으면 fast path 비율이 높지만 slow path 오류 위험도 있다"는
      트레이드오프를 정량화한다.

    반환
    ────
    {
      'preds'          : list[int] — 예측 cat_id
      'fast_mask'      : list[bool] — fast path 여부
      'fast_ratio'     : float — fast path 비율
      'elapsed_ms_per' : float — 건당 평균 처리 시간(ms)
      'theta'          : float
    }
    """
    t0 = time.time()

    n         = sim_matrix.shape[0]
    preds     = []
    fast_mask = []

    for i in range(n):
        sims    = sim_matrix[i]           # (NUM_CATS,) 유사도 벡터
        best_i  = int(np.argmax(sims))
        max_sim = float(sims[best_i])

        preds.append(best_i)
        fast_mask.append(max_sim >= theta)

    elapsed      = time.time() - t0
    ms_per       = elapsed / n * 1000    # ms/건

    fast_ratio   = sum(fast_mask) / n
    slow_count   = n - sum(fast_mask)

    print(f'    θ={theta}  fast={fast_ratio*100:.1f}%  '
          f'slow={slow_count}건  속도={ms_per:.3f}ms/건')

    return {
        'preds'          : preds,
        'fast_mask'      : fast_mask,
        'fast_ratio'     : fast_ratio,
        'slow_count'     : slow_count,
        'elapsed_ms_per' : ms_per,
        'theta'          : theta,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. 정확도 평가
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(df_target: pd.DataFrame,
             assignment: dict) -> dict:
    """
    run_assignment() 결과를 정답(df_target['cat_id'])과 비교해 성능 지표 계산.

    fast path 전체 / slow path 전용 / 전체 세 가지 시각으로 평가해
    "fast path 의 정확도가 slow path 보다 높은가"를 검증한다.
    → fast path 정확도가 높으면 "임계값 이상의 기사는 쉬운 케이스"라는 증거.
    → slow path 정확도가 낮으면 "임계값 미만은 실제로 BERT 재처리가 필요"한 근거.
    """
    trues       = df_target['cat_id'].tolist()
    preds       = assignment['preds']
    fast_mask   = assignment['fast_mask']

    # 전체 Macro-F1
    macro_f1_all = f1_score(trues, preds, average='macro', zero_division=0)

    # fast path 만
    fast_trues = [t for t, f in zip(trues, fast_mask) if f]
    fast_preds = [p for p, f in zip(preds, fast_mask) if f]
    macro_f1_fast = (f1_score(fast_trues, fast_preds, average='macro', zero_division=0)
                     if fast_trues else 0.0)

    # slow path 만
    slow_trues = [t for t, f in zip(trues, fast_mask) if not f]
    slow_preds = [p for p, f in zip(preds, fast_mask) if not f]
    macro_f1_slow = (f1_score(slow_trues, slow_preds, average='macro', zero_division=0)
                     if slow_trues else 0.0)

    # 클래스별 report (전체)
    report = classification_report(
        trues, preds, target_names=CAT_CLASSES,
        output_dict=True, zero_division=0)

    return {
        'theta'           : assignment['theta'],
        'macro_f1_all'    : round(macro_f1_all, 4),
        'macro_f1_fast'   : round(macro_f1_fast, 4),
        'macro_f1_slow'   : round(macro_f1_slow, 4),
        'fast_ratio'      : round(assignment['fast_ratio'], 4),
        'slow_count'      : assignment['slow_count'],
        'elapsed_ms_per'  : round(assignment['elapsed_ms_per'], 4),
        'report'          : report,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. 전략 × θ 전체 결과 집계
# ──────────────────────────────────────────────────────────────────────────────

def build_result_record(method: str, strategy: str,
                        theta_results: list,
                        stability: dict) -> dict:
    """
    단일 (method, strategy) 조합의 전체 결과를 하나의 dict 로 묶는다.
    JSON 저장 및 visualizer 입력 형식으로 사용된다.

    method   : 'tfidf' | 'bert'
    strategy : 'sequential' | 'random' | 'stratified'
    theta_results : [evaluate() 반환값, ...] (θ 개수만큼)
    stability     : measure_stability() 반환값
    """
    return {
        'method'        : method,
        'strategy'      : strategy,
        'stability'     : {
            'mean_var'    : stability['mean_var'],
            'per_cat_var' : stability['per_cat_var'],
        },
        'theta_results' : theta_results,  # θ 별 평가 결과 리스트
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. 결과 행렬 추출 (시각화용 편의 함수)
# ──────────────────────────────────────────────────────────────────────────────

def extract_matrix(records: list, metric: str) -> dict:
    """
    records 리스트에서 특정 metric 을 전략 × θ 행렬로 추출한다.

    metric : 'macro_f1_all' | 'fast_ratio' | 'elapsed_ms_per' 등

    반환 : {strategy: [θ별 값, ...]}
    → visualizer 의 히트맵/꺾은선 그래프에서 바로 사용 가능.
    """
    result = {}
    for rec in records:
        strategy = rec['strategy']
        vals = [tr[metric] for tr in rec['theta_results']]
        result[strategy] = vals
    return result
