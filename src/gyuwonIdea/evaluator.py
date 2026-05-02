import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from config import CAT_CLASSES


def run_assignment(df_target: pd.DataFrame,
                   sim_matrix: np.ndarray,
                   theta: float,
                   slow_preds=None,
                   slow_ms_per: float = 0.0) -> dict:
    """
    fast path  : centroid similarity argmax
    slow path  : 별도 classifier 예측값 사용

    slow_preds가 None이면 기존 방식처럼 centroid argmax를 fallback으로 사용한다.
    slow_ms_per는 slow classifier 1건당 평균 추론 시간(ms)이다.
    """
    t0 = time.time()

    n = sim_matrix.shape[0]
    preds = []
    fast_mask = []

    for i in range(n):
        sims = sim_matrix[i]
        best_i = int(np.argmax(sims))
        max_sim = float(sims[best_i])

        if max_sim >= theta:
            preds.append(best_i)
            fast_mask.append(True)
        else:
            if slow_preds is not None:
                preds.append(int(slow_preds[i]))
            else:
                preds.append(best_i)
            fast_mask.append(False)

    routing_elapsed = time.time() - t0

    fast_count = sum(fast_mask)
    slow_count = n - fast_count
    fast_ratio = fast_count / n if n else 0.0

    # 실제 서비스 가정: slow path 대상만 classifier 비용이 추가됨
    total_elapsed = routing_elapsed + (slow_count * slow_ms_per / 1000.0)
    ms_per = total_elapsed / n * 1000 if n else 0.0

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


def evaluate(df_target: pd.DataFrame,
             assignment: dict) -> dict:
    trues = df_target['cat_id'].tolist()
    preds = assignment['preds']
    fast_mask = assignment['fast_mask']

    macro_f1_all = f1_score(trues, preds, average='macro', zero_division=0)

    fast_trues = [t for t, f in zip(trues, fast_mask) if f]
    fast_preds = [p for p, f in zip(preds, fast_mask) if f]
    macro_f1_fast = (
        f1_score(fast_trues, fast_preds, average='macro', zero_division=0)
        if fast_trues else 0.0
    )

    slow_trues = [t for t, f in zip(trues, fast_mask) if not f]
    slow_preds = [p for p, f in zip(preds, fast_mask) if not f]
    macro_f1_slow = (
        f1_score(slow_trues, slow_preds, average='macro', zero_division=0)
        if slow_trues else 0.0
    )

    report = classification_report(
        trues,
        preds,
        target_names=CAT_CLASSES,
        output_dict=True,
        zero_division=0
    )

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


def build_result_record(method: str,
                        strategy: str,
                        theta_results: list,
                        stability: dict) -> dict:
    return {
        'method'        : method,
        'strategy'      : strategy,
        'stability'     : {
            'mean_var'    : stability['mean_var'],
            'per_cat_var' : stability['per_cat_var'],
        },
        'theta_results' : theta_results,
    }


def extract_matrix(records: list, metric: str) -> dict:
    result = {}
    for rec in records:
        strategy = rec['strategy']
        vals = [tr[metric] for tr in rec['theta_results']]
        result[strategy] = vals
    return result