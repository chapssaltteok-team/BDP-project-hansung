"""
dynamic_alpha_fusion.py
=======================
방향 B: 동적 α Late Fusion 실험
목적: KoCLIP 유사도가 높은 기사에서만 이미지 가중치를 높여
      멀티모달 성능 향상 가능 여부 검증

실험 구성
──────────
1. 고정 α (기존 방식): α=0.85 고정
2. 동적 α (개선 방식): KoCLIP 유사도 구간별 α 조정
   - 유사도 HIGH (≥0.5) → α=0.60 (이미지 40% 반영)
   - 유사도 MID  (0.25~0.5) → α=0.80 (이미지 20% 반영)
   - 유사도 LOW  (<0.25) → α=0.95 (텍스트 거의 전담)
3. 구간 탐색: 최적 임계값 조합 자동 탐색

실행: python src/models/dynamic_alpha_fusion.py
"""

import os, sys, json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score
from itertools import product

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

PROC_DIR   = 'data/processed'
RESULT_DIR = 'results'
OUTPUT_DIR = 'outputs'

LABEL2ID = {'긍정': 0, '중립': 1, '부정': 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def load_data():
    val_df  = pd.read_csv(os.path.join(PROC_DIR, 'val.csv'),  encoding='utf-8-sig')
    test_df = pd.read_csv(os.path.join(PROC_DIR, 'test.csv'), encoding='utf-8-sig')

    # 확률 벡터 로드
    bert_val    = np.load(os.path.join(OUTPUT_DIR, 'bert_probs_val.npy'))
    bert_test   = np.load(os.path.join(OUTPUT_DIR, 'bert_probs_test.npy'))
    resnet_val  = np.load(os.path.join(OUTPUT_DIR, 'resnet_probs_val.npy'))
    resnet_test = np.load(os.path.join(OUTPUT_DIR, 'resnet_probs_test.npy'))

    # KoCLIP 유사도 로드
    sim_df = pd.read_csv(os.path.join(RESULT_DIR, 'koclip_similarity_results.csv'),
                         encoding='utf-8-sig')

    # val/test 유사도 분리
    val_urls  = val_df['url'].tolist()
    test_urls = test_df['url'].tolist()
    url2sim   = dict(zip(sim_df['url'], sim_df['koclip_similarity']))

    val_sims  = np.array([url2sim.get(u, 0.0) for u in val_urls])
    test_sims = np.array([url2sim.get(u, 0.0) for u in test_urls])

    y_val  = val_df['sentiment_str'].map(LABEL2ID).values
    y_test = test_df['sentiment_str'].map(LABEL2ID).values

    return (bert_val, resnet_val, val_sims, y_val,
            bert_test, resnet_test, test_sims, y_test)


# ── 고정 α 앙상블 ─────────────────────────────────────────────────────────────
def fixed_alpha_ensemble(p_bert, p_resnet, y_true, alpha=0.85):
    p_ens = alpha * p_bert + (1 - alpha) * p_resnet
    preds = p_ens.argmax(axis=1)
    f1  = f1_score(y_true, preds, average='macro', zero_division=0)
    acc = accuracy_score(y_true, preds)
    return f1, acc


# ── 동적 α 앙상블 ─────────────────────────────────────────────────────────────
def dynamic_alpha_ensemble(p_bert, p_resnet, sims, y_true,
                            thresh_high=0.5, thresh_mid=0.25,
                            alpha_high=0.60, alpha_mid=0.80, alpha_low=0.95):
    """
    유사도 구간별 α 적용
    HIGH: 유사도 ≥ thresh_high → alpha_high (이미지 많이 반영)
    MID:  thresh_mid ≤ 유사도 < thresh_high → alpha_mid
    LOW:  유사도 < thresh_mid → alpha_low (텍스트 거의 전담)
    """
    alphas = np.where(sims >= thresh_high, alpha_high,
             np.where(sims >= thresh_mid,  alpha_mid, alpha_low))

    p_ens = alphas[:, None] * p_bert + (1 - alphas[:, None]) * p_resnet
    preds = p_ens.argmax(axis=1)
    f1  = f1_score(y_true, preds, average='macro', zero_division=0)
    acc = accuracy_score(y_true, preds)

    # 구간별 통계
    n_high = (sims >= thresh_high).sum()
    n_mid  = ((sims >= thresh_mid) & (sims < thresh_high)).sum()
    n_low  = (sims < thresh_mid).sum()

    return f1, acc, n_high, n_mid, n_low


# ── 최적 파라미터 탐색 ────────────────────────────────────────────────────────
def search_best_params(p_bert, p_resnet, sims, y_true):
    """검증셋에서 최적 임계값 + α 조합 탐색"""
    thresh_highs = [0.40, 0.50, 0.60, 0.70]
    thresh_mids  = [0.20, 0.25, 0.30]
    alpha_highs  = [0.50, 0.60, 0.70]
    alpha_mids   = [0.75, 0.80, 0.85]
    alpha_lows   = [0.90, 0.95, 1.00]

    best_f1     = -1
    best_params = {}
    results     = []

    for th, tm, ah, am, al in product(thresh_highs, thresh_mids,
                                       alpha_highs, alpha_mids, alpha_lows):
        if th <= tm:
            continue
        f1, acc, nh, nm, nl = dynamic_alpha_ensemble(
            p_bert, p_resnet, sims, y_true, th, tm, ah, am, al)
        results.append({
            'thresh_high': th, 'thresh_mid': tm,
            'alpha_high': ah, 'alpha_mid': am, 'alpha_low': al,
            'macro_f1': round(f1, 4), 'accuracy': round(acc, 4),
            'n_high': int(nh), 'n_mid': int(nm), 'n_low': int(nl)
        })
        if f1 > best_f1:
            best_f1 = f1
            best_params = results[-1].copy()

    return best_params, sorted(results, key=lambda x: -x['macro_f1'])[:10]


# ── 메인 실행 ─────────────────────────────────────────────────────────────────
def run_dynamic_alpha():
    print(f"\n{'='*60}")
    print(f"  동적 α Late Fusion 실험")
    print(f"{'='*60}")

    (bert_val, resnet_val, val_sims, y_val,
     bert_test, resnet_test, test_sims, y_test) = load_data()

    print(f"  val: {len(y_val)}건 | test: {len(y_test)}건")
    print(f"  val 유사도 — 평균: {val_sims.mean():.4f} | 중앙값: {np.median(val_sims):.4f}")

    # ── 1. BERT 단독 ─────────────────────────────────────
    bert_f1  = f1_score(y_test, bert_test.argmax(axis=1), average='macro', zero_division=0)
    bert_acc = accuracy_score(y_test, bert_test.argmax(axis=1))

    # ── 2. ResNet 단독 ───────────────────────────────────
    resnet_f1  = f1_score(y_test, resnet_test.argmax(axis=1), average='macro', zero_division=0)
    resnet_acc = accuracy_score(y_test, resnet_test.argmax(axis=1))

    # ── 3. 고정 α=0.85 (기존) ────────────────────────────
    fixed_f1, fixed_acc = fixed_alpha_ensemble(bert_test, resnet_test, y_test, alpha=0.85)

    # ── 4. 최적 α 탐색 (검증셋 기준) ─────────────────────
    print(f"\n  최적 파라미터 탐색 중 (검증셋)...")
    best_params, top10 = search_best_params(bert_val, resnet_val, val_sims, y_val)
    print(f"  탐색 완료 — 최적: {best_params}")

    # ── 5. 최적 파라미터로 테스트셋 평가 ─────────────────
    dyn_f1, dyn_acc, nh, nm, nl = dynamic_alpha_ensemble(
        bert_test, resnet_test, test_sims, y_test,
        best_params['thresh_high'], best_params['thresh_mid'],
        best_params['alpha_high'],  best_params['alpha_mid'],
        best_params['alpha_low']
    )

    # ── 6. 유사도 HIGH 케이스만 별도 평가 ────────────────
    high_mask = test_sims >= best_params['thresh_high']
    if high_mask.sum() > 0:
        hi_f1_bert   = f1_score(y_test[high_mask], bert_test[high_mask].argmax(axis=1),
                                 average='macro', zero_division=0)
        hi_f1_dyn    = f1_score(y_test[high_mask],
                                 (best_params['alpha_high'] * bert_test[high_mask] +
                                  (1-best_params['alpha_high']) * resnet_test[high_mask]).argmax(axis=1),
                                 average='macro', zero_division=0)
    else:
        hi_f1_bert = hi_f1_dyn = 0.0

    # ── 결과 출력 ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  최종 성능 비교 (테스트셋)")
    print(f"{'='*60}")
    print(f"  {'모델':<30} {'Accuracy':>10}  {'Macro-F1':>10}")
    print(f"  {'-'*54}")
    print(f"  {'BERT 단독':<30} {bert_acc:>10.4f}  {bert_f1:>10.4f}")
    print(f"  {'ResNet 단독':<30} {resnet_acc:>10.4f}  {resnet_f1:>10.4f}")
    print(f"  {'Late Fusion 고정 α=0.85':<30} {fixed_acc:>10.4f}  {fixed_f1:>10.4f}")
    print(f"  {'Late Fusion 동적 α':<30} {dyn_acc:>10.4f}  {dyn_f1:>10.4f}  ← 개선")

    print(f"\n  [동적 α 최적 파라미터]")
    print(f"  thresh_high={best_params['thresh_high']} / thresh_mid={best_params['thresh_mid']}")
    print(f"  alpha_high={best_params['alpha_high']} / alpha_mid={best_params['alpha_mid']} / alpha_low={best_params['alpha_low']}")
    print(f"\n  [유사도 구간별 기사 수 (테스트셋)]")
    print(f"  HIGH (≥{best_params['thresh_high']}): {nh}건 → α={best_params['alpha_high']} (이미지 {(1-best_params['alpha_high'])*100:.0f}% 반영)")
    print(f"  MID  ({best_params['thresh_mid']}~{best_params['thresh_high']}): {nm}건 → α={best_params['alpha_mid']} (이미지 {(1-best_params['alpha_mid'])*100:.0f}% 반영)")
    print(f"  LOW  (<{best_params['thresh_mid']}): {nl}건 → α={best_params['alpha_low']} (이미지 {(1-best_params['alpha_low'])*100:.0f}% 반영)")

    print(f"\n  [유사도 HIGH 케이스 단독 분석 ({nh}건)]")
    print(f"  BERT 단독 Macro-F1:    {hi_f1_bert:.4f}")
    print(f"  동적 α 앙상블 Macro-F1: {hi_f1_dyn:.4f}")
    diff = dyn_f1 - fixed_f1
    print(f"\n  동적 α vs 고정 α 차이: {diff:+.4f}")
    if diff > 0:
        print(f"  → 이미지 유사도 활용 시 성능 향상 확인")
    else:
        print(f"  → KBO 뉴스 도메인에서 이미지 신호 자체가 제한적임을 재확인")

    # ── 결과 저장 ─────────────────────────────────────────
    summary = {
        'model'           : 'DynamicAlpha_LateFusion',
        'bert_f1'         : round(bert_f1, 4),
        'resnet_f1'       : round(resnet_f1, 4),
        'fixed_alpha_f1'  : round(fixed_f1, 4),
        'dynamic_alpha_f1': round(dyn_f1, 4),
        'improvement'     : round(dyn_f1 - fixed_f1, 4),
        'best_params'     : best_params,
        'top10_val_params': top10,
        'high_case_analysis': {
            'n_high'       : int(nh),
            'bert_f1'      : round(hi_f1_bert, 4),
            'dynamic_f1'   : round(hi_f1_dyn, 4),
        }
    }

    scores_path = os.path.join(RESULT_DIR, 'scores.json')
    data = json.load(open(scores_path, encoding='utf-8')) if os.path.exists(scores_path) else []
    data.append(summary)
    json.dump(data, open(scores_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    result_path = os.path.join(RESULT_DIR, 'dynamic_alpha_results.json')
    json.dump(summary, open(result_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {result_path}")

    return summary


if __name__ == '__main__':
    for f in ['bert_probs_val.npy', 'bert_probs_test.npy',
              'resnet_probs_val.npy', 'resnet_probs_test.npy']:
        if not os.path.exists(os.path.join(OUTPUT_DIR, f)):
            print(f"❌ {f} 없음 → outputs/ 폴더에 npy 파일 넣어주세요")
            sys.exit(1)
    if not os.path.exists(os.path.join(RESULT_DIR, 'koclip_similarity_results.csv')):
        print("❌ koclip_similarity_results.csv 없음 → koclip_analysis.py 먼저 실행")
        sys.exit(1)

    run_dynamic_alpha()
