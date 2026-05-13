"""
main.py
=======
gyuwonIdea 실험 전체 진입점.

실행 : python src/gyuwonIdea/main.py
       (프로젝트 루트 BDP-project-hansung/ 에서 실행)

실행 흐름
──────────
[1] 데이터 로드
[2] 실험 A : TF-IDF Centroid (CPU, 항상 실행)
[3] 실험 B : BERT Centroid   (GPU 있을 때만 실행, 없으면 자동 스킵)
[4] 시각화 생성
[5] 결과 저장 (gyuwon_idea_scores.json + scores.json append)
[6] 콘솔 요약 출력

GPU 유무에 따른 자동 분기
──────────────────────────
GPU 없음 (로컬) : 실험 A만 완료 → 그래프 4종 (①③④⑤) 생성
GPU 있음 (RunPod): A+B 모두 완료 → 그래프 6종 전체 생성
"""

import os, sys, json, time
import torch

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
# 이 파일은 src/gyuwonIdea/main.py 에 위치한다.
# 프로젝트 루트를 sys.path 에 추가해 config, data_loader 등을 import 할 수 있게 한다.
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))  # src/gyuwonIdea/
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
sys.path.insert(0, _SCRIPT_DIR)     # gyuwonIdea 모듈 인식
sys.path.insert(0, _PROJECT_ROOT)   # 프로젝트 루트 인식 (data/, results/ 접근)
os.chdir(_PROJECT_ROOT)             # 작업 디렉토리를 루트로 변경

from config import (RESULT_DIR, FIG_DIR, JSON_PATH, SCORES_PATH,
                    THETA_LIST, STRATEGY_KEYS, STRATEGY_LABELS)
from data_loader import load_splits
from tfidf_centroid import run_tfidf_experiment
from bert_centroid import run_bert_experiment
from visualizer import run_all_plots


# ──────────────────────────────────────────────────────────────────────────────
# 결과 저장
# ──────────────────────────────────────────────────────────────────────────────

def save_results(tfidf_records: list, bert_records: list = None):
    """
    전체 실험 결과를 gyuwon_idea_scores.json 에 저장하고
    팀 공유 scores.json 에도 append 한다.

    centroids_all (numpy 배열) 은 JSON 직렬화 불가이므로 stability 에서 제외.
    """
    def clean_record(rec: dict) -> dict:
        """JSON 직렬화 가능한 형태로 정제"""
        cleaned = {
            'method'  : rec['method'],
            'strategy': rec['strategy'],
            'stability': {
                'mean_var'   : rec['stability']['mean_var'],
                'per_cat_var': rec['stability']['per_cat_var'],
            },
            'theta_results': [
                {k: v for k, v in tr.items() if k != 'report'}
                for tr in rec['theta_results']
            ],
        }
        return cleaned

    all_records = tfidf_records + (bert_records or [])
    cleaned     = [clean_record(r) for r in all_records]

    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f'  → {JSON_PATH}')

    # scores.json append (팀 공유)
    existing = []
    if os.path.exists(SCORES_PATH):
        with open(SCORES_PATH, encoding='utf-8') as f:
            existing = json.load(f)

    for rec in all_records:
        # θ=0.5, stratified 기준 대표 수치 추출
        strat_idx = STRATEGY_KEYS.index('stratified')
        theta_idx = THETA_LIST.index(0.5) if 0.5 in THETA_LIST else 1
        rep = rec['theta_results'][theta_idx]

        existing.append({
            'model'        : f"FastCat_{rec['method'].upper()}_{rec['strategy']}",
            'experiment_id': 'D-1-FAST-CATEGORY',
            'task'         : 'category_fast_assign',
            'method'       : rec['method'],
            'strategy'     : rec['strategy'],
            'accuracy'     : rep['macro_f1_all'],    # 편의상 accuracy 키에 F1 저장
            'macro_f1'     : rep['macro_f1_all'],
            'fast_ratio'   : rep['fast_ratio'],
            'ms_per_sample': rep['elapsed_ms_per'],
            'stability_var': rec['stability']['mean_var'],
            'notes'        : f'θ=0.5 stratified 기준 대표 수치',
        })

    with open(SCORES_PATH, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f'  → {SCORES_PATH} (append 완료)')


# ──────────────────────────────────────────────────────────────────────────────
# 콘솔 요약
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(tfidf_records: list, bert_records: list = None):
    """
    전략 × θ 별 Macro-F1 + Fast Path 비율을 표 형태로 출력.
    발표 준비 시 수치를 빠르게 확인하는 용도.
    """
    print(f'\n{"="*70}')
    print('  gyuwonIdea 실험 결과 요약')
    print(f'{"="*70}')

    for records, method in [(tfidf_records, 'TF-IDF')] + (
            [(bert_records, 'BERT')] if bert_records else []):
        print(f'\n  ── {method} Centroid ──────────────────────────────')
        print(f'  {"전략":<20} {"θ":>5}  {"Macro-F1":>9}  '
              f'{"Fast%":>7}  {"ms/건":>7}  {"안정성(분산)":>12}')
        print(f'  {"-"*65}')
        for rec in records:
            stab = rec['stability']['mean_var']
            for tr in rec['theta_results']:
                print(f'  {STRATEGY_LABELS[rec["strategy"]].replace(chr(10)," "):<20} '
                      f'{tr["theta"]:>5.1f}  '
                      f'{tr["macro_f1_all"]:>9.4f}  '
                      f'{tr["fast_ratio"]*100:>6.1f}%  '
                      f'{tr["elapsed_ms_per"]:>7.3f}  '
                      f'{stab:>12.6f}')

    # 핵심 발견 자동 출력
    print(f'\n  ── 핵심 발견 ─────────────────────────────────────────')

    # TF-IDF 기준 최고 F1 전략 찾기
    best_f1    = 0.0
    best_label = ''
    for rec in tfidf_records:
        for tr in rec['theta_results']:
            if tr['macro_f1_all'] > best_f1:
                best_f1    = tr['macro_f1_all']
                best_label = f'{rec["strategy"]} θ={tr["theta"]}'

    # Sequential vs Stratified 안정성 비교
    seq_var   = next(r['stability']['mean_var']
                     for r in tfidf_records if r['strategy'] == 'sequential')
    strat_var = next(r['stability']['mean_var']
                     for r in tfidf_records if r['strategy'] == 'stratified')

    print(f'  TF-IDF 최고 Macro-F1 : {best_f1:.4f}  ({best_label})')
    print(f'  Sequential 안정성    : {seq_var:.6f}')
    print(f'  Stratified 안정성    : {strat_var:.6f}')
    print(f'  → Stratified가 {"더 안정적" if strat_var < seq_var else "더 불안정"} '
          f'(분산 차이 {abs(seq_var - strat_var):.6f})')

    if bert_records:
        bert_strat = next(r for r in bert_records if r['strategy'] == 'stratified')
        tfidf_strat = next(r for r in tfidf_records if r['strategy'] == 'stratified')
        theta_idx = THETA_LIST.index(0.5) if 0.5 in THETA_LIST else 1
        f1_diff = (bert_strat['theta_results'][theta_idx]['macro_f1_all']
                   - tfidf_strat['theta_results'][theta_idx]['macro_f1_all'])
        print(f'  BERT vs TF-IDF Macro-F1 차이 : {f1_diff:+.4f} (θ=0.5, Stratified)')


# ──────────────────────────────────────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t_total = time.time()

    print(f'\n{"="*60}')
    print('  gyuwonIdea — 빠른 카테고리 배정 실험')
    print('  D-1: 초기 집중 추출 + 임계값 기반 분산처리')
    print(f'  실험 A: TF-IDF Centroid (CPU)')
    print(f'  실험 B: BERT Centroid   (GPU)')
    print(f'{"="*60}')

    # ── [1] 데이터 로드 ───────────────────────────────────────────────────────
    print('\n[1] 데이터 로드')
    splits = load_splits()

    # ── [2] 실험 A : TF-IDF (항상 실행) ──────────────────────────────────────
    print('\n[2] 실험 A : TF-IDF Centroid')
    tfidf_records = run_tfidf_experiment(splits)

    # ── [3] 실험 B : BERT (GPU 있을 때만) ────────────────────────────────────
    bert_records = None
    gpu_available = torch.cuda.is_available()

    print(f'\n[3] 실험 B : BERT Centroid')
    if gpu_available:
        print(f'  GPU 감지: {torch.cuda.get_device_name(0)}')
        bert_records = run_bert_experiment(splits)
    else:
        print('  ⚠️  GPU 없음 → 실험 B 스킵 (RunPod 에서 실행하세요)')
        print('     TF-IDF 결과만으로 시각화 및 저장을 진행합니다.')

    # ── [4] 시각화 ───────────────────────────────────────────────────────────
    print('\n[4] 시각화 생성')
    run_all_plots(tfidf_records, bert_records)

    # ── [5] 결과 저장 ─────────────────────────────────────────────────────────
    print('\n[5] 결과 저장')
    save_results(tfidf_records, bert_records)

    # ── [6] 요약 ──────────────────────────────────────────────────────────────
    print_summary(tfidf_records, bert_records)

    elapsed = time.time() - t_total
    print(f'\n✅ 전체 실험 완료  소요시간={elapsed/60:.1f}분')
    print(f'   그래프 → {FIG_DIR}/')
    print(f'   수치   → {JSON_PATH}')
