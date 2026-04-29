"""
run_midterm.py
==============
중간발표용 전체 실험 자동 실행 스크립트
담당: 안성민

실행 순서
──────────
[로컬] Stage 0: TF-IDF 베이스라인
[로컬] Stage 1: FISA 추론 전용 (비교 기준선)
[GPU]  Stage 2: BERT 감성 fine-tuning
[GPU]  Stage 3: BERT 카테고리 fine-tuning (v2 + v1 ablation)
[GPU]  Stage 4: ResNet Exp-1 + Exp-2
[로컬] Stage 5: 결과 요약

사전 조건
──────────
- data/processed/에 train/val/test.csv 존재 (prepro.py 실행 완료)
- data/processed/image_map.csv 존재
- data/images/ 에 이미지 존재

실행: python src/run_midterm.py
"""
import os, sys, json, time, subprocess

RESULT_DIR = 'results'
os.makedirs(RESULT_DIR, exist_ok=True)

# 실행할 스크립트 경로
SCRIPTS = {
    'tfidf'         : 'src/models/tfidf_baseline.py',
    'fisa'          : 'src/models/fisa_inference.py',
    'bert_sentiment': 'src/models/bert_sentiment.py',
    'bert_category' : 'src/models/bert_category.py',
    'image'         : 'src/models/image_classifier.py',
    'report'        : 'src/midterm_report.py',
}


def check_prerequisites():
    """사전 조건 체크"""
    print(f"\n{'='*55}")
    print(f"  사전 조건 확인")
    print(f"{'='*55}")

    required_files = [
        'data/processed/train.csv',
        'data/processed/val.csv',
        'data/processed/test.csv',
        'data/processed/image_map.csv',
    ]
    all_ok = True
    for path in required_files:
        exists = os.path.exists(path)
        status = '✅' if exists else '❌'
        print(f"  {status} {path}")
        if not exists:
            all_ok = False

    # category_v1 컬럼 확인
    try:
        import pandas as pd
        df = pd.read_csv('data/processed/train.csv', encoding='utf-8-sig', nrows=1)
        has_v1 = 'category_v1' in df.columns
        print(f"  {'✅' if has_v1 else '❌'} category_v1 컬럼 {'있음' if has_v1 else '없음 → prepro.py 재실행 필요'}")
        if not has_v1:
            all_ok = False
    except Exception as e:
        print(f"  ❌ CSV 읽기 실패: {e}")
        all_ok = False

    # GPU 확인
    try:
        import torch
        gpu = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if gpu else 'CPU만 사용 가능'
        print(f"  {'✅' if gpu else '⚠️'} GPU: {name}")
    except:
        print(f"  ⚠️ PyTorch 미설치")

    # 이미지 폴더 확인
    img_count = len([f for f in os.listdir('data/images') if f.endswith('.jpg')]) \
                if os.path.exists('data/images') else 0
    print(f"  {'✅' if img_count > 1000 else '❌'} data/images/: {img_count}장")
    if img_count < 1000:
        all_ok = False

    return all_ok


def run_stage(name: str, script: str, skip_on_no_gpu=False):
    """단일 스테이지 실행"""
    print(f"\n{'='*55}")
    print(f"  실행: {name}")
    print(f"  스크립트: {script}")
    print(f"{'='*55}")

    if not os.path.exists(script):
        print(f"  ❌ 스크립트 없음: {script}")
        return False

    if skip_on_no_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                print(f"  ⚠️ GPU 없음 → {name} 스킵 (RunPod에서 실행)")
                return False
        except:
            print(f"  ⚠️ GPU 확인 불가 → 스킵")
            return False

    t0 = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  ✅ {name} 완료 ({elapsed/60:.1f}분)")
        return True
    else:
        print(f"\n  ❌ {name} 실패 (returncode={result.returncode})")
        return False


def print_summary():
    """최종 결과 요약"""
    path = os.path.join(RESULT_DIR, 'scores.json')
    if not os.path.exists(path):
        print("  결과 없음")
        return

    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n{'='*55}")
    print(f"  최종 성능 요약")
    print(f"{'='*55}")
    print(f"  {'모델':<35} {'Acc':>8}  {'Macro-F1':>10}")
    print(f"  {'-'*55}")

    for r in data:
        name = r.get('model', '?')
        acc  = r.get('accuracy', '-')
        f1   = r.get('macro_f1', '-')
        if isinstance(acc, float):
            print(f"  {name:<35} {acc:>8.4f}  {f1:>10.4f}")
        else:
            print(f"  {name:<35} {'':>8}  {'':>10}")


if __name__ == '__main__':
    import torch

    print(f"\n{'='*55}")
    print(f"  찹쌀떡팀 - 중간발표 실험 파이프라인")
    print(f"{'='*55}")

    # 사전 조건 확인
    if not check_prerequisites():
        print("\n❌ 사전 조건 미충족. 위 항목 해결 후 재실행")
        sys.exit(1)

    gpu_available = torch.cuda.is_available()

    # ── 로컬 실행 스테이지 ─────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  [로컬 스테이지] GPU 불필요 항목")
    print(f"{'─'*55}")

    run_stage('TF-IDF 베이스라인', SCRIPTS['tfidf'])
    run_stage('FISA 추론 전용',    SCRIPTS['fisa'])

    # ── GPU 스테이지 ───────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  [GPU 스테이지] RunPod에서 실행")
    print(f"{'─'*55}")

    if not gpu_available:
        print(f"\n  ⚠️ GPU 없음 → 아래 스테이지는 RunPod에서 실행하세요")
        print(f"  python src/models/bert_sentiment.py")
        print(f"  python src/models/bert_category.py")
        print(f"  python src/models/image_classifier.py")
    else:
        run_stage('BERT 감성 분류',     SCRIPTS['bert_sentiment'], skip_on_no_gpu=True)
        run_stage('BERT 카테고리 분류', SCRIPTS['bert_category'],  skip_on_no_gpu=True)
        run_stage('ResNet 이미지 분류', SCRIPTS['image'],          skip_on_no_gpu=True)

    # ── 결과 시각화 ────────────────────────────────────
    run_stage('결과 시각화', SCRIPTS['report'])

    # ── 최종 요약 ──────────────────────────────────────
    print_summary()
    print(f"\n✅ 중간발표 파이프라인 완료")
    print(f"   results/scores.json → 성능 비교표")
    print(f"   results/figures/    → 시각화 결과")
