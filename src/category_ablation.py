"""
category_ablation.py
====================
카테고리 개수(3~15개) 변화에 따른 TF-IDF 분류 성능 실험
담당: 이윤서

실험 방식
─────────
방법 1 (자동): 전체 키워드를 n등분하여 카테고리 자동 생성
방법 2 (수동): 미리 정의된 세부 카테고리 딕셔너리 사용

실행: python src/category_ablation.py
"""

import os, json, time
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

PROC_DIR   = 'data/processed'
RESULT_DIR = 'results'
FIG_DIR    = os.path.join(RESULT_DIR, 'figures')
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)

# ── 한글 폰트 설정 ─────────────────────────────────────────────────────────────
def set_korean_font():
    """macOS/Linux 환경에서 한글 폰트 자동 설정"""
    font_candidates = [
        '/System/Library/Fonts/AppleSDGothicNeo.ttc',       # macOS
        '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',  # Linux
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for path in font_candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()


# ════════════════════════════════════════════════════════════════
# 방법 2: 수동 정의 카테고리 딕셔너리 (3~15개)
# ════════════════════════════════════════════════════════════════
MANUAL_CATEGORY_MAP = {
    3: {
        '타격·투구'  : ['홈런','안타','타율','타점','삼진','방어율','선발','불펜','마무리','완봉','세이브'],
        '선수·운영'  : ['트레이드','이적','FA','자유계약','영입','방출','감독','코치','구단','단장'],
        '부상·기타'  : ['부상','수술','재활','결장','심판','오심','판정'],
    },
    4: {
        '타격'       : ['홈런','안타','타율','타점','출루율','장타율','득점','타선'],
        '투구'       : ['삼진','방어율','선발','불펜','마무리','완봉','세이브','홀드','이닝'],
        '선수이동'   : ['트레이드','이적','FA','자유계약','영입','방출','계약','연봉'],
        '운영·기타'  : ['감독','코치','구단','부상','수술','재활','심판'],
    },
    5: {
        '타격'       : ['홈런','안타','타율','타점','출루율','장타율','득점','타선'],
        '투구'       : ['삼진','방어율','선발','불펜','마무리','완봉','세이브','홀드','이닝'],
        '선수이동'   : ['트레이드','이적','FA','자유계약','영입','방출','계약','연봉'],
        '감독·운영'  : ['감독','코치','구단','단장','프런트','스프링캠프'],
        '부상·기타'  : ['부상','수술','재활','결장','심판','오심'],
    },
    6: {  # 기존 v2
        '타격'       : ['홈런','안타','타율','타점','출루율','장타율','타격','배팅','득점','타선'],
        '투구'       : ['삼진','방어율','선발','불펜','마무리','투구','완봉','세이브','홀드','이닝'],
        '선수이동'   : ['트레이드','이적','FA','자유계약','영입','방출','계약','다년계약','연봉'],
        '부상'       : ['부상','수술','재활','결장','부상우려','통증'],
        '감독·운영'  : ['감독','코치','구단','단장','프런트','스프링캠프'],
        '기타'       : ['심판','오심','판정','비디오판독'],
    },
    7: {
        '홈런·장타'  : ['홈런','장타','장타율','만루홈런','솔로홈런'],
        '안타·타율'  : ['안타','타율','타점','출루율','득점','타선','배팅'],
        '선발투수'   : ['선발','이닝','방어율','완봉','완투','퀄리티스타트'],
        '불펜·마무리': ['불펜','마무리','세이브','홀드','중계'],
        '선수이동'   : ['트레이드','이적','FA','자유계약','영입','방출','계약','연봉'],
        '부상·재활'  : ['부상','수술','재활','결장','통증'],
        '운영·기타'  : ['감독','코치','구단','심판','오심'],
    },
    8: {  # 기존 v1 기반
        '타격'       : ['홈런','안타','타율','타점','출루율','득점'],
        '투구'       : ['삼진','방어율','선발','완봉','세이브','이닝'],
        '트레이드'   : ['트레이드','이적','교환','영입'],
        '부상'       : ['부상','수술','재활','결장'],
        'FA'         : ['FA','자유계약','다년계약','연봉','잔류'],
        '감독'       : ['감독','코치','벤치','단장','구단'],
        '심판'       : ['심판','오심','판정','비디오판독'],
        '기타'       : ['스프링캠프','개막','시범경기','올스타'],
    },
    9: {
        '홈런'       : ['홈런','만루홈런','솔로홈런','투런','쓰리런'],
        '타격'       : ['안타','타율','타점','출루율','장타율','배팅','타선'],
        '선발투수'   : ['선발','이닝','방어율','완봉','완투'],
        '구원투수'   : ['불펜','마무리','세이브','홀드','중계','삼진'],
        '트레이드'   : ['트레이드','이적','교환','영입'],
        'FA·계약'    : ['FA','자유계약','다년계약','연봉','잔류'],
        '부상'       : ['부상','수술','재활','결장','통증'],
        '감독·운영'  : ['감독','코치','구단','단장','프런트'],
        '심판·기타'  : ['심판','오심','판정','스프링캠프','개막'],
    },
    10: {
        '홈런'       : ['홈런','만루홈런','솔로홈런','투런','쓰리런'],
        '타격'       : ['안타','타율','타점','출루율','배팅','클린업'],
        '선발투수'   : ['선발','이닝','방어율','완봉','완투','퀄리티스타트'],
        '구원투수'   : ['불펜','마무리','세이브','홀드','중계'],
        '삼진·구위'  : ['삼진','탈삼진','구위','구속','직구','변화구'],
        '트레이드'   : ['트레이드','이적','교환','영입','방출'],
        'FA·계약'    : ['FA','자유계약','다년계약','연봉','잔류','해외진출'],
        '부상'       : ['부상','수술','재활','결장','통증','접질'],
        '감독·운영'  : ['감독','코치','구단','단장','프런트','스프링캠프'],
        '심판·기타'  : ['심판','오심','판정','비디오판독','개막'],
    },
    11: {
        '홈런'       : ['홈런','만루홈런','솔로홈런'],
        '타율·출루'  : ['타율','출루율','안타','득점'],
        '타점·장타'  : ['타점','장타율','장타','배팅'],
        '선발투수'   : ['선발','이닝','방어율','완봉'],
        '구원투수'   : ['불펜','마무리','세이브','홀드'],
        '삼진·구위'  : ['삼진','탈삼진','구위','구속'],
        '트레이드'   : ['트레이드','이적','교환','영입'],
        'FA·계약'    : ['FA','자유계약','연봉','다년계약'],
        '부상·재활'  : ['부상','수술','재활','결장'],
        '감독·코치'  : ['감독','코치','벤치','작전'],
        '구단·기타'  : ['구단','단장','심판','오심','스프링캠프'],
    },
    12: {
        '홈런'       : ['홈런','만루홈런','솔로홈런'],
        '타율'       : ['타율','안타','출루율','득점'],
        '타점'       : ['타점','장타율','배팅','클린업'],
        '선발투수'   : ['선발','이닝','방어율','완봉'],
        '마무리'     : ['마무리','세이브','홀드','중계'],
        '불펜'       : ['불펜','구원','롱릴리프'],
        '삼진'       : ['삼진','탈삼진','구위','구속'],
        '트레이드'   : ['트레이드','이적','교환','영입'],
        'FA'         : ['FA','자유계약','연봉','다년계약'],
        '부상'       : ['부상','수술','재활','결장'],
        '감독·운영'  : ['감독','코치','구단','단장'],
        '심판·기타'  : ['심판','오심','판정','스프링캠프'],
    },
    13: {
        '홈런'       : ['홈런','만루홈런'],
        '안타'       : ['안타','출루율','득점'],
        '타율·타점'  : ['타율','타점','배팅'],
        '장타'       : ['장타율','장타','클린업'],
        '선발투수'   : ['선발','이닝','방어율'],
        '마무리·세이브': ['마무리','세이브','홀드'],
        '불펜'       : ['불펜','중계','구원'],
        '삼진·구위'  : ['삼진','구위','구속'],
        '트레이드'   : ['트레이드','이적','영입'],
        'FA·계약'    : ['FA','자유계약','연봉'],
        '부상'       : ['부상','수술','재활'],
        '감독·운영'  : ['감독','코치','구단'],
        '심판·기타'  : ['심판','오심','스프링캠프'],
    },
    14: {
        '홈런'       : ['홈런','만루홈런'],
        '안타'       : ['안타','출루율'],
        '타율'       : ['타율','득점'],
        '타점·장타'  : ['타점','장타율','배팅'],
        '선발투수'   : ['선발','이닝','방어율'],
        '마무리'     : ['마무리','세이브'],
        '불펜·홀드'  : ['불펜','홀드','중계'],
        '삼진'       : ['삼진','탈삼진','구위'],
        '트레이드'   : ['트레이드','이적'],
        'FA'         : ['FA','자유계약','연봉'],
        '방출·영입'  : ['방출','영입','계약'],
        '부상'       : ['부상','수술','재활'],
        '감독·운영'  : ['감독','코치','구단','단장'],
        '심판·기타'  : ['심판','오심','스프링캠프'],
    },
    15: {
        '홈런'       : ['홈런','만루홈런'],
        '안타'       : ['안타'],
        '타율'       : ['타율','출루율'],
        '타점'       : ['타점','득점'],
        '장타'       : ['장타율','배팅','클린업'],
        '선발투수'   : ['선발','이닝','방어율'],
        '마무리'     : ['마무리','세이브'],
        '불펜·홀드'  : ['불펜','홀드','중계'],
        '삼진·구위'  : ['삼진','구위','구속'],
        '트레이드'   : ['트레이드','이적'],
        'FA·계약'    : ['FA','자유계약','연봉'],
        '방출·영입'  : ['방출','영입'],
        '부상'       : ['부상','수술','재활'],
        '감독·운영'  : ['감독','코치','구단'],
        '심판·기타'  : ['심판','오심','스프링캠프'],
    },
}


# ════════════════════════════════════════════════════════════════
# 방법 1: 키워드 자동 분할
# ════════════════════════════════════════════════════════════════

ALL_KEYWORDS = {
    '홈런'    : ['홈런','만루홈런','솔로홈런','투런','쓰리런'],
    '안타'    : ['안타','출루율','득점'],
    '타율'    : ['타율','타점','배팅','클린업','장타율','장타'],
    '선발투수': ['선발','이닝','방어율','완봉','완투','퀄리티스타트'],
    '마무리'  : ['마무리','세이브','홀드'],
    '불펜'    : ['불펜','중계','구원'],
    '삼진'    : ['삼진','탈삼진','구위','구속','직구','변화구'],
    '트레이드': ['트레이드','이적','교환'],
    '영입·방출': ['영입','방출'],
    'FA'      : ['FA','자유계약','연봉','다년계약','잔류','해외진출'],
    '부상'    : ['부상','수술','재활','결장','통증','접질'],
    '감독'    : ['감독','코치','벤치','작전'],
    '구단운영': ['구단','단장','프런트','스프링캠프','개막'],
    '심판'    : ['심판','오심','판정','비디오판독'],
    '기타'    : ['시범경기','올스타','드래프트','외국인선수'],
}

def build_auto_categories(n: int) -> dict:
    keys  = list(ALL_KEYWORDS.keys())
    total = len(keys)  # 15개

    if n >= total:
        return {k: ALL_KEYWORDS[k] for k in keys[:n]}

    merged = {}
    chunk  = total / n
    for i in range(n):
        start      = int(i * chunk)
        end        = int((i + 1) * chunk)
        group_keys = keys[start:end]
        cat_name   = '·'.join(group_keys)
        cat_kws    = []
        for k in group_keys:
            cat_kws.extend(ALL_KEYWORDS[k])
        merged[cat_name] = cat_kws
    return merged


# ════════════════════════════════════════════════════════════════
# 라벨링 함수
# ════════════════════════════════════════════════════════════════

def label_category(text: str, kw_dict: dict) -> str:
    scores = {cat: sum(text.count(kw) for kw in kws) for cat, kws in kw_dict.items()}
    best   = max(scores, key=scores.get)
    return best if scores[best] > 0 else list(kw_dict.keys())[-1]


def assign_labels(df: pd.DataFrame, kw_dict: dict, col: str = 'cat_tmp') -> pd.DataFrame:
    df = df.copy()
    df[col] = df['text'].fillna('').apply(lambda t: label_category(t, kw_dict))
    return df


# ════════════════════════════════════════════════════════════════
# TF-IDF 학습 및 평가
# ════════════════════════════════════════════════════════════════

def build_pipeline() -> Pipeline:
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True,
            min_df=2,
            analyzer='char_wb',
        )),
        ('clf', LinearSVC(C=1.0, max_iter=2000, random_state=42)),
    ])


def run_experiment(n_cat: int, kw_dict: dict,
                   train_df: pd.DataFrame,
                   val_df: pd.DataFrame,
                   test_df: pd.DataFrame) -> dict:
    col = 'cat_tmp'

    train_all     = pd.concat([train_df, val_df], ignore_index=True)
    train_labeled = assign_labels(train_all, kw_dict, col)
    test_labeled  = assign_labels(test_df,   kw_dict, col)

    X_train = train_labeled['text'].fillna('').tolist()
    y_train = train_labeled[col].tolist()
    X_test  = test_labeled['text'].fillna('').tolist()
    y_test  = test_labeled[col].tolist()

    actual_labels = sorted(set(y_test))

    pipe = build_pipeline()
    t0   = time.time()
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    elapsed = time.time() - t0

    report = classification_report(
        y_test, y_pred,
        target_names=actual_labels,
        output_dict=True, zero_division=0,
    )
    acc = accuracy_score(y_test, y_pred)
    f1  = report['macro avg']['f1-score']

    print(f"  n={n_cat:2d}개 | Acc={acc:.4f} | Macro-F1={f1:.4f} | {elapsed:.1f}초")
    return {
        'n_categories': n_cat,
        'accuracy'    : round(acc, 4),
        'macro_f1'    : round(f1, 4),
        'elapsed_sec' : round(elapsed, 2),
        'categories'  : list(kw_dict.keys()),
    }


# ════════════════════════════════════════════════════════════════
# 결과 저장 (scores.json에 추가)
# ════════════════════════════════════════════════════════════════

def _save_result(result: dict):
    path = os.path.join(RESULT_DIR, 'scores.json')
    data = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    data.append(result)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  scores.json 저장: {path}")


# ════════════════════════════════════════════════════════════════
# 메인 실험 루프
# ════════════════════════════════════════════════════════════════

def run_ablation(min_cat: int = 3, max_cat: int = 15):
    print(f"\n{'='*55}")
    print(f"  카테고리 개수 Ablation 실험 ({min_cat}~{max_cat}개)")
    print(f"{'='*55}")

    train_df = pd.read_csv(os.path.join(PROC_DIR, 'train.csv'), encoding='utf-8-sig')
    val_df   = pd.read_csv(os.path.join(PROC_DIR, 'val.csv'),   encoding='utf-8-sig')
    test_df  = pd.read_csv(os.path.join(PROC_DIR, 'test.csv'),  encoding='utf-8-sig')
    print(f"  데이터 로드: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    results_auto   = []
    results_manual = []

    for n in range(min_cat, max_cat + 1):

        print(f"\n[방법1-자동] n={n}")
        kw_auto = build_auto_categories(n)
        r_auto  = run_experiment(n, kw_auto, train_df, val_df, test_df)
        r_auto['method'] = 'auto'
        results_auto.append(r_auto)

        if n in MANUAL_CATEGORY_MAP:
            print(f"[방법2-수동] n={n}")
            kw_manual = MANUAL_CATEGORY_MAP[n]
            r_manual  = run_experiment(n, kw_manual, train_df, val_df, test_df)
            r_manual['method'] = 'manual'
            results_manual.append(r_manual)
        else:
            print(f"[방법2-수동] n={n} → 수동 정의 없음, 스킵")

    # scores.json에 추가 저장
    _save_result({
        'model'  : 'Category_Ablation',
        'auto'   : results_auto,
        'manual' : results_manual,
    })

    # 그래프 출력
    plot_results(results_auto, results_manual, min_cat, max_cat)

    return all_results


# ════════════════════════════════════════════════════════════════
# 결과 그래프
# ════════════════════════════════════════════════════════════════

def plot_results(results_auto, results_manual, min_cat, max_cat):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ns_auto   = [r['n_categories'] for r in results_auto]
    f1_auto   = [r['macro_f1']     for r in results_auto]
    acc_auto  = [r['accuracy']     for r in results_auto]

    ns_manual  = [r['n_categories'] for r in results_manual]
    f1_manual  = [r['macro_f1']     for r in results_manual]
    acc_manual = [r['accuracy']     for r in results_manual]

    ax = axes[0]
    ax.plot(ns_auto,   f1_auto,   'o-',  color='steelblue', label='방법1 (자동)', linewidth=2)
    ax.plot(ns_manual, f1_manual, 's--', color='tomato',    label='방법2 (수동)', linewidth=2)
    ax.axvline(x=6, color='green',  linestyle=':', alpha=0.7, label='v2 (6개)')
    ax.axvline(x=8, color='orange', linestyle=':', alpha=0.7, label='v1 (8개)')
    ax.set_xlabel('카테고리 개수')
    ax.set_ylabel('Macro-F1')
    ax.set_title('카테고리 개수 vs Macro-F1')
    ax.set_xticks(range(min_cat, max_cat + 1))
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(ns_auto,   acc_auto,   'o-',  color='steelblue', label='방법1 (자동)', linewidth=2)
    ax2.plot(ns_manual, acc_manual, 's--', color='tomato',    label='방법2 (수동)', linewidth=2)
    ax2.axvline(x=6, color='green',  linestyle=':', alpha=0.7, label='v2 (6개)')
    ax2.axvline(x=8, color='orange', linestyle=':', alpha=0.7, label='v1 (8개)')
    ax2.set_xlabel('카테고리 개수')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('카테고리 개수 vs Accuracy')
    ax2.set_xticks(range(min_cat, max_cat + 1))
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, 'category_ablation_plot.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  그래프 저장: {save_path}")

    if f1_auto:
        best_auto = results_auto[f1_auto.index(max(f1_auto))]
        print(f"\n  [방법1-자동] 최고 성능: {best_auto['n_categories']}개 → Macro-F1={best_auto['macro_f1']}")
    if f1_manual:
        best_manual = results_manual[f1_manual.index(max(f1_manual))]
        print(f"  [방법2-수동] 최고 성능: {best_manual['n_categories']}개 → Macro-F1={best_manual['macro_f1']}")


# ════════════════════════════════════════════════════════════════
# 실행
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    results = run_ablation(min_cat=3, max_cat=15)

    print(f"\n{'='*55}")
    print(f"  카테고리 Ablation 완료")
    print(f"  결과: results/scores.json")
    print(f"  그래프: results/figures/category_ablation_plot.png")
    print(f"{'='*55}")