import os
import time
import json
import re
from typing import Dict, Any

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# --------------------------------------------------------------------
# (옵션) BERT 맥락 분류 모델 로딩
# --------------------------------------------------------------------
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline
except ImportError:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    hf_pipeline = None

# BERT 모델 이름 (예: 부상 관련/비관련, 혹은 감성 분류 등으로 fine-tuning 한 것)
# 사용하지 않을 거면 그대로 None 두면 됨.
BERT_MODEL_NAME = None  # 예: "your-org/your-korean-injury-context-model"
USE_BERT_CONTEXT = BERT_MODEL_NAME is not None and AutoTokenizer is not None

context_clf = None
if USE_BERT_CONTEXT:
    try:
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        ctx_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME)
        context_clf = hf_pipeline(
            "text-classification",
            model=ctx_model,
            tokenizer=tokenizer,
            truncation=True
        )
    except Exception as e:
        print(f"[WARN] BERT 맥락 모델 로딩 실패: {e}")
        print("       BERT 기반 맥락 필터는 비활성화하고, 규칙 기반만 사용합니다.")
        context_clf = None
        USE_BERT_CONTEXT = False


def bert_context_score(text: str) -> float:
    """
    BERT 기반 맥락 점수.
    - 여기서는 예시로 sentiment-like 스코어를 float로 반환한다고 가정.
    - 실제로는 사용자가 fine-tuning한 태스크(부상 관련 여부 등)에 따라 로직 수정 필요.
    """
    if not USE_BERT_CONTEXT or context_clf is None:
        return 0.0
    out = context_clf(text, max_length=128)[0]  # {'label': 'POSITIVE', 'score': 0.98} 등
    label = out["label"].lower()
    score = out["score"]
    # 예시: 부정일수록 +, 긍정일수록 - 값으로 매핑 (필요에 따라 수정)
    if "neg" in label:
        return +score
    elif "pos" in label:
        return -score
    else:
        return 0.0


# --------------------------------------------------------------------
# 디렉토리 설정
# --------------------------------------------------------------------
PROC_DIR = "data/processed"
RESULT_DIR = "results"
FIG_DIR = os.path.join(RESULT_DIR, "figures")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# --------------------------------------------------------------------
# 방법 2: 수동 정의 카테고리 딕셔너리 (3~15개)
# --------------------------------------------------------------------
MANUAL_CATEGORY_MAP = {
    3: {
        '타격·투구': ['홈런', '안타', '타율', '타점', '삼진', '방어율', '선발', '불펜', '마무리', '완봉', '세이브'],
        '선수·운영': ['트레이드', '이적', 'FA', '자유계약', '영입', '방출', '감독', '코치', '구단', '단장'],
        '부상·기타': ['부상', '수술', '재활', '결장', '심판', '오심', '판정'],
    },
    4: {
        '타격': ['홈런', '안타', '타율', '타점', '출루율', '장타율', '득점', '타선'],
        '투구': ['삼진', '방어율', '선발', '불펜', '마무리', '완봉', '세이브', '홀드', '이닝'],
        '선수이동': ['트레이드', '이적', 'FA', '자유계약', '영입', '방출', '계약', '연봉'],
        '운영·기타': ['감독', '코치', '구단', '부상', '수술', '재활', '심판'],
    },
    5: {
        '타격': ['홈런', '안타', '타율', '타점', '출루율', '장타율', '득점', '타선'],
        '투구': ['삼진', '방어율', '선발', '불펜', '마무리', '완봉', '세이브', '홀드', '이닝'],
        '선수이동': ['트레이드', '이적', 'FA', '자유계약', '영입', '방출', '계약', '연봉'],
        '감독·운영': ['감독', '코치', '구단', '단장', '프런트', '스프링캠프'],
        '부상·기타': ['부상', '수술', '재활', '결장', '심판', '오심'],
    },
    6: {  # 기존 v2
        '타격': ['홈런', '안타', '타율', '타점', '출루율', '장타율', '타격', '배팅', '득점', '타선'],
        '투구': ['삼진', '방어율', '선발', '불펜', '마무리', '투구', '완봉', '세이브', '홀드', '이닝'],
        '선수이동': ['트레이드', '이적', 'FA', '자유계약', '영입', '방출', '계약', '다년계약', '연봉'],
        '부상': ['부상', '수술', '재활', '결장', '부상우려', '통증'],
        '감독·운영': ['감독', '코치', '구단', '단장', '프런트', '스프링캠프'],
        '기타': ['심판', '오심', '판정', '비디오판독'],
    },
    7: {
        '홈런·장타': ['홈런', '장타', '장타율', '만루홈런', '솔로홈런'],
        '안타·타율': ['안타', '타율', '타점', '출루율', '득점', '타선', '배팅'],
        '선발투수': ['선발', '이닝', '방어율', '완봉', '완투', '퀄리티스타트'],
        '불펜·마무리': ['불펜', '마무리', '세이브', '홀드', '중계'],
        '선수이동': ['트레이드', '이적', 'FA', '자유계약', '영입', '방출', '계약', '연봉'],
        '부상·재활': ['부상', '수술', '재활', '결장', '통증'],
        '운영·기타': ['감독', '코치', '구단', '심판', '오심'],
    },
    8: {  # 기존 v1 기반
        '타격': ['홈런', '안타', '타율', '타점', '출루율', '득점'],
        '투구': ['삼진', '방어율', '선발', '완봉', '세이브', '이닝'],
        '트레이드': ['트레이드', '이적', '교환', '영입'],
        '부상': ['부상', '수술', '재활', '결장'],
        'FA': ['FA', '자유계약', '다년계약', '연봉', '잔류'],
        '감독': ['감독', '코치', '벤치', '단장', '구단'],
        '심판': ['심판', '오심', '판정', '비디오판독'],
        '기타': ['스프링캠프', '개막', '시범경기', '올스타'],
    },
    9: {
        '홈런': ['홈런', '만루홈런', '솔로홈런', '투런', '쓰리런'],
        '타격': ['안타', '타율', '타점', '출루율', '장타율', '배팅', '타선'],
        '선발투수': ['선발', '이닝', '방어율', '완봉', '완투'],
        '구원투수': ['불펜', '마무리', '세이브', '홀드', '중계', '삼진'],
        '트레이드': ['트레이드', '이적', '교환', '영입'],
        'FA·계약': ['FA', '자유계약', '다년계약', '연봉', '잔류'],
        '부상': ['부상', '수술', '재활', '결장', '통증'],
        '감독·운영': ['감독', '코치', '구단', '단장', '프런트'],
        '심판·기타': ['심판', '오심', '판정', '스프링캠프', '개막'],
    },
    10: {
        '홈런': ['홈런', '만루홈런', '솔로홈런', '투런', '쓰리런'],
        '타격': ['안타', '타율', '타점', '출루율', '배팅', '클린업'],
        '선발투수': ['선발', '이닝', '방어율', '완봉', '완투', '퀄리티스타트'],
        '구원투수': ['불펜', '마무리', '세이브', '홀드', '중계'],
        '삼진·구위': ['삼진', '탈삼진', '구위', '구속', '직구', '변화구'],
        '트레이드': ['트레이드', '이적', '교환', '영입', '방출'],
        'FA·계약': ['FA', '자유계약', '다년계약', '연봉', '잔류', '해외진출'],
        '부상': ['부상', '수술', '재활', '결장', '통증', '접질'],
        '감독·운영': ['감독', '코치', '구단', '단장', '프런트', '스프링캠프'],
        '심판·기타': ['심판', '오심', '판정', '비디오판독', '개막'],
    },
    11: {
        '홈런': ['홈런', '만루홈런', '솔로홈런'],
        '타율·출루': ['타율', '출루율', '안타', '득점'],
        '타점·장타': ['타점', '장타율', '장타', '배팅'],
        '선발투수': ['선발', '이닝', '방어율', '완봉'],
        '구원투수': ['불펜', '마무리', '세이브', '홀드'],
        '삼진·구위': ['삼진', '탈삼진', '구위', '구속'],
        '트레이드': ['트레이드', '이적', '교환', '영입'],
        'FA·계약': ['FA', '자유계약', '연봉', '다년계약'],
        '부상·재활': ['부상', '수술', '재활', '결장'],
        '감독·코치': ['감독', '코치', '벤치', '작전'],
        '구단·기타': ['구단', '단장', '심판', '오심', '스프링캠프'],
    },
    12: {
        '홈런': ['홈런', '만루홈런', '솔로홈런'],
        '타율': ['타율', '안타', '출루율', '득점'],
        '타점': ['타점', '장타율', '배팅', '클린업'],
        '선발투수': ['선발', '이닝', '방어율', '완봉'],
        '마무리': ['마무리', '세이브', '홀드', '중계'],
        '불펜': ['불펜', '구원', '롱릴리프'],
        '삼진': ['삼진', '탈삼진', '구위', '구속'],
        '트레이드': ['트레이드', '이적', '교환', '영입'],
        'FA': ['FA', '자유계약', '연봉', '다년계약'],
        '부상': ['부상', '수술', '재활', '결장'],
        '감독·운영': ['감독', '코치', '구단', '단장'],
        '심판·기타': ['심판', '오심', '판정', '스프링캠프'],
    },
    13: {
        '홈런': ['홈런', '만루홈런'],
        '안타': ['안타', '출루율', '득점'],
        '타율·타점': ['타율', '타점', '배팅'],
        '장타': ['장타율', '장타', '클린업'],
        '선발투수': ['선발', '이닝', '방어율'],
        '마무리·세이브': ['마무리', '세이브', '홀드'],
        '불펜': ['불펜', '중계', '구원'],
        '삼진·구위': ['삼진', '구위', '구속'],
        '트레이드': ['트레이드', '이적', '영입'],
        'FA·계약': ['FA', '자유계약', '연봉'],
        '부상': ['부상', '수술', '재활'],
        '감독·운영': ['감독', '코치', '구단'],
        '심판·기타': ['심판', '오심', '스프링캠프'],
    },
    14: {
        '홈런': ['홈런', '만루홈런'],
        '안타': ['안타', '출루율'],
        '타율': ['타율', '득점'],
        '타점·장타': ['타점', '장타율', '배팅'],
        '선발투수': ['선발', '이닝', '방어율'],
        '마무리': ['마무리', '세이브'],
        '불펜·홀드': ['불펜', '홀드', '중계'],
        '삼진': ['삼진', '탈삼진', '구위'],
        '트레이드': ['트레이드', '이적'],
        'FA': ['FA', '자유계약', '연봉'],
        '방출·영입': ['방출', '영입', '계약'],
        '부상': ['부상', '수술', '재활'],
        '감독·운영': ['감독', '코치', '구단', '단장'],
        '심판·기타': ['심판', '오심', '스프링캠프'],
    },
    15: {
        '홈런': ['홈런', '만루홈런'],
        '안타': ['안타'],
        '타율': ['타율', '출루율'],
        '타점': ['타점', '득점'],
        '장타': ['장타율', '배팅', '클린업'],
        '선발투수': ['선발', '이닝', '방어율'],
        '마무리': ['마무리', '세이브'],
        '불펜·홀드': ['불펜', '홀드', '중계'],
        '삼진·구위': ['삼진', '구위', '구속'],
        '트레이드': ['트레이드', '이적'],
        'FA·계약': ['FA', '자유계약', '연봉'],
        '방출·영입': ['방출', '영입'],
        '부상': ['부상', '수술', '재활'],
        '감독·운영': ['감독', '코치', '구단'],
        '심판·기타': ['심판', '오심', '스프링캠프'],
    },
}


# --------------------------------------------------------------------
# 키워드 자동 분할용 전체 키워드
# --------------------------------------------------------------------
ALL_KEYWORDS = {
    '홈런': ['홈런', '만루홈런', '솔로홈런', '투런', '쓰리런'],
    '안타': ['안타', '출루율', '득점'],
    '타율': ['타율', '타점', '배팅', '클린업', '장타율', '장타'],
    '선발투수': ['선발', '이닝', '방어율', '완봉', '완투', '퀄리티스타트'],
    '마무리': ['마무리', '세이브', '홀드'],
    '불펜': ['불펜', '중계', '구원'],
    '삼진': ['삼진', '탈삼진', '구위', '구속', '직구', '변화구'],
    '트레이드': ['트레이드', '이적', '교환'],
    '영입·방출': ['영입', '방출'],
    'FA': ['FA', '자유계약', '연봉', '다년계약', '잔류', '해외진출'],
    '부상': ['부상', '수술', '재활', '결장', '통증', '접질'],
    '감독': ['감독', '코치', '벤치', '작전'],
    '구단운영': ['구단', '단장', '프런트', '스프링캠프', '개막'],
    '심판': ['심판', '오심', '판정', '비디오판독'],
    '기타': ['시범경기', '올스타', '드래프트', '외국인선수'],
}


def build_auto_categories(n: int) -> dict:
    keys = list(ALL_KEYWORDS.keys())
    total = len(keys)  # 15개

    if n >= total:
        return {k: ALL_KEYWORDS[k] for k in keys[:n]}

    merged: Dict[str, Any] = {}
    chunk = total / n
    for i in range(n):
        start = int(i * chunk)
        end = int((i + 1) * chunk)
        group_keys = keys[start:end]
        cat_name = '·'.join(group_keys)
        cat_kws = []
        for k in group_keys:
            cat_kws.extend(ALL_KEYWORDS[k])
        merged[cat_name] = cat_kws
    return merged


# --------------------------------------------------------------------
# 맥락(단어 + 서술어) 민감 키워드 설정
# --------------------------------------------------------------------
CONTEXT_SENSITIVE_KW: Dict[str, Dict[str, Any]] = {
    # '회복'은 그냥 나오면 애매하지만, 부상/수술/재활/통증과 같이 나오면
    # 부상 관련 이슈라고 보고 싶을 때 사용
    '회복': {
        'window': 25,  # 앞뒤로 볼 문자 수
        'require_any_of': ['부상', '수술', '재활', '통증'],
        'positive_clues': ['순조롭', '順調', '잘 되고', '順조롭', '원활'],
        'negative_clues': ['지연', '더디', '늦어지', '난항', '쉽지 않', '악화'],
        'use_bert': True,
    },
    # '부상' 자체도 맥락을 보고 싶다면 여기에 규칙 추가
    '부상': {
        'window': 20,
        'require_any_of': [],  # '부상' 자체로도 충분하다고 보고 비워둠
        'positive_clues': ['회복', '복귀', '완치', '극복'],
        'negative_clues': ['악화', '재발', '이탈', '장기 결장', '시즌 아웃'],
        'use_bert': True,
    },
    # 필요시 '재활', '통증' 등도 여기에 추가
}


def _window_around(text: str, start: int, end: int, window: int) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:right]


def keyword_is_active(text: str, base_keyword: str, cfg: Dict[str, Any]) -> bool:
    """
    특정 키워드(예: '회복')가 이번 문맥에서
    실제로 카테고리 점수에 반영할 만한 상황인지 판단.
    - require_any_of: 이 중 아무거나 주변에 있으면 '관련 있음'으로 간주
    - positive/negative_clues: 서술어 기반 단서
    - BERT: 규칙으로 애매한 경우 보조 판정
    """
    window = cfg.get('window', 20)
    req_any = cfg.get('require_any_of', [])
    pos_clues = cfg.get('positive_clues', [])
    neg_clues = cfg.get('negative_clues', [])
    use_bert = cfg.get('use_bert', False)

    found_any = False

    for m in re.finditer(re.escape(base_keyword), text):
        snippet = _window_around(text, m.start(), m.end(), window)

        # 1) require_any_of 단서
        if req_any:
            if any(tok in snippet for tok in req_any):
                found_any = True
            else:
                # 이 위치는 pass, 다른 위치에 대해 계속 탐색
                continue

        # 2) positive/negative 단서
        pos_hits = sum(snippet.count(c) for c in pos_clues)
        neg_hits = sum(snippet.count(c) for c in neg_clues)

        if pos_hits > 0 or neg_hits > 0:
            return True  # 문맥상 확실히 관련

        # 3) BERT 기반 추가 판정 (옵션)
        if use_bert:
            ctx_score = bert_context_score(snippet)
            if abs(ctx_score) > 0.4:  # 임계값은 튜닝 포인트
                return True

    # require_any_of가 있었고 한 번이라도 만족했다면 True,
    # 아무런 단서도 못 찾았으면 False
    return found_any


# --------------------------------------------------------------------
# 라벨링 함수 (맥락-aware 버전)
# --------------------------------------------------------------------
def label_category_with_context(text: str, kw_dict: dict) -> str:
    """
    TF-IDF용 카테고리 라벨링 함수 (char n-gram + LinearSVC 학습용).
    - 대부분 키워드는 기존처럼 단순 count
    - CONTEXT_SENSITIVE_KW에 등록된 키워드는 keyword_is_active(...)가
      True인 경우에만 count
    """
    text = text or ""
    scores: Dict[str, int] = {cat: 0 for cat in kw_dict.keys()}

    for cat, kws in kw_dict.items():
        for kw in kws:
            if kw in CONTEXT_SENSITIVE_KW:
                cfg = CONTEXT_SENSITIVE_KW[kw]
                if keyword_is_active(text, kw, cfg):
                    scores[cat] += text.count(kw)
            else:
                scores[cat] += text.count(kw)

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else list(kw_dict.keys())[-1]


def assign_labels(df: pd.DataFrame, kw_dict: dict, col: str = "cat_tmp") -> pd.DataFrame:
    df = df.copy()
    df[col] = df["text"].fillna("").apply(lambda t: label_category_with_context(t, kw_dict))
    return df


# --------------------------------------------------------------------
# TF-IDF + LinearSVC 파이프라인
# --------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True,
            min_df=2,
            analyzer="char_wb",
        )),
        ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=42)),
    ])


def run_experiment(n_cat: int,
                   kw_dict: dict,
                   train_df: pd.DataFrame,
                   val_df: pd.DataFrame,
                   test_df: pd.DataFrame) -> dict:
    col = "cat_tmp"

    train_all = pd.concat([train_df, val_df], ignore_index=True)
    train_labeled = assign_labels(train_all, kw_dict, col)
    test_labeled = assign_labels(test_df, kw_dict, col)

    X_train = train_labeled["text"].fillna("").tolist()
    y_train = train_labeled[col].tolist()
    X_test = test_labeled["text"].fillna("").tolist()
    y_test = test_labeled[col].tolist()

    actual_labels = sorted(set(y_test))

    pipe = build_pipeline()
    t0 = time.time()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    elapsed = time.time() - t0

    report = classification_report(
        y_test,
        y_pred,
        target_names=actual_labels,
        output_dict=True,
        zero_division=0,
    )
    acc = accuracy_score(y_test, y_pred)
    f1 = report["macro avg"]["f1-score"]

    print(f"  n={n_cat:2d}개 | Acc={acc:.4f} | Macro-F1={f1:.4f} | {elapsed:.1f}초")
    return {
        "n_categories": n_cat,
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "elapsed_sec": round(elapsed, 2),
        "categories": list(kw_dict.keys()),
    }


# --------------------------------------------------------------------
# 결과 저장 함수 (scores.json에 append)
# --------------------------------------------------------------------
def _save_result(result: dict, path: str = None) -> None:
    if path is None:
        path = os.path.join(RESULT_DIR, "scores.json")

    data = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                data = existing
            else:
                data = [existing]
        except Exception:
            data = []
    data.append(result)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------
# Ablation 실험 루프
# --------------------------------------------------------------------
def run_ablation(min_cat: int = 3, max_cat: int = 15):
    print(f"\n{'=' * 55}")
    print(f"  카테고리 개수 Ablation 실험 ({min_cat}~{max_cat}개)")
    print(f"{'=' * 55}")

    train_df = pd.read_csv(os.path.join(PROC_DIR, "train.csv"), encoding="utf-8-sig")
    val_df = pd.read_csv(os.path.join(PROC_DIR, "val.csv"), encoding="utf-8-sig")
    test_df = pd.read_csv(os.path.join(PROC_DIR, "test.csv"), encoding="utf-8-sig")
    print(f"  데이터 로드: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    results_auto = []
    results_manual = []

    for n in range(min_cat, max_cat + 1):
        print(f"\n[방법1-자동] n={n}")
        kw_auto = build_auto_categories(n)
        r_auto = run_experiment(n, kw_auto, train_df, val_df, test_df)
        r_auto["method"] = "auto"
        results_auto.append(r_auto)

        if n in MANUAL_CATEGORY_MAP:
            print(f"[방법2-수동] n={n}")
            kw_manual = MANUAL_CATEGORY_MAP[n]
            r_manual = run_experiment(n, kw_manual, train_df, val_df, test_df)
            r_manual["method"] = "manual"
            results_manual.append(r_manual)
        else:
            print(f"[방법2-수동] n={n} → 수동 정의 없음, 스킵")

    _save_result({
        "model": "Category_Ablation_ContextAware",
        "auto": results_auto,
        "manual": results_manual,
    })


if __name__ == "__main__":
    run_ablation()