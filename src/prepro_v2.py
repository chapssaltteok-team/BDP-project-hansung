"""
prepro_v2.py
============
KBO 뉴스 본문 정제 (Layer 1 공통 + Layer 2 매체별 분기)

목표
────
- Layer 1: 모든 매체 공통 노이즈 제거 (사진 출처, 캡션, 매체헤더, 이메일 등)
- Layer 2: 상위 5개 신문사 + 4개 방송사 특이 패턴 처리
- 검수 단계 자동화 (길이 감소율, 샘플 비교)

실행: python src/prepro_v2.py
"""

import os
import re
import json
import pandas as pd
from collections import Counter

RAW_DIR = 'data/원시데이터'
OUT_DIR = 'data/processed'
LOG_DIR = 'results'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

KBO_TEAMS = [
    '삼성 라이온즈', 'LG 트윈스', '두산 베어스', 'KIA 타이거즈',
    '롯데 자이언츠', 'SSG 랜더스', '한화 이글스', '키움 히어로즈',
    'KT 위즈', 'NC 다이노스',
    '삼성', 'LG', '두산', 'KIA', '롯데', 'SSG', '한화', '키움', 'KT', 'NC'
]
TEAMS_PAT = '|'.join(re.escape(t) for t in KBO_TEAMS)


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — 매체 무관 공통 정제
# ═════════════════════════════════════════════════════════════════════

def clean_common(text: str) -> str:
    """모든 매체에 공통 적용되는 노이즈 제거"""
    if not isinstance(text, str):
        return text

    # ── 1. 긴 캡션: "날짜 ... 있다./매체" 형태 ──────────────────
    # 예: "2025년 8월 9일 오후 서울 잠실구장... 이야기를 하고 있다./마이데일리"
    # 핵심: 행동 키워드는 사진 캡션 특유 표현만 (본문 오삭제 방지)
    # "훈련", "경기", "장면", "모습" 등은 본문에도 흔히 나와서 제외
    text = re.sub(
        r'(?:\d{4}년\s*)?(?:\d{1,2}월\s*)?\d{1,2}일'
        r'(?:\s+(?:오전|오후|새벽|밤))?'
        r'\s+[가-힣A-Za-z0-9\'\"\s.,\-(){}『』]{1,150}?'    # 150자로 더 제한
        r'(?:투구하고|타격하고|수비하고|주루하고|던지고|치고\s있다'
        r'|기뻐하고|환호하고|세리머니|이야기를?\s*(?:나누|하고)'
        r'|기념촬영|입국하고|출국하고|소감을?\s*전하|인사를?\s*하고'
        r'|이끌고\s*있다)'
        r'[가-힣A-Za-z0-9\s.,\-]{0,60}?'
        r'(?:있다|있었다|모습)\s*\.?\s*'                    # 종결 동사 필수
        r'/\s*[가-힣A-Za-z]+',
        '',
        text,
    )

    # ── 2. 사진 캡션 출처 표기 ──────────────────────────────────
    text = re.sub(r'사진\s*[=:]\s*[가-힣A-Za-z0-9\s]{1,30}(?=\s|$|\.|\n|,)', '', text)
    text = re.sub(r'사진\s*제공\s*[=:]?\s*[가-힣A-Za-z]+', '', text)
    text = re.sub(r'\[\s*사진\s*[^\]]{0,50}\]', '', text)

    # ── 3. ⓒ 저작권 표기 ─────────────────────────────────────────
    text = re.sub(r'ⓒ\s*[가-힣A-Za-z]+(?:\s+[가-힣]+)?', '', text)

    # ── 4. "[매체] 제공" / "[매체]제공" ──────────────────────────
    text = re.sub(
        r'(?<![가-힣])'
        r'(?:연합뉴스|뉴스1|마이데일리|OSEN|스포츠조선|스포츠서울'
        r'|뉴시스|스포티비뉴스|엑스포츠뉴스|마니아타임즈|일간스포츠'
        r'|MK스포츠|스타뉴스|티케이오시비)'
        r'\s*제공',
        '',
        text,
    )

    # ── 5. 자료사진 표기 ─────────────────────────────────────────
    text = re.sub(r'자료\s*사진', '', text)
    text = re.sub(r'게티이미지(?:코리아|뱅크)?', '', text)

    # ── 6. 일반 매체헤더 [매체=기자명] [매체 = 기자명] [매체 | 기자명]
    text = re.sub(
        r'\[\s*[가-힣A-Za-z0-9]+(?:뉴스|일보|미디어|타임즈|스포츠|TV|방송)?'
        r'\s*[=|]\s*[^\]]{0,40}기자\s*\]',
        '',
        text,
    )

    # ── 7. 매체헤더 (등호 없는 형태) [매체 기자명 기자] ─────────
    text = re.sub(r'\[\s*[가-힣]+(?:뉴스|타임즈|미디어)\s+[가-힣]{2,4}\s+기자\s*\]', '', text)

    # ── 8. [기자명 매체명 기자] (역순) ─────────────────────────
    text = re.sub(r'\[\s*[가-힣]{2,4}\s+[가-힣]+(?:뉴스|타임즈|미디어)\s+기자\s*\]', '', text)

    # ── 9. (매체명 기자명 기자) 소괄호 — 엑스포츠뉴스 등 ──────
    text = re.sub(
        r'\([가-힣A-Z]+(?:\s+[가-힣]+,)?\s+[가-힣]{2,4}\s+(?:인턴)?기자\)',
        '',
        text,
    )

    # ── 10. 이메일 시그니처 /xxx@xxx.com ────────────────────────
    text = re.sub(r'/\s*[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,4}', '', text)
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,4}', '', text)

    # ── 11. 짧은 캡션: "팀명 선수명./팀명" 또는 "팀명 선수명/팀명" ──
    # 예: "삼성 라이온즈 이호성./삼성 라이온즈", "한화 라이언 와이스./한화 이글스"
    text = re.sub(
        r'(?:' + TEAMS_PAT + r')\s+'
        r'[가-힣]{2,4}(?:\s+[가-힣]{2,4})?'   # 선수명 (한국명 2-4자 또는 외국명 두 단어)
        r'\.?\s*/\s*'
        r'(?:' + TEAMS_PAT + r')',
        '',
        text,
    )

    # ── 12. 짧은 캡션: "선수명/팀명" (팀명 접두 없음) ───────────
    text = re.sub(
        r'(?<![가-힣A-Za-z])[가-힣]{2,4}\s*/\s*(?:' + TEAMS_PAT + r')(?![가-힣A-Za-z])',
        '',
        text,
    )

    # ── 13. "직책/이름. / [본문 시작]" 형태 캡션 ─────────────
    # 예: "SSG 랜더스 이숭용 감독. / 프로야구 SSG..."
    # 본문 시작부에 자주 등장. /로 본문이 시작되는 매우 특이 패턴
    text = re.sub(
        r'^(?:' + TEAMS_PAT + r')?\s*[가-힣]{2,4}\s+'
        r'(?:감독|코치|선수|단장|회장|위원|투수|타자|포수|내야수|외야수)\.\s*/\s*',
        '',
        text,
    )

    # ── 14. 끊긴 기자 시그니처: "=이름 (대)기자" ──────────────
    # 예: "경기가 =김진경 대기자 한화 이글스가..."
    # 엄격 규칙: 앞에 공백이 있어야 함 (자연스러운 문맥 보호)
    text = re.sub(
        r'(?<=\s)=\s*[가-힣]{2,4}\s+(?:대|선임|수석)?기자(?=\s|$)',
        '',
        text,
    )

    # ── 15. ▲로 시작하는 캡션 (스포티비뉴스 특이 패턴) ────────
    # 예: "▲ 비슬리 ⓒ곽혜미 기자", "▲ 올 시즌... 리베라토 ⓒ곽혜미 기자"
    # 안전 규칙: ▲ 시작 + 짧은 텍스트(150자 이내) + "기자" 또는 "ⓒ"로 끝나는 경우만
    text = re.sub(
        r'▲\s+[^▲\n]{1,150}?\s*(?:ⓒ\s*[가-힣A-Za-z]+\s*기자|[가-힣]{2,4}\s+기자)',
        '',
        text,
    )
    # ▲ + 짧은 문장 (제목성 캡션, 본문이 아닌 명사구로 끝나는 경우)
    # 너무 위험하므로 추가 패턴 없음. ▲ 한 글자만 단독으로 남으면 제거
    text = re.sub(r'(?<!\w)▲(?!\w)', '', text)

    # ── 16. [지역=매체] 헤더 (뉴시스 특이 패턴) ────────────────
    # 예: "[서울=뉴시스]", "[부산=뉴시스]"
    text = re.sub(r'\[\s*[가-힣]+\s*=\s*[가-힣]+(?:뉴스|일보|미디어)?\s*\]', '', text)

    # ── 17. 뉴시스식 캡션: 이름 기자 = 날짜+장소+행동+날짜 ────
    # 예: "최동준 기자 = 23일 서울 구로구... 기뻐하고 있다. 2025.05.23."
    # 엄격 규칙: 반드시 끝에 "있다.+날짜" 또는 "(사진=...) 날짜"가 있어야 함
    # 본문 오삭제 방지를 위해 종결자가 명확한 경우만 매칭
    text = re.sub(
        r'[가-힣]{2,4}\s+기자\s*=\s*'
        r'\d{1,2}일\s+[^=]{10,200}?'           # 날짜 시작 + 본문 (= 만나면 멈춤)
        r'(?:있다|모습|있었다|기뻐하고|치고|던지고)\s*\.\s*'  # 종결 동사 필수
        r'\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.?',  # 종결 날짜 필수
        '',
        text,
    )

    # ── 18. 뉴시스 메타 표기 ───────────────────────────────────
    # "*재판매 및 DB 금지", "2025.09.15." 같은 날짜 단독
    text = re.sub(r'\*\s*재판매\s*및?\s*DB\s*금지', '', text)
    text = re.sub(r'(?<![가-힣\d])\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.?(?!\d)', '', text)

    # ── 19. 캡션 종결자: "사진=...기자 / DB / 팀명" 복합 ─────
    # 예: "사진=고척, 김한준 기자 / 엑스포츠뉴스 DB / 롯데 자이언츠"
    # 예: "사진=KIA 타이거즈, 한화 이글스"
    text = re.sub(
        r'사진\s*[=:]\s*[가-힣A-Za-z\s,()]+(?:기자|DB|제공|이글스|라이온즈|트윈스|베어스|타이거즈|자이언츠|랜더스|히어로즈|위즈|다이노스)'
        r'(?:\s*/\s*[가-힣A-Za-z\s]+)*',
        '',
        text,
    )
    # "사진[연합뉴스]" 같은 변형
    text = re.sub(r'사진\s*\[\s*[가-힣A-Za-z]+\s*\]', '', text)

    # ── 19-b. 본문 시작 "이름 / [잔재]" 패턴 (마니아타임즈) ──
    # 예: "안현민 / WBC 출격을..." (앞에 "사진=연합뉴스"가 제거된 잔재)
    text = re.sub(
        r'^[가-힣]{2,4}\s*/\s*(?=[가-힣A-Z])',
        '',
        text,
    )

    # ── 20. 본문 시작 캡션: "...있다. 사진 ○○ 기자" ──────────
    # 예: "박병호...인사하고 있다. 사진 김한준 기자"
    # "있다." 종결은 보존하고 "사진 ○○ 기자" 부분만 제거
    text = re.sub(
        r'((?:있다|모습|있었다)\.)\s*사진\s+[가-힣]{2,4}\s+기자',
        r'\1',
        text,
    )

    # ── 21. 잔재 청소: ", , 디자인" "기자 / /" 등 ─────────────
    text = re.sub(r',\s*,(?:\s*,)*', ',', text)             # 연속 콤마
    text = re.sub(r'/\s*/+', '', text)                       # 연속 슬래시
    text = re.sub(r'^\s*[,/.\s]+', '', text)                 # 시작부 잔재
    text = re.sub(r'(?<=\s)[,/]\s+(?=[가-힣])', '', text)   # 단어 사이 단독 콤마/슬래시
    # 빈 괄호 변형 청소: ( 제공), ( ), (,) 등
    text = re.sub(r'\(\s*제공\s*\)', '', text)
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)

    return text


# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — 상위 5개 신문사 분기 처리
# ═════════════════════════════════════════════════════════════════════

def clean_mydaily(text: str) -> str:
    """마이데일리 (1,014건) — 긴 캡션 "/마이데일리" 종결"""
    # 본문 끝 또는 중간에 "/마이데일리"로 끝나는 캡션
    text = re.sub(
        r'[가-힣\s\d\'\"\.]*?'
        r'(?:있다|모습|장면|중|뒤|기념촬영)\.?\s*'
        r'/\s*마이데일리',
        '',
        text,
    )
    # 단순 "/마이데일리" 잔재
    text = re.sub(r'/\s*마이데일리(?:\s*DB)?', '', text)
    return text


def clean_osen(text: str) -> str:
    """OSEN (842건) — 본문 종결 /xxx@osen.co.kr 패턴"""
    # OSEN 이메일 종결자 (이미 공통 처리되지만 안전망)
    text = re.sub(r'/\s*[a-zA-Z0-9._]+@osen\.co\.kr', '', text)
    # OSEN DB 표기
    text = re.sub(r'OSEN\s*DB', '', text)
    # [OSEN=지역, 기자명 기자] 추가 변형
    text = re.sub(r'\[\s*OSEN\s*=\s*[^\]]{0,40}기자\s*\]', '', text)
    return text


def clean_xports(text: str) -> str:
    """엑스포츠뉴스 (700건) — 소괄호 본문 시작 + DB 종결"""
    # (엑스포츠뉴스 ... 기자) — 공통 처리에 포함되지만 안전망
    text = re.sub(
        r'\(엑스포츠뉴스(?:\s+[가-힣]+,)?\s+[가-힣]{2,4}\s+기자\)',
        '',
        text,
    )
    # 캡션 종결자 "엑스포츠뉴스 DB"
    text = re.sub(r'엑스포츠뉴스\s*DB', '', text)
    return text


def clean_sportschosun(text: str) -> str:
    """스포츠조선 (677건) — 캡션 + 이메일 + 날짜 복합"""
    # 전체 캡션 종결자: "지역=이름 기자 이메일/날짜/"
    text = re.sub(
        r'[가-힣]{2,5}\s*=\s*[가-힣]{2,4}\s+기자\s*'
        r'(?:[a-zA-Z0-9._]+@sportschosun\.com)?\s*'
        r'/?\s*\d{4}\.\d+\.\d+\.?/?',
        '',
        text,
    )
    # "지역=이름 기자" 캡션 단독 (이메일·날짜 없는 경우)
    text = re.sub(r'(?<![가-힣A-Za-z])[가-힣]{2,5}\s*=\s*[가-힣]{2,4}\s+기자', '', text)
    # 단순 "/2025.8.30/" 같은 날짜 잔재
    text = re.sub(r'/\s*\d{4}\.\d+\.\d+\.?/?', '', text)
    return text


def clean_mania(text: str) -> str:
    """마니아타임즈 (634건) — 짧고 명시적 '사진=연합뉴스'"""
    text = re.sub(r'\[[가-힣]+\s+마니아타임즈\s+(?:인턴)?기자\]', '', text)
    # 추가 표기: "사진=마니아타임즈"
    text = re.sub(r'사진\s*=\s*마니아타임즈', '', text)
    return text


# ═════════════════════════════════════════════════════════════════════
# LAYER 3 — 방송사 분기 처리
# ═════════════════════════════════════════════════════════════════════

def clean_kbs(text: str) -> str:
    """KBS (496건) — 가장 긴 노이즈: 제보 안내 + 영상편집"""
    # ① 제보 안내 (가장 긴 노이즈, 200자 가량)
    text = re.sub(
        r'■\s*제보하기.*?(?:KBS뉴스를?\s*구독해주세요!|$)',
        '',
        text,
        flags=re.DOTALL,
    )
    # ② "KBS 뉴스 ○○입니다. 촬영기자:.../영상편집:..."
    # 종료 조건을 넓게: 공백/끝/문장경계까지
    text = re.sub(
        r'KBS\s*뉴스\s+[가-힣]{2,4}입니다\.\s*'
        r'촬영기자\s*[:：][^\n■]+?'
        r'(?:/?영상편집\s*[:：][^\n■]*?)?'
        r'(?=\s*■|\s*$|\n)',
        '',
        text,
    )
    # ③ [KBS 지역명] 헤더
    text = re.sub(r'\[KBS\s+[가-힣]+\]', '', text)
    # ④ KBS 뉴스 기자명 (단독)
    text = re.sub(r'KBS\s*뉴스\s+[가-힣]{2,4}입니다\.?', '', text)
    return text


def clean_sbs(text: str) -> str:
    """SBS (322건) — 괄호형 제작진"""
    # 괄호 안 어디든 영상취재/영상편집/디자인이 있으면 괄호 전체 제거
    # (콤마 앞 항목이 공통 정규식에서 먼저 지워져 빈 콤마만 남는 케이스 방지)
    text = re.sub(
        r'\([^)]*?(?:영상취재|영상편집|디자인|촬영기자)[^)]*\)',
        '',
        text,
    )
    # <앵커> <기자> 마커
    text = re.sub(r'<\s*(?:앵커|기자|리포트)\s*>', '', text)
    return text


def clean_ytn(text: str) -> str:
    """YTN (298건) — 자체 제보 안내 + 영상편집"""
    # ① 제보 안내 (※로 시작)
    text = re.sub(
        r"※\s*['\"]?당신의 제보가 뉴스가 됩니다['\"]?.*?(?=$|\n\n)",
        '',
        text,
        flags=re.DOTALL,
    )
    # ② "YTN 기자명입니다. 영상편집:..."
    text = re.sub(
        r'YTN\s+[가-힣]{2,4}입니다\.?\s*'
        r'(?:영상편집\s*[:：][^\n.]+)?',
        '',
        text,
    )
    return text


def clean_mbc(text: str) -> str:
    """MBC (193건) — KBS/SBS 유사 패턴"""
    text = re.sub(
        r'\((?:영상취재|영상편집|디자인|촬영기자)\s*[:：][^)]{0,200}\)',
        '',
        text,
    )
    text = re.sub(r'MBC\s*뉴스\s+[가-힣]{2,4}입니다\.?', '', text)
    return text


def clean_broadcast_common(text: str) -> str:
    """방송사 공통 — 앵커/기자/인터뷰 마커"""
    # [앵커] [기자] [리포트] [중계 멘트 : "..."]
    text = re.sub(r'\[\s*(?:앵커|기자|리포트)\s*\]', '', text)
    text = re.sub(r'\[\s*중계\s*멘트\s*[:：][^\]]+\]', '', text)
    # ▶ 싱크 : ...
    text = re.sub(r'▶\s*싱크\s*[:：][^.\n]+', '', text)
    # 단독 "촬영기자:이름" / "영상편집:이름"
    text = re.sub(r'촬영기자\s*[:：]\s*[가-힣]{2,4}(?:\s+[가-힣]{2,4})?', '', text)
    text = re.sub(r'영상편집\s*[:：]\s*[가-힣]{2,4}', '', text)
    text = re.sub(r'영상취재\s*[:：]\s*[가-힣]{2,4}', '', text)
    # 【앵커멘트】【기자】 일본식 괄호
    text = re.sub(r'【[^】]{0,15}】', '', text)
    return text


# ═════════════════════════════════════════════════════════════════════
# 통합 파이프라인
# ═════════════════════════════════════════════════════════════════════

PRESS_HANDLERS = {
    '마이데일리': clean_mydaily,
    'OSEN': clean_osen,
    '엑스포츠뉴스': clean_xports,
    '스포츠조선': clean_sportschosun,
    '마니아타임즈': clean_mania,
}

BROADCAST_HANDLERS = {
    'KBS': clean_kbs,
    'SBS': clean_sbs,
    'YTN': clean_ytn,
    'MBC': clean_mbc,
    'MBN': clean_mbc,            # MBC와 유사한 형식
    'JTBC': clean_mbc,
    'kbc광주방송': clean_kbs,    # KBS 지역사와 유사
    'TV조선': clean_mbc,
    '채널A': clean_mbc,
}

BROADCASTERS = set(BROADCAST_HANDLERS.keys())


def clean_article(body: str, press: str) -> str:
    """단일 기사 정제 — 매체별 분기"""
    if not isinstance(body, str):
        return body

    # 1. Layer 1: 공통 정제
    text = clean_common(body)

    # 2. Layer 2: 신문사 분기 (매체별 함수 있으면 적용)
    if press in PRESS_HANDLERS:
        text = PRESS_HANDLERS[press](text)

    # 3. Layer 3: 방송사 처리
    if press in BROADCASTERS:
        text = clean_broadcast_common(text)
        text = BROADCAST_HANDLERS[press](text)

    # 4. 마지막: 공백·구두점 정리
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*([,.!?])\s*', r'\1 ', text)
    text = re.sub(r'\.{2,}', '.', text)
    # 단독 슬래시·기호 잔재 청소 (예: "...남게 됐습니다. /")
    text = re.sub(r'(?<=[.!?])\s*[/▶■·]+\s*$', '', text)
    text = re.sub(r'\s+[/▶■·]+\s*$', '', text)
    # 빈 괄호 잔재
    text = re.sub(r'\(\s*[,.\s]*\)', '', text)
    text = re.sub(r'\[\s*[,.\s]*\]', '', text)
    text = text.strip()

    return text


# ═════════════════════════════════════════════════════════════════════
# 검수 도구
# ═════════════════════════════════════════════════════════════════════

def load_raw_all() -> pd.DataFrame:
    files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.csv')])
    dfs = [pd.read_csv(os.path.join(RAW_DIR, f), encoding='utf-8-sig') for f in files]
    df = pd.concat(dfs, ignore_index=True).dropna(subset=['body']).reset_index(drop=True)
    return df


def sample_review(df: pd.DataFrame, n_per_press: int = 5, seed: int = 42):
    """매체별로 n_per_press건씩 before/after 비교 출력"""
    sample_press = (
        list(PRESS_HANDLERS.keys())
        + list(BROADCAST_HANDLERS.keys())
        + ['스포티비뉴스', '스포츠서울', '뉴시스']   # 공통 처리만 적용되는 매체 검증
    )

    print(f"\n{'='*70}")
    print(f"  매체별 샘플 검수 (매체당 {n_per_press}건)")
    print(f"{'='*70}")

    rows = []
    for press in sample_press:
        subset = df[df['press'] == press]
        if subset.empty:
            continue
        sample = subset.sample(min(n_per_press, len(subset)), random_state=seed)

        print(f"\n{'─'*70}")
        print(f"  ▼ {press} ({len(subset)}건 中 {len(sample)}건)")
        print(f"{'─'*70}")

        for idx, row in sample.iterrows():
            before = row['body']
            after = clean_article(before, press)
            reduction = (1 - len(after) / len(before)) * 100 if len(before) > 0 else 0

            rows.append({
                'press': press,
                'title': row.get('title', '')[:40],
                'len_before': len(before),
                'len_after': len(after),
                'reduction_pct': round(reduction, 2),
            })

            # 콘솔 출력은 길이 변화가 큰 케이스 위주
            if reduction > 5:
                print(f"\n  ▸ [{row.get('title','')[:50]}] ({len(before)}→{len(after)}자, -{reduction:.1f}%)")
                # 200자 어림 비교
                print(f"    BEFORE 끝: ...{before[-200:].strip()}")
                print(f"    AFTER  끝: ...{after[-200:].strip()}")

    return pd.DataFrame(rows)


def summary_stats(df: pd.DataFrame):
    """전체 적용 결과 요약"""
    print(f"\n{'='*70}")
    print(f"  전체 적용 요약")
    print(f"{'='*70}")

    df = df.copy()
    df['body_v2'] = df.apply(lambda r: clean_article(r['body'], r.get('press', '')), axis=1)
    df['len_before'] = df['body'].str.len()
    df['len_after'] = df['body_v2'].str.len()
    df['reduction_pct'] = (1 - df['len_after'] / df['len_before']) * 100

    print(f"\n  전체 평균 본문 길이")
    print(f"    Before: {df['len_before'].mean():.0f}자")
    print(f"    After : {df['len_after'].mean():.0f}자")
    print(f"    평균 감소율: {df['reduction_pct'].mean():.2f}%")

    print(f"\n  매체별 평균 감소율 (상위 10개):")
    by_press = df.groupby('press').agg(
        n=('body', 'size'),
        mean_red=('reduction_pct', 'mean'),
        len_before=('len_before', 'mean'),
        len_after=('len_after', 'mean'),
    ).sort_values('n', ascending=False).head(10)
    print(by_press.to_string())

    # 위험 케이스: 50% 이상 감소 → 오삭제 의심
    extreme = df[df['reduction_pct'] > 50]
    print(f"\n  ⚠️  감소율 50% 초과 케이스: {len(extreme)}건 ({len(extreme)/len(df)*100:.2f}%)")
    if len(extreme) > 0 and len(extreme) < 20:
        for _, row in extreme.head(5).iterrows():
            print(f"    - [{row.get('press','?')}] {row.get('title','')[:50]}")
            print(f"      ({row['len_before']}→{row['len_after']}자, -{row['reduction_pct']:.1f}%)")

    return df


def save_processed(df: pd.DataFrame):
    """전처리 v2 결과 저장"""
    out_path = os.path.join(OUT_DIR, 'articles_v2.csv')
    df[['title', 'body_v2', 'image_url', 'press', 'date', 'url']].rename(
        columns={'body_v2': 'body'}
    ).to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n  ✅ 저장 완료: {out_path}")

    # 비교용 메타정보 저장
    meta = {
        'total_articles': len(df),
        'mean_len_before': float(df['len_before'].mean()),
        'mean_len_after': float(df['len_after'].mean()),
        'mean_reduction_pct': float(df['reduction_pct'].mean()),
        'press_coverage': {
            'newspaper_handled': list(PRESS_HANDLERS.keys()),
            'broadcast_handled': list(BROADCAST_HANDLERS.keys()),
        },
    }
    with open(os.path.join(LOG_DIR, 'prepro_v2_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ═════════════════════════════════════════════════════════════════════
# 메인
# ═════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=' * 70)
    print('  전처리 v2 — Layer 1 공통 + Layer 2 매체별 분기')
    print('=' * 70)

    df = load_raw_all()
    print(f"  원시 데이터: {len(df):,}건")

    # ── 1단계: 매체별 샘플 검수 (실행 빠름)
    print("\n[1] 매체별 샘플 검수 (5건씩)")
    review_df = sample_review(df, n_per_press=5)
    review_df.to_csv(os.path.join(LOG_DIR, 'prepro_v2_review.csv'),
                     index=False, encoding='utf-8-sig')

    # ── 2단계: 전체 적용 + 통계
    print("\n[2] 전체 적용")
    df_cleaned = summary_stats(df)

    # ── 3단계: 저장
    print("\n[3] 저장")
    save_processed(df_cleaned)

    print(f"\n{'='*70}")
    print(f"  완료. 다음 단계:")
    print(f"  1. results/prepro_v2_review.csv 검수 (매체별 샘플 비교)")
    print(f"  2. 감소율 50% 초과 케이스 확인 (오삭제 의심)")
    print(f"  3. 문제 없으면 articles_v2.csv로 자동 라벨링 재산출")
    print(f"{'='*70}")