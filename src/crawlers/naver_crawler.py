"""
naver_crawler.py
================
네이버 스포츠 KBO 뉴스 수집
담당: 크롤링 담당자

수집 방식
─────────
1차: requests + BeautifulSoup (HTML 파싱)
     네이버가 React SPA이면 0건 → Selenium으로 자동 전환
2차: Selenium (Chrome headless) — JS 렌더링 후 파싱
기사 본문: n.news.naver.com 으로 URL 변환 후 requests 파싱 (정적 HTML)

수집 컬럼
─────────
title, body, image_url, press, date, url, source

실행: python src/crawlers/naver_crawler.py
"""
import os
import re
import time
import random
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

RAW_DIR = 'data/raw'
os.makedirs(RAW_DIR, exist_ok=True)

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Referer': 'https://sports.naver.com/',
}

# 네이버 스포츠 모바일 URL (날짜별)
NAVER_LIST_URL = (
    "https://m.sports.naver.com/kbaseball/news"
    "?sectionId=kbaseball&sort=latest&date={date}&isPhoto=N"
)


# ── 날짜 범위 생성 ────────────────────────────────────────────────────────────
def date_range(start: str, end: str):
    """YYYYMMDD 문자열 범위를 하루씩 yield"""
    s = datetime.strptime(start, '%Y%m%d')
    e = datetime.strptime(end, '%Y%m%d')
    while s <= e:
        yield s.strftime('%Y%m%d')
        s += timedelta(days=1)


# ── HTML 파싱 (기사 목록) ─────────────────────────────────────────────────────
def _parse_list_html(soup: BeautifulSoup, date_str: str) -> list[dict]:
    """
    실제 HTML 구조 기반 선택자
    li[class*='NewsItem_news_item'] → 제목·URL·언론사·썸네일
    """
    items = soup.select('li[class*="NewsItem_news_item"]')
    if not items:
        return []

    result = []
    for item in items:
        a = item.select_one('a[class*="NewsItem_link_news"]')
        if not a:
            continue
        href = a.get('href', '')
        if not href:
            continue
        if not href.startswith('http'):
            href = 'https://sports.news.naver.com' + href

        title_tag = item.select_one('em[class*="NewsItem_title"]')
        title = title_tag.get_text(strip=True) if title_tag else ''

        press_tag = item.select_one('span[class*="NewsItem_press"]')
        press = press_tag.get_text(strip=True) if press_tag else ''

        # 썸네일 — 목록에서 바로 추출 (개별 기사 방문 불필요)
        img_tag = item.select_one('div[class*="NewsItem_image_wrap"] img')
        image_url = ''
        if img_tag and img_tag.get('src'):
            from urllib.parse import unquote, parse_qs, urlparse
            thumb_src = img_tag['src']
            try:
                qs = parse_qs(urlparse(thumb_src).query)
                src_val = qs.get('src', [''])[0]
                image_url = unquote(src_val).strip('"')
            except Exception:
                image_url = thumb_src

        result.append({
            'url'      : href,
            'title'    : title,
            'press'    : press,
            'image_url': image_url,
            'date'     : date_str,
        })

    print(f"    NewsItem 선택자로 {len(result)}건 파싱 성공")
    return result


# ── requests 방식 목록 수집 ───────────────────────────────────────────────────
def fetch_list_via_requests(date_str: str) -> list[dict]:
    url = NAVER_LIST_URL.format(date=date_str)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        return _parse_list_html(soup, date_str)
    except Exception as e:
        print(f"  [requests 실패] {date_str}: {e}")
        return []


# ── Selenium 방식 목록 수집 ───────────────────────────────────────────────────
def fetch_list_via_selenium(date_str: str, driver) -> list[dict]:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    url = NAVER_LIST_URL.format(date=date_str)
    driver.get(url)
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'li[class*="NewsItem_news_item"]')
            )
        )
    except Exception:
        print(f"  [Selenium 대기 시간 초과] {date_str}")

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    result = _parse_list_html(soup, date_str)
    print(f"  [Selenium] {date_str}: {len(result)}건 수집")
    return result


# ── 기사 본문 파싱 ────────────────────────────────────────────────────────────
def parse_article_body(url: str, existing_image_url: str = '') -> tuple[str, str]:
    """
    sports.news.naver.com → n.news.naver.com 변환 후 정적 HTML 파싱
    Returns: (body_text, image_url)
    """
    # URL 변환
    m = re.search(r'/article/(\d+)/(\d+)', url)
    if m:
        oid, aid = m.group(1), m.group(2)
        url = f"https://n.news.naver.com/mnews/article/{oid}/{aid}"

    if not url or not url.startswith('http'):
        return '', existing_image_url

    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception:
        return '', existing_image_url

    soup = BeautifulSoup(resp.text, 'html.parser')

    # 이미지 — 목록에서 이미 수집한 경우 건너뜀
    image_url = existing_image_url
    if not image_url:
        og = soup.select_one('meta[property="og:image"]')
        if og and og.get('content'):
            image_url = og['content'].strip()

    # 본문 선택자 (우선순위 순)
    body_selectors = [
        'div#articleBodyContents',
        'div.newsct_article',
        'div._article_body_contents',
        'div#newsEndContents',
        'div.news_end',
        'div._article_content',
        'div.article_body',
        'article',
    ]
    for sel in body_selectors:
        tag = soup.select_one(sel)
        if tag:
            for rm in tag.select('script, style, iframe, div.ad_area, figure'):
                rm.decompose()
            text = tag.get_text(separator=' ', strip=True)
            if len(text) > 100:
                return text, image_url

    return '', image_url


# ── Selenium 드라이버 초기화 ──────────────────────────────────────────────────
def init_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument(f'user-agent={HEADERS["User-Agent"]}')
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )
    return driver


# ── 메인 크롤링 ───────────────────────────────────────────────────────────────
def crawl(
    start_date  : str   = '20260401',
    end_date    : str   = '20260413',
    delay       : float = 1.0,
    use_selenium: bool  = False,
) -> pd.DataFrame:

    print(f"\n[네이버] {start_date} ~ {end_date} 크롤링 시작")
    print(f"  URL: {NAVER_LIST_URL.format(date='YYYYMMDD')}")

    driver = None
    if use_selenium:
        driver = init_driver()

    all_articles = []
    seen_urls = set()

    try:
        for date_str in date_range(start_date, end_date):
            if use_selenium and driver:
                articles = fetch_list_via_selenium(date_str, driver)
            else:
                articles = fetch_list_via_requests(date_str)
                if not articles:
                    print(f"  → requests 0건, Selenium으로 재시도")
                    if driver is None:
                        driver = init_driver()
                    articles = fetch_list_via_selenium(date_str, driver)

            new_count = 0
            for art in articles:
                if art['url'] in seen_urls:
                    continue
                seen_urls.add(art['url'])

                # 본문 수집
                body, img = parse_article_body(
                    art['url'], existing_image_url=art.get('image_url', '')
                )
                if not body:
                    continue

                all_articles.append({
                    'title'    : art['title'],
                    'body'     : body,
                    'image_url': img,
                    'press'    : art.get('press', ''),
                    'date'     : date_str,
                    'url'      : art['url'],
                    'source'   : 'naver',
                })
                new_count += 1
                time.sleep(random.uniform(delay * 0.5, delay * 1.2))

            print(f"  → {date_str} 완료: 누적 {len(all_articles)}건 (+{new_count})")

    finally:
        if driver:
            driver.quit()

    if not all_articles:
        print("\n[경고] 수집된 기사가 0건입니다.")
        print("  해결 방법:")
        print("  1. pip install selenium webdriver-manager")
        print("  2. crawl(use_selenium=True) 로 재실행")
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)
    save_path = os.path.join(RAW_DIR, 'naver_news.csv')
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n[완료] {len(df)}건 저장 → {save_path}")
    return df


if __name__ == '__main__':
    crawl(
        start_date  = '20260301',
        end_date    = '20260425',
        delay       = 1.0,
        use_selenium= True,   # 네이버는 Selenium 권장
    )
