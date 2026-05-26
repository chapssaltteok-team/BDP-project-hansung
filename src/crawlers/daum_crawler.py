"""
daum_crawler.py
===============
다음 스포츠 KBO 뉴스 수집
담당: 크롤링 담당자

수집 방식
─────────
1차: JSON API 직접 호출 (빠름, 안정적)
2차: HTML 폴백 (API 실패 시)

수집 컬럼
─────────
title, body, image_url, press, date, url, source

실행: python src/crawlers/daum_crawler.py
"""
import os
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
    'Referer': 'https://sports.daum.net/',
}

# 다음 스포츠 KBO 뉴스 API
DAUM_API_URL = (
    "https://sports-core.kakao.com/v2/news/categories/kbo"
    "?offset={offset}&limit=40&date={date}"
)

DAUM_LIST_URL = (
    "https://sports.daum.net/news/baseball/kbo"
    "?date={date}"
)


# ── 날짜 범위 생성 ────────────────────────────────────────────────────────────
def date_range(start: str, end: str):
    s = datetime.strptime(start, '%Y%m%d')
    e = datetime.strptime(end, '%Y%m%d')
    while s <= e:
        yield s.strftime('%Y%m%d')
        s += timedelta(days=1)


# ── JSON API 방식 ─────────────────────────────────────────────────────────────
def fetch_list_via_api(date_str: str) -> list[dict]:
    """다음 스포츠 내부 API로 기사 목록 수집"""
    articles = []
    offset = 0

    while True:
        url = DAUM_API_URL.format(offset=offset, date=date_str)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [API 실패] {date_str} offset={offset}: {e}")
            break

        items = (
            data.get('items') or
            data.get('news') or
            data.get('list') or
            data.get('result', {}).get('items') or
            []
        )
        if not items:
            break

        for item in items:
            link = item.get('url') or item.get('link') or item.get('newsUrl', '')
            if not link:
                continue
            articles.append({
                'url'      : link,
                'title'    : item.get('title', '').strip(),
                'press'    : item.get('mediaName') or item.get('officeName', ''),
                'image_url': item.get('thumbnail') or item.get('imageUrl', ''),
                'date'     : date_str,
            })

        if len(items) < 40:
            break
        offset += 40
        time.sleep(random.uniform(0.3, 0.6))

    print(f"  [API] {date_str}: {len(articles)}건")
    return articles


# ── HTML 폴백 방식 ────────────────────────────────────────────────────────────
def fetch_list_via_html(date_str: str) -> list[dict]:
    """API 실패 시 HTML 파싱 폴백"""
    url = DAUM_LIST_URL.format(date=date_str)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
    except Exception as e:
        print(f"  [HTML 폴백 실패] {date_str}: {e}")
        return []

    articles = []
    for a in soup.select('a[href*="/news/"]'):
        href = a.get('href', '')
        if not href.startswith('http'):
            href = 'https://sports.daum.net' + href
        title_tag = a.select_one('strong, em, span.tit')
        title = title_tag.get_text(strip=True) if title_tag else a.get_text(strip=True)
        if not title or len(title) < 5:
            continue
        articles.append({
            'url'      : href,
            'title'    : title,
            'press'    : '',
            'image_url': '',
            'date'     : date_str,
        })

    print(f"  [HTML 폴백] {date_str}: {len(articles)}건")
    return articles


# ── 기사 본문 파싱 ────────────────────────────────────────────────────────────
def parse_article_body(url: str, existing_image_url: str = '') -> tuple[str, str]:
    if not url or not url.startswith('http'):
        return '', existing_image_url
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception:
        return '', existing_image_url

    soup = BeautifulSoup(resp.text, 'html.parser')

    image_url = existing_image_url
    if not image_url:
        og = soup.select_one('meta[property="og:image"]')
        if og and og.get('content'):
            image_url = og['content'].strip()

    body_selectors = [
        'div.article_view',
        'div#harmonyContainer',
        'div.news_view',
        'div.article_body',
        'div[class*="article"]',
        'article',
    ]
    for sel in body_selectors:
        tag = soup.select_one(sel)
        if tag:
            for rm in tag.select('script, style, iframe, figure, .ad'):
                rm.decompose()
            text = tag.get_text(separator=' ', strip=True)
            if len(text) > 100:
                return text, image_url

    return '', image_url


# ── 메인 크롤링 ───────────────────────────────────────────────────────────────
def crawl(
    start_date: str   = '20260401',
    end_date  : str   = '20260413',
    delay     : float = 0.8,
) -> pd.DataFrame:

    print(f"\n[다음] {start_date} ~ {end_date} 크롤링 시작")

    all_articles = []
    seen_urls = set()

    for date_str in date_range(start_date, end_date):
        articles = fetch_list_via_api(date_str)
        if not articles:
            articles = fetch_list_via_html(date_str)

        new_count = 0
        for art in articles:
            if art['url'] in seen_urls:
                continue
            seen_urls.add(art['url'])

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
                'source'   : 'daum',
            })
            new_count += 1
            time.sleep(random.uniform(delay * 0.5, delay * 1.2))

        print(f"  → {date_str} 완료: 누적 {len(all_articles)}건 (+{new_count})")

    if not all_articles:
        print("\n[경고] 수집된 기사가 0건입니다.")
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)
    save_path = os.path.join(RAW_DIR, 'daum_news.csv')
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n[완료] {len(df)}건 저장 → {save_path}")
    return df


if __name__ == '__main__':
    crawl(
        start_date='20260401',
        end_date  ='20260413',
        delay     = 0.8,
    )
