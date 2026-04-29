"""
press_crawler.py
================
스포츠조선 + OSEN 통합 크롤러 (TV조선 제외)
담당: 크롤링 담당자

수집 방식
─────────
requests + BeautifulSoup (두 사이트 모두 정적 HTML)

수집 컬럼
─────────
title, body, image_url, press, date, url, source

실행: python src/crawlers/press_crawler.py
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
}

# ── 사이트별 URL 설정 ─────────────────────────────────────────────────────────
SITES = {
    'sportschosun': {
        'list_url': 'https://www.sportschosun.com/news/baseball/kbo_news.htm?date={date}',
        'base_url': 'https://www.sportschosun.com',
        'list_selector' : 'ul.news_list li a, div.news_list_wrap a',
        'title_selector': 'h1.tit, h2.art_tit, div.article_header h1',
        'body_selector' : 'div.article_txt, div#articleContent, div.art_txt',
    },
    'osen': {
        'list_url': 'https://osen.mt.co.kr/baseball/index.html?page=1&date={date}',
        'base_url': 'https://osen.mt.co.kr',
        'list_selector' : 'ul.news_list li a, div.list_area a.tit',
        'title_selector': 'h1.article_tit, div.article_head h2, h1.tit',
        'body_selector' : 'div#articleContent, div.article_body, div.news_txt',
    },
}


def date_range(start: str, end: str):
    s = datetime.strptime(start, '%Y%m%d')
    e = datetime.strptime(end, '%Y%m%d')
    while s <= e:
        yield s.strftime('%Y%m%d')
        s += timedelta(days=1)


# ── 목록 수집 ─────────────────────────────────────────────────────────────────
def fetch_list(site_key: str, date_str: str) -> list[dict]:
    cfg = SITES[site_key]
    url = cfg['list_url'].format(date=date_str)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
    except Exception as e:
        print(f"  [{site_key}] 목록 실패 {date_str}: {e}")
        return []

    articles = []
    for a in soup.select(cfg['list_selector']):
        href = a.get('href', '')
        if not href:
            continue
        if not href.startswith('http'):
            href = cfg['base_url'] + href
        # KBO 관련 URL 필터
        if not any(kw in href for kw in ['baseball', 'kbo', 'sports']):
            continue
        title = a.get_text(strip=True)
        if len(title) < 5:
            continue
        articles.append({'url': href, 'title': title, 'press': site_key, 'date': date_str})

    print(f"  [{site_key}] {date_str}: 목록 {len(articles)}건")
    return articles


# ── 본문 파싱 ─────────────────────────────────────────────────────────────────
def parse_article_body(url: str, site_key: str) -> tuple[str, str]:
    cfg = SITES[site_key]
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
    except Exception:
        return '', ''

    image_url = ''
    og = soup.select_one('meta[property="og:image"]')
    if og and og.get('content'):
        image_url = og['content'].strip()

    for sel in cfg['body_selector'].split(', '):
        tag = soup.select_one(sel.strip())
        if tag:
            for rm in tag.select('script, style, iframe, .ad, figure'):
                rm.decompose()
            text = tag.get_text(separator=' ', strip=True)
            if len(text) > 100:
                return text, image_url

    return '', image_url


# ── 사이트별 크롤링 ───────────────────────────────────────────────────────────
def crawl_site(
    site_key  : str,
    start_date: str,
    end_date  : str,
    delay     : float,
) -> list[dict]:
    all_articles = []
    seen_urls = set()

    for date_str in date_range(start_date, end_date):
        articles = fetch_list(site_key, date_str)
        new_count = 0

        for art in articles:
            if art['url'] in seen_urls:
                continue
            seen_urls.add(art['url'])

            body, img = parse_article_body(art['url'], site_key)
            if not body:
                continue

            all_articles.append({
                'title'    : art['title'],
                'body'     : body,
                'image_url': img,
                'press'    : site_key,
                'date'     : date_str,
                'url'      : art['url'],
                'source'   : site_key,
            })
            new_count += 1
            time.sleep(random.uniform(delay * 0.5, delay * 1.2))

        print(f"  → {date_str} 완료: 누적 {len(all_articles)}건 (+{new_count})")

    return all_articles


# ── 메인 ──────────────────────────────────────────────────────────────────────
def crawl(
    start_date: str   = '20260401',
    end_date  : str   = '20260413',
    delay     : float = 1.0,
) -> pd.DataFrame:

    all_data = []

    for site_key in ['sportschosun', 'osen']:
        print(f"\n[{site_key.upper()}] {start_date} ~ {end_date} 크롤링 시작")
        data = crawl_site(site_key, start_date, end_date, delay)
        all_data.extend(data)
        print(f"  [{site_key}] 소계: {len(data)}건")

    if not all_data:
        print("\n[경고] 수집된 기사가 0건입니다.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    save_path = os.path.join(RAW_DIR, 'press_news.csv')
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n[완료] 총 {len(df)}건 저장 → {save_path}")
    return df


if __name__ == '__main__':
    crawl(
        start_date='20260401',
        end_date  ='20260413',
        delay     = 1.0,
    )
