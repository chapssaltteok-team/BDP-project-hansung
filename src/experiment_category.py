import pandas as pd
import os, json, re
from sklearn.model_selection import train_test_split
from prepro import clean_text, label_category

CATEGORY_KEYWORDS_EXP = {
    '타격': ['홈런','안타','타율','타점','출루율','장타','타격','배팅','득점'],
    '투구': ['삼진','방어율','선발','불펜','마무리','투구','완봉','세이브','홀드','구위'],
    '이적/계약': ['트레이드','이적','교환','영입','방출','FA','자유계약','연봉','계약'],
    '부상/재활': ['부상','등록 말소','수술','재활','회복','결장','통증','접질'],
    '감독/코치': ['감독','코치','벤치','작전','선발 로테이션','경질','수뇌부'],
    '구단운영': ['구단','프런트','스프링캠프','단장','구단주','재정','마케팅'],
    '심판/판정': ['심판','오심','판정','비디오 판독','어필','스트라이크존'],
    '기록/순위': ['신기록','최다','연승','연패','순위','포스트시즌','매직넘버'],
    '수비/실책': ['실책','포구','송구','병살','다이빙캐치','내야수','외야수'],
    '고교/드래프트': ['신인','드래프트','지명','유망주','육성군','2군'],
    '해외파/MLB': ['메이저리그','MLB','포스팅','해외진출','진출','복귀'],
    '관중/이벤트': ['관중','매진','사인회','팬서비스','시구','응원','홈구장'],
    '날씨/장소': ['우천취소','돔구장','잔디','구장','원정경기','더블헤더'],
    '국가대표': ['WBC','프리미어12','아시안게임','국대','대표팀','차출'],
    '사건/사고': ['음주운전','도핑','징계','사과','논란','갈등','충돌']
}

RAW_DIR = 'data/raw'
EXP_PROC_DIR = 'data/processed_exp'
os.makedirs(EXP_PROC_DIR, exist_ok=True)

def run_category_experiment(n_categories):
    print(f"🚀 [데이터 풀 가동] 카테고리 {n_categories}개 생성 중...")
    csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
    dfs = [pd.read_csv(os.path.join(RAW_DIR, f), encoding='utf-8-sig') for f in csv_files]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='url')
    
    # 텍스트 길이를 2000자로 늘려 기사 본문 정보를 거의 다 담습니다.
    df['text'] = (df['title'].apply(clean_text) + ' ' + df['body'].apply(clean_text)).str[:2000]
    df['temp_full_cat'] = df['text'].apply(lambda t: label_category(t, CATEGORY_KEYWORDS_EXP))
    
    valid_cats = df[df['temp_full_cat'] != '기타']['temp_full_cat'].value_counts()
    top_n_cats = valid_cats.nlargest(n_categories).index.tolist()
    df['category'] = df['temp_full_cat'].apply(lambda x: x if x in top_n_cats else '기타')
    
    # [수정] 샘플링 없이 모든 유효 데이터를 다 사용합니다.
    df = df.reset_index(drop=True)
    
    # 에러 방지용 가짜 감성 컬럼
    df['sentiment_str'] = ['긍정' if i % 2 == 0 else '부정' for i in range(len(df))]

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv(os.path.join(EXP_PROC_DIR, f'train_cat_{n_categories}.csv'), index=False, encoding='utf-8-sig')
    val_df.to_csv(os.path.join(EXP_PROC_DIR, f'val_cat_{n_categories}.csv'), index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    for n in [3, 6, 9, 12, 15]:
        run_category_experiment(n)
    print("✅ 원본 규모 데이터셋 생성 완료.")