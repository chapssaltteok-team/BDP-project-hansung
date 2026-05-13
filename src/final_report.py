import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def calculate_best_score(train_p, val_p):
    # 데이터 로드
    train = pd.read_csv(train_p)
    val = pd.read_csv(val_p)
    
    # TF-IDF 설정: max_features를 10,000개로 늘리고, 단어 조합(ngram)을 활용해 성능 극대화
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    X_train = tfidf.fit_transform(train['text'].fillna(''))
    X_val = tfidf.transform(val['text'].fillna(''))
    
    # 모델 설정: 규제 파라미터(C)를 조절해 학습 최적화
    model = LogisticRegression(C=5.0, max_iter=2000, multi_class='multinomial', solver='lbfgs')
    model.fit(X_train, train['category'])
    
    # F1 Score 계산 (Macro)
    score = f1_score(val['category'], model.predict(X_val), average='macro')
    return round(score, 4)

def generate_report():
    results = []
    # 실험할 카테고리 개수 리스트
    target_categories = [3, 6, 9, 12, 15]
    
    for n in target_categories:
        t_p = f'data/processed_exp/train_cat_{n}.csv'
        v_p = f'data/processed_exp/val_cat_{n}.csv'
        
        if os.path.exists(t_p):
            score = calculate_best_score(t_p, v_p)
            results.append({
                "카테고리 개수": f"{n}개", 
                "F1 Score (Macro)": score
            })
            print(f"📊 {n}개 카테고리 분석 완료... (Score: {score})")
    
    # 최종 결과 보고서 출력
    if results:
        print("\n" + "="*35)
        print("      [ 모델 성능 실험 결과 ]")
        print("-"*35)
        print(pd.DataFrame(results).to_string(index=False))
        print("="*35)
    else:
        print("❌ 분석할 데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")

if __name__ == "__main__":
    generate_report()