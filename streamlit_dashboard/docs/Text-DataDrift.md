# Data Drift in Text Data

텍스트 데이터는 이미지, 오디오 등과 함께 비정형 데이터에 속하며 전통적인 방식으로는 직접적인 분포 비교가 어렵다. 따라서 자연어 처리에서는 임베딩 방식을 활용해 텍스트 데이터를 고차원 벡터 공간으로 변환한 후, 데이터 드리프트를 탐지하는 것이 일반적이다. 본 문서에서는 임베딩을 활용한 텍스트 데이터 드리프트 탐지 방법과 각 접근 방식의 특성을 비교한다.

---
## 1. 임베딩 벡터 변환
### 1.1 비정형 데이터 특성
- 텍스트 데이터는 단순 수치형 데이터나 테이블형 데이터와 달리 직접적인 비교가 어려움
- 즉 '사과'와 '바나나' 라는 단어는 개별 문자로 비교하면 유사성이 전혀 없지만 의미적으로는 둘 다 과일이라는 공통점이 있음
- 이러한 의미 기반의 비교를 위해 임베딩 활용

### 1.2 임베딩 벡터 변환
임베딩을 활용하면 텍스트 데이터를 다차원 공간의 벡터로 변환할 수 있다. 대표적인 방법으로 다음과 같은 사전 학습된 언어모델을 채택할 수 있다.

- BERT
- FastText
- Word2Vec
- GloVe

이러한 모델을 활용하면 텍스트를 고차원 벡터로 변환할 수 있어 해당 벡터를 기반으로 데이터 드리프트 탐지


### 1.3 텍스트 데이터에서 발생할 수 있는 드리프트 예시
- **새로운 인기 단어나 밈 등장**
    - ex. "챗GPT" 같은 신조어 증가
- **새로운 클래스 등장 또는 클래스 비율 변화**
    - ex. 제품 리뷰에서 "친환경" 관련 키워드 급증
- **스팸성 콘텐츠 증가**
    - ex. 광고성 문구 증가
- **감정 변화 (긍정 → 부정)**
    - ex. 특정 브랜드에 대한 소비자 인식 변화
- **새로운 언어 등장**
    - ex. 기존 영어 리뷰에 스페인어 리뷰 증가

## 2. 텍스트 데이터 드리프트 탐지 방법 비교
드리프트 탐지 방법 비교는 다음과 같다.

- **유클리드 거리 (Euclidean Distance)**
    - 두 데이터셋의 평균 임베딩 간 거리를 계산
    - 값이 클수록 두 데이터셋 간 차이가 크다는 의미
- **코사인 거리 (Cosine Distance)**
    - 두 벡터 간 각도를 계산
    - 동일한 벡터는 0, 완전히 다른 벡터는 2
- **모델 기반 탐지 (Classifier)**
    - 바이너리 분류기를 학습해 참조와 현재 데이터를 구별
    - ROC AUC(0.5~1)로 드리프트 여부 판단
- **임베딩 구성 요소 비율 (Share of Drifted Components)**
    - 각 임베딩의 수치 구성 요소 간 변화 비율 측정
    - 변화된 구성 요소 비율이 일정 임계값을 넘으면 드리프트로 판단
- **최대 평균 차이 (Maximum Mean Discrepancy, MMD)**
    - 두 분포 간 평균 차이 측정
    - 값이 0에 가까우면 동일, 값이 클수록 차이가 큼



----
## References
[1] https://www.evidentlyai.com/blog/embedding-drift-detection
[2] R. Feldhans, A. Wilke, S. Heindorf, M. H. Shaker, B. Hammer, A.-C. Ngonga Ngomo, and E. Hüllermeier, "Drift detection in text data with document embeddings," in *Machine Learning and Knowledge Discovery in Databases. Research Track*, M. T. Ribeiro, J. Gama, and A. Lyons, Eds. Cham, Switzerland: Springer, 2022, pp. 179–195. [Online]. Available: https://doi.org/10.1007/978-3-030-91608-4_11