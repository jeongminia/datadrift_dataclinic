# Dimensionality Reduction


## **1. Principal Component Analysis**

데이터를 선형 변환하여 분산이 가장 큰 축을 기준으로 데이터를 표현하여 데이터의 분산을 최대화하는 주성분을 찾는 기법

### 1.1 특징
- 선형 기법으로 데이터 분산을 최대 보존
- 가장 큰 분산을 가진 방향을 주성분으로 선택
- 데이터 압축 및 노이즈 제거에 유용
- 고유값 분해 또는 SVD 계산 가능

### 1.2 작동 원리
- 데이터 정규화 : 평균 0, 분산 1 로 스케일링
- 공분산 행렬 계산 후 공분산 행렬의 고유값 $λ$과 고유벡터 $𝑣$를 계산
- 고유값 크기 순으로 정렬해 주요 고유벡터 선택
- 데이터를 고유벡터 공간으로 투영

## 2. Kernel Principal Component Analysis

선형 분리 불가능한 데이터에 대해 Kernel 함수를 사용해 고차원 특징 공간에서 주성분 분석을 수행

### 2.1 특징
- 비선형 차원 축소 수행
- 커널 함수를 사용해 고차원 공간으로 데이터 매핑
- 클러스터링 및 비선형 데이터 분리에 효과적

### 2.2 작동 원리
- 데이터 정규화
- 커널 행렬을 계산한 후 이를 중심화하고 고유값 및 고유벡터 계산
- 데이터를 고유벡터 공간으로 투영

## 3. Truncated Singular Value Decomposition

데이터 정규화는 사용하지 않고 희소 행렬을 사용하여 행렬 분해를 통해 저차원 근사 행렬 생성

### 3.1 특징
- 데이터 스케일링 없이도 작동
- 희소행렬 및 텍스트 데이터에 자주 사용
- 차원 축소를 위한 선형 기법

### 3.2 작동 원리
- 원본 행렬을 특이값 분해 후 상위 $k$개의 특이값과 대응하는 벡터만 사용하여 저차원 근사 생성
- 데이터를 축소된 특이벡터 공간으로 표현

## 4. Gaussian Random Projection

고차원 데이터를 무작위 투영 기법을 사용해 저차원으로 축소해 유클리드 거리를 근사적으로 보존

### 4.1 특징
- 매우 빠르고 계산 효율적
- 대규모 데이터에 적합
- 정보 보존이 확률적이므로 보장이 어렵지만 적절한 차원을 설정하면 성능 우수

### 4.2 작동 원리
- 원본 데이터 $𝑋 ∈ 𝑅^{𝑛×𝑑}$ 와 랜덤 투영 행렬 $R∈R^{d×k}$ 생성
- 데이터에 무작위 투영 적용

## 5. Autoencoder

신경망을 기반으로 입력 데이터를 저차원 잠재 공간으로 압축한 후 다시 복원하는 방식으로 학습하는 비선형 차원 축소 기법

### 5.1 특징
- 비선형 기법으로 복잡한 데이터 구조를 효과적으로 학습 가능
- 고차원 데이터의 압축과 특징 학습에 적합
- 학습 기반으로 PCA보다 계산 비용이 높음

### 5.2 작동 원리
- 신경망 구성
    - 인코더 : 입력 데이터를 잠재 공간으로 압축
    - 디코더 : 잠재 공간 데이터를 원래 데이터로 복원
- 손실 함수 최소화
    입력 데이터 $X$와 복원 데이터 $\hat{𝑋}$간의 손실을 최소화
- 학습 완료 후, 인코더를 사용해 데이터를 저차원 공간으로 매핑 
    → $Z = Encoder(X)$


## Summary
| **방법** | **특징** | **선형/비선형** | **수식** |
| --- | --- | --- | --- |
| **PCA** | 분산 보존, 노이즈 제거 | 선형 | $\mathbf{C} = \frac{1}{n-1} \mathbf{X}^\top \mathbf{X}$ |
| **Kernel PCA** | 비선형 분리 가능 | 비선형 | $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ |
| **Truncated SVD** | 희소 행렬 및 텍스트 데이터에 유용 | 선형 | $\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top$ |
| **GRP** | 빠르고 계산 효율적, 거리 보존 | 선형 | $\mathbf{Z} = \frac{1}{\sqrt{k}} \mathbf{X} \mathbf{R}$ |
| **Autoencoder** | 비선형 데이터 구조 학습, 복잡한 데이터에 적합 | 비선형 | $\mathcal{L} = \|\mathbf{X} - \hat{\mathbf{X}}\|^2$ |

## References
[1] E. Bingham and H. Mannila, "Random projection in dimensionality reduction: Applications to image and text data," Proceedings of the Seventh ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, vol. 7, no. 1, pp. 245–250, 2001.

[2] S. Mika, G. Ratsch, J. Weston, B. Scholkopf, A. Smola, "Fisher discriminant analysis with kernels,” Proceedings of the IEEE Neural Networks for Signal Processing IX, vol. 1, pp. 41-48, 1999.

[3] C. Hu, X. Hou, Y. Lu, "Improving the Architecture of an Autoencoder for Dimension Reduction,” Proceedings of the 2014 IEEE International Conference on Ubiquitous Intelligence and Computing, Autonomic and Trusted Computing, Scalable Computing and Communications, and Its Associated Workshops, pp. 851-858, 2014.
