# Dimensionality Reduction


## **1. Principal Component Analysis**

데이터를 선형 변환하여 분산이 가장 큰 축을 기준으로 데이터를 표현하여 데이터의 분산을 최대화하는 주성분을 찾는 기법

### 1.1 특징

### 1.2 작동 원리

## 2. Kernel Principal Component Analysis

선형 분리 불가능한 데이터에 대해 Kernel 함수를 사용해 고차원 특징 공간에서 주성분 분석을 수행

### 2.1 특징

### 2.2 작동 원리

## 3. Truncated Singular Value Decomposition

데이터 정규화는 사용하지 않고 희소 행렬을 사용하여 행렬 분해를 통해 저차원 근사 행렬 생성

### 3.1 특징

### 3.2 작동 원리

## 4. Gaussian Random Projection

고차원 데이터를 무작위 투영 기법을 사용해 저차원으로 축소해 유클리드 거리를 근사적으로 보존

### 4.1 특징

### 4.2 작동 원리

## 5. Autoencoder

신경망을 기반으로 입력 데이터를 저차원 잠재 공간으로 압축한 후 다시 복원하는 방식으로 학습하는 비선형 차원 축소 기법

### 5.1 특징

### 5.2 작동 원리

## Summary
| **방법** | **특징** | **선형/비선형** | **수식** |
| --- | --- | --- | --- |
| **PCA** | 분산 보존, 노이즈 제거 | 선형 | $\mathbf{C} = \frac{1}{n-1} \mathbf{X}^\top \mathbf{X}$ |
| **Kernel PCA** | 비선형 분리 가능 | 비선형 | $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ |
| **Truncated SVD** | 희소 행렬 및 텍스트 데이터에 유용 | 선형 | $\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top$ |
| **GRP** | 빠르고 계산 효율적, 거리 보존 | 선형 | $\mathbf{Z} = \frac{1}{\sqrt{k}} \mathbf{X} \mathbf{R}$ |
| **Autoencoder** | 비선형 데이터 구조 학습, 복잡한 데이터에 적합 | 비선형 | $\mathcal{L} = \|\mathbf{X} - \hat{\mathbf{X}}\|^2$ |

## References
