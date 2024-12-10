# README

> **📌 Initial settings**
>    - 운영 환경에서 data drift 탐지를 목적으로 하기에 차원축소 전에 학습은 trainset으로만 적용    
> ----
>**data**
>  - train : law domain train data
>  - valid : law domain valid data
>  - test : LBOX casename test data       
>
> **purpose**
>  - 전체 데이터의 변화를 모니터링하는 것은 비효율적
>  - 드리프트가 특정 차원에 국한될 경우 불필요한 차원이 오히려 탐지 방해

---


## 1. [DataDrift without Dimension Reduction](datadrift_without-dr.ipynb)

---

> 차원 축소 없이 데이트 드리프트 탐지 진행

|  | Opensource | feature | `valid` vs `train` | `test` vs **`train`** |
| --- | --- | --- | --- | --- |
| **Maximum Mean Discrepancy** | EvidentlyAI | 전체 데이터 분포를 직접 비교하는 고차원 분포 간 유사성 측정 | 0.0001 | 0.2087 |
| **Wasserstein Distance** | EvidentlyAI | 분포 간의 이동 거리 계산, 분포 형태보다는 거리 비용에 초점 | 0.0247 | 0.9635 |
| **Energy Distacne** | EvidentlyAI | 거리 기반 접근으로 중심 및 분산 차이를 동시에 고려 | 0.0 | 0.474 |

 * drift score : 선택된 메트릭을 통해 Reference Data와 Current Data의 분포 차이를 수치화한 값

</br>

[EvidentlyAI](https://docs.evidentlyai.com/) 는 Report를 제공하고 있어 아래와 같은 시각화 결과를 도출할 수 있음

|  | `valid` vs `train` | `test` vs `train` |
| --- | --- | --- |
| **MMD** | ![**valid** vs ** | ![**test**  |
| **Wasserstein Distance** | ![**valid** vs ** | ![**test**  |
| **Energy Distacne** | ! | ![**test** vs **train**]() |



## 2. [Methods for Dimension Reduction](dimension-reduction_base.ipynb)

---

> **dimension reduction order**
> 
> 1. train에 대해서만 먼저 적합 진행
>     
>     train 기준으로 차원 축소를 진행한 뒤, 다른 데이터로 같은 공간에 투영해 변화 감지
>     
>     → 실제 시스템에서 배포된 모델이 학습한 데이터와 새로 들어온 데이터 간의 변화를 감지하는 데 효과적
>     
> 2. 병합한 뒤 pca 적용
> 3. 각각 pca 적용

### 2.1 Select Dim

|  | PCA | KernelPCA | Truncated SVD | GRP | Autoencoder |
| --------- | --------- | -------- | --------- | --------- | --------- |
| **Plot**                     |  ![alt text](image.png) | ![alt text](image-15.png) |![alt text](image-5.png)| ![alt text](image-10.png)   |  |
| **Stress**                   | 93                      |      91                   |          75          |       88          |  |
| **Explained Variance Ratio** | 188                     |      -                     |           -          |           -      |  |
| **Reconstruction Error**     | 115                     |     101                   |           111        |         -        |  |
| **size of dim**              | 188                     |      101                     |         111          |       88          |                             |


적절한 차원 선택을 위해 데이터의 구조적 변화를 감지하기 위해 평가방법에 따라 다른 메트릭을 선택

</br>

### 2.2 Visualization of Dimension Reduction Results

데이터 드리프트 탐지 목적이기에, 차원 축소를 **train-valid**와 **train-test**에서 각각 따로 진행

#### 2.2.1 Distance

|  | dim | `valid` vs `train` | `test` vs **`train`** |
| --- | --- | --- | --- |
| PCA | 188 | ![alt text](image-1.png) | ![alt text](image-2.png) |
| Kernel PCA | 88 | ![alt text](image-16.png) | ![alt text](image-17.png) |
| Truncated SVD | 111 | ![alt text](image-6.png) | ![alt text](image-7.png) |
| GRP | 88 | ![alt text](image-11.png) | ![alt text](image-12.png) |
| AutoEncoder | 328 | ![alt text](img_fil4es/image-3.png) | ![alt text](img_f4iles/image-1.png) |


#### 2.2.2 Plot

|  | dim | `valid` vs `train` | `test` vs **`train`** |
| --- | --- | --- | --- |
| PCA | 188 | ![alt text](image-3.png) | ![alt text](image-4.png) |
| Kernel PCA | 77 | ![alt text](image-18.png) | ![alt text](image-19.png) |
| Truncated SVD | 111 | ![alt text](image-8.png) | ![alt text](image-9.png) |
| GRP | 88 | ![alt text](image-13.png) | ![alt text](image-14.png) |
| AutoEncoder | 328 | ![alt text](img_fil4es/image-2.png) | ![alt text](img_file4s/image-28.png) |

#### 2.2.3 Evaluation



</br>

## 3. [DataDrift with Dimension Reduction](datadrift_with-dr.ipynb)

---


- 차원축소를 적용한 경우, 대부분의 메트릭에서 Valid와 Test 간의 드리프트 점수가 Base보다 높거나 명확하게 나타남
- 이는 고차원 데이터의 노이즈가 줄어들고, 데이터의 주요 특징만 남게 되어 드리프트 감지가 더 효과적으로 이루어진 것으로 해석
