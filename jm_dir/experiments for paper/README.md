# README

> **📌 Initial settings**
>    - 분포 비교를 더 명확히 하고 싶기 때문에 각 데이터셋을 **병합**한 뒤 차원 축소를 적용해 시각화
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


## 1. DataDrift without Dimension Reduction

---

> 차원 축소 없이 데이트 드리프트 탐지 진행
> 
> - `EvidentlyAI`
>     - MMD, Wasserstein Distance, KL Divergence, JS Divergence, Energy Distance
> - `Alibi-Detect`
>     - LSDD, KDE

|  | Opensource | feature | `valid` vs `train` | `test` vs **`train`** |
| --- | --- | --- | --- | --- |
| **Maximum Mean Discrepancy** | EvidentlyAI | 전체 데이터 분포를 직접 비교하는 고차원 분포 간 유사성 측정 | 0 | 0.209 |
| **Wasserstein Distance** | EvidentlyAI | 분포 간의 이동 거리 계산, 분포 형태보다는 거리 비용에 초점 | 0.025 | 0.964 |
| **Kullback–Leibler Divergence** | EvidentlyAI | 한 분포가 다른 분포와 얼마나 차이나는지 비대칭적으로 측정 | 0 | 0.697 |
| **JS Divergence** | EvidentlyAI | KL Divergence를 대칭적으로 변환, 두 분포 간 차이를 직관적으로 이해 | 0 | 0.897 |
| **Energy Distacne** | EvidentlyAI | 거리 기반 접근으로 중심 및 분산 차이를 동시에 고려 | 0 | 0.474 |
| **Latent Space Density Difference** | Alibi Detect | 잠재 공간에서 국소적인 밀도 차이를 기반으로 분포 간 드리프트 탐지 | 0.00097 | 0.191 |
| **KDE-Based Drift Detection** | Alibi Detect | 커널 밀도 추정을 사용해 밀도 차이 기반으로 드리프트 탐지 | 0 | 0.206 |

 * drift score : 선택된 메트릭을 통해 Reference Data와 Current Data의 분포 차이를 수치화한 값

</br>

[EvidentlyAI](https://docs.evidentlyai.com/) 는 Report를 제공하고 있어 아래와 같은 시각화 결과를 도출할 수 있음

|  | `valid` vs `train` | `test` vs `train` |
| --- | --- | --- |
| **MMD** | ![**valid** vs **train**](img_files/image.png) | ![**test** vs **train**](img_files/image%201.png) |
| **Wasserstein Distance** |![**valid** vs **train**](img_files/image%202.png) | ![**test** vs **train**](img_files/image%203.png) |
| **KL Divergence** | ![**valid** vs **train**](img_files/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-11-29_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_4.08.06.png) |![**test** vs **train**](img_files/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-11-29_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_4.10.00.png)  |
| **JS Divergence** | ![**valid** vs **train**](img_files/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-11-29_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_3.58.50.png) | ![**test** vs **train**](img_files/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-11-29_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_3.59.49.png) |
| **Energy Distacne** | ![**valid** vs **train**](img_files/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-11-29_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_4.28.50.png) | ![**test** vs **train**](img_files/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-11-29_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_4.29.14.png) |



## 2. Methods for Dimension Reduction

---

> **dimension reduction order**
> 
> 1. train에 대해서만 먼저 적합 진행
>     
>     train 기준으로 차원 축소를 진행한 뒤, 다른 데이터로 같은 공간에 투영해 변화 감지
>     
>     → 실제 시스템에서 배포된 모델이 학습한 데이터와 새로 들어온 데이터 간의 변화를 감지하는 데 효과적
>     
> 2. 병합한 뒤 pca 적용 → **KS test**를 적용하려면 동일한 기준에서 두 분포를 비교해야 신뢰가능
> 3. 각각 pca 적용

### 2.1 Select Dim

적절한 차원 선택을 위해 데이터의 구조적 변화를 감지하기 위해 평가방법에 따라 다른 메트릭을 선택

|  | Reconstruction Error | Pairwise Distance Preservation | Total Variation Distance |
| --- | --- | --- | --- |
| **PCA** | ✅ | ✅ | ❌ |
| **Kernel PCA** | ✅ | ✅ | ❌ |
| **UMAP** | ❌ | ✅ | ✅ |
| **t-SNE** | ❌ | ✅ | ✅ |
| **SVD** | ❌  | ✅  | ❌ |
| **GRP** | ❌ | ✅ | ✅ |
| **Autoencoders** | ✅ | ❌ | ❌ |

#### 2.1.1 PCA

Explained Variance Ratio 분산 변동 비율을 기준으로 적합한 `n_components` 값 확인

![image.png](img_files/image%204.png)

이때, Train의 95% 분산 변동비율을 기준으로 `n_components = 186` 선택

| **Dataset** | **99% Variance Components** | **95% Variance Components** |
| --- | --- | --- |
| Train | 342 | 186 |
| Validation | 271 | 147 |
| Test | 284 | 144 |

#### 2.1.2 Kernel PCA

Reconstruction Error을 기준으로 적합한 `n_components` 값 확인

![image.png](img_files/image%205.png)

이때, Train 기준으로 기울기가 급격히 완화되는 지점인 Elbow point를 도출했고 이는 `n_components = 77` 선택

![image.png](img_files/image%206.png)

#### 2.1.3 UMAP

Pairwise Distance Preservation PDP를 기준으로 적합한 `n_components` 값 확인

![image.png](img_files/image%207.png)

#### 2.1.4 t-SNE

#### 2.1.5 SVD

Explained Variance Ratio 분산 변동 비율을 기준으로 적합한 `n_components` 값 확인

![image.png](img_files/image%208.png)

이때, Train의 95% 분산 변동비율을 기준으로 `n_components = 190` 선택

| **Dataset** | **99% Variance Components** | **95% Variance Components** |
| --- | --- | --- |
| Train | 345 | 190 |
| Validation | 274 | 150 |
| Test | 300 | 163 |

#### 2.1.6 GRP

![image.png](img_files/image%209.png)

| **Dataset** | **Elbow Point Components** |
| --- | --- |
| Train | 65 |
| Validation | 67 |
| Test | 62 |

#### 2.1.7 Autoencoder

Reconstruction Error으로 계산한 결과, train 기준으로 `n_components=328` 선택

![image.png](img_files/image%2010.png)

### 2.2 Visualization of Dimension Reduction Results

데이터 드리프트 탐지 목적이기에, 차원 축소를 **train-valid**와 **train-test**에서 각각 따로 진행

#### 2.2.1 Distance

|  | dim | `valid` vs `train` | `test` vs **`train`** |
| --- | --- | --- | --- |
| PCA | 186 | ![alt text](img_files/image-27.png) | ![alt text](img_files/image-25.png) |
| Kernel PCA | 77 | ![alt text](img_files/image-23.png) | ![alt text](img_files/image-21.png) |
| UMAP | 20 | ![alt text](img_files/image-19.png) | ![alt text](img_files/image-17.png) |
| t-SNE | 3 | ![alt text](img_files/image-15.png) | ![alt text](img_files/image-13.png) |
| SVD | 190 | ![alt text](img_files/image-11.png) | ![alt text](img_files/image-9.png) |
| GRP | 65 | ![alt text](img_files/image-7.png) | ![alt text](img_files/image-5.png) |
| AutoEncoder | 328 | ![alt text](img_files/image-3.png) | ![alt text](img_files/image-1.png) |


#### 2.2.2 Plot

|  | dim | `valid` vs `train` | `test` vs **`train`** |
| --- | --- | --- | --- |
| PCA | 186 | ![alt text](img_files/image-26.png) | ![alt text](img_files/image-24.png) |
| Kernel PCA | 77 | ![alt text](img_files/image-22.png) | ![alt text](img_files/image-20.png) |
| UMAP | 20 | ![alt text](img_files/image-18.png) | ![alt text](img_files/image-16.png) |
| t-SNE | 3 | ![alt text](img_files/image-14.png) | ![alt text](img_files/image-12.png) |
| SVD | 190 | ![alt text](img_files/image-10.png) | ![alt text](img_files/image-8.png) |
| GRP | 65 | ![alt text](img_files/image-6.png) | ![alt text](img_files/image-4.png) |
| AutoEncoder | 328 | ![alt text](img_files/image-2.png) | ![alt text](img_files/image-28.png) |

#### 2.2.3 Evaluation
**Pairwise Distance Preservation**

- 차원 축소 전후 데이터 간의 쌍별 거리 유지율을 계산하며 평균 변화율로 평가

| Pairwise Distance Preservation | PCA | Kernel PCA | UMAP | t-SNE | SVD | GRP | AutoEncoder |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **dim** | 186 | 77 | 20 | 3 | 190 | 65 | 328 |
| **times** | 0.2s | 0.7s | 12.3s | 10.9s | 0.2s | 0.0s | 3.5s |
| **train** | 0.9983 | 0.9807 | 0.3515 | 0.419 | 0.9984 | 0.8871 | 0.556 |
| **valid** | 0.997 | 0.9771 | 0.3396 | 0.3107 | 0.9972 | 0.9021 | 0.6188 |
| **test**  | 0.9935 | 0.9651 | 0.2564 | 0.4457 | 0.9939 | 0.9013 | 0.4337 |


## 3. Dimension Reduction through Ensemble Methods

---

## 4. DataDrift with Dimension Reduction

---