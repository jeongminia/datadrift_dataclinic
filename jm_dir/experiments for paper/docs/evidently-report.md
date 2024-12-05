# evidently report

## Report

![스크린샷 2024-12-04 오후 3.18.47.png](evidently%20report%2015286814722780d79ea5dbb8a9daf26b/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-12-04_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_3.18.47.png)

```python
from evidently.metrics import EmbeddingsDriftMetric # Reports or Test Suites를 생성 ⇒ EmbeddingsDriftMetric or TestEmbeddingsDrift
from evidently.report import Report
```
</br>

> 💬 Metric에 따라 달라지는 시각화

- 분포 간 차이를 시각적으로 표현할 때도 사용된 메트릭의 특성이 반영
- 데이터를 비교하는 방식에 영향을 주기 때문에 시각화도 달라짐
    
    ```python
    def calculate(self, data: InputData) -> EmbeddingsDriftMetricResults:
            if data.reference_data is None:
                raise ValueError("Reference dataset should be present")
            drift_method = self.drift_method or model(bootstrap=data.reference_data.shape[0] < 1000)
            emb_dict = data.data_definition.embeddings
            if emb_dict is None:
                raise ValueError("Embeddings should be defined in column mapping")
            if self.embeddings_name not in emb_dict.keys():
                raise ValueError(f"{self.embeddings_name} not in column_mapping.embeddings")
            emb_list = emb_dict[self.embeddings_name]
            drift_score, drift_detected, method_name = drift_method(
                data.current_data[emb_list], data.reference_data[emb_list]
            )
            # visualisation
            ref_sample_size = min(SAMPLE_CONSTANT, data.reference_data.shape[0])
            curr_sample_size = min(SAMPLE_CONSTANT, data.current_data.shape[0])
            ref_sample = data.reference_data[emb_list].sample(ref_sample_size, random_state=24)
            curr_sample = data.current_data[emb_list].sample(curr_sample_size, random_state=24)
            data_2d = TSNE(n_components=2).fit_transform(pd.concat([ref_sample, curr_sample]))
            reference, _, _ = get_gaussian_kde(data_2d[:ref_sample_size, 0], data_2d[:ref_sample_size, 1])
            current, _, _ = get_gaussian_kde(data_2d[ref_sample_size:, 0], data_2d[ref_sample_size:, 1])
    
            return EmbeddingsDriftMetricResults(
                embeddings_name=self.embeddings_name,
                drift_score=drift_score,
                drift_detected=drift_detected,
                method_name=method_name,
                reference=reference,
                current=current,
            )
    ```
    
    - 선택된 drift_method에 따라 분포 비교 방식이 달라지고 계산된 **`drift_score`와 `drift_detected`** 값이 시각화에 반영

</br>

> 💬 valid vs train // test vs train 에 따라 달라지는 시각화

- 해당 클래스를 분석한 결과,
    - `SAMPLE_CONSTANT` : 데이터의 크기(`curr_sample_size`)가 달라지면서 추출됨
    - `TSNE(n_components=2).fit_transform(pd.concat([ref_sample, curr_sample]))` :
        
        비교 대상 데이터(valid 또는 test)에 따라 공간을 재배치
        
    - `get_gaussian_kde` : 커널 밀도 추정(KDE)을 수행하기 때문에, 차원 축소된 좌표가 달라짐

</br>

## Compare

임베딩 데이터 기반의 데이터 드리프트 탐지

- 기준 데이터와 현재 데이터를 구분하는 모델을 학습시켜 두 분포 간의 차이를 평가
- 분류기의 성능은 Metric 지표로 측정

| **drift_method** | **설명** | Option |
| --- | --- | --- |
| **`model`** | • 이진 분류 모델을 사용해 current와 reference 분포 간 임베딩을 구분 </br> • ROC AUC를 drift_score로 반환 | ROC AUC |
| **`ratio`** | • 개별 임베딩 구성 요소 간 분포 드리프트를 계산 </br> • 모든 tabular numerical 드리프트 탐지 방법 사용 가능 </br> • drift_score로 드리프트된 임베딩의 비율 반환 | `evidently.calculations.stattests` Wasserstein Distance, Kullback–Leibler Divergence, Jensen Shannon Divergence, Energy Distance | 
| **`distance`** | • current와 reference 데이터셋 간 평균 임베딩 거리 계산 </br> • distance 값을 drift_score로 반환 | `scipy.spatial.distance` Euclidean, Cosine, Cityblock, Chebyshev |
| **`mmd`** | • Maximum Mean Discrepancy MMD 를 계산 </br> • MMD값을 drift_score로 반환 | `evidently.metrics.data_drift.embedding_drift_methods` |

</br>

## 1. `ratio`

### 1.1 code flow

> drift_score 로 드리프트된 임베딩의 비율 반환
> 
> - path : evidently/src/evidently/metrics/data_drift/embedding_drift_methods.py

개별 차원 비교

```python
stattest_func = get_stattest(
            reference_emb.iloc[:, 0], current_emb.iloc[:, 0], ColumnType.Numerical, self.component_stattest
        )
```

드리프트 비율이 전체 임계값을 초과하는 지 여부

```python
n_drifted / reference_emb.shape[1], # 드리프트가 감지된 차원의 비율
n_drifted / reference_emb.shape[1] > self.threshold, # 드리프트 비율이 전체 임계값을 초과하는지 여부
"ratio"
```

</br>

### 1.2  metric code & **mathematical expression**

> 임베딩 데이터 즉, 수치형 데이터를 아래 메트릭을 통해 개별 임베딩 구성 요소 간 분포 계산
> 
> - path : evidently/src/evidently/calculations/stattests

| Metric                           | Mathematical Expression                                                                                                                                                                       | Feature Description              | Functionality/Definition                                |
|----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------|--------------------------------------------------------|
| **[Wasserstein Distance](https://en.wikipedia.org/wiki/Wasserstein_metric)** | \(\mathcal{W}_p(P, Q) = \left( \inf_{\gamma \in \Pi(P, Q)} \int_{\mathbb{R} \times \mathbb{R}} \|x - y\|^p \, d\gamma(x, y) \right)^{\frac{1}{p}}\)                                                | 물리적 이동 | `norm` → `stats.wasserstein_distance`                 |
| **[Kullback–Leibler Divergence](https://ko.wikipedia.org/wiki/%EC%BF%A8%EB%B0%B1-%EB%9D%BC%EC%9D%B4%EB%B8%94%EB%9F%AC_%EB%B0%9C%EC%82%B0)** | \(D_{\text{KL}}(P \| Q) = -\int_{X} \log{\frac{dQ}{dP}} \, dP\)                                                               | 분포의 정보 비교 | `get_binned_data` → `stats.entropy`                 |
| **[Jensen–Shannon Divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)** | \(JS(P \| Q) = \frac{1}{2} D_{\text{KL}}(P \| M) + \frac{1}{2} D_{\text{KL}}(Q \| M), \text{where } M = \frac{P + Q}{2}\)                                                | 대칭성 비교 | `get_binned_data` → `distance.jensenshannon`        |
| **[Energy Distance](https://en.wikipedia.org/wiki/Energy_distance)**       | \(E[(X, X') \sim P] \|X - X'\| + 2 E[(X, Y) \sim (P, Q)] \|X - Y\| - E[(Y, Y') \sim Q] \|Y - Y'\|\)                                                                                            | 거리 기반 차이 | `stats.energy_distance`                             |




- get_binned_data
    
    분포 비교를 위해 데이터를 여러 구간으로 나눠 각 구간에 속한 데이터 비율 계산
    
- stats, distance
    
    scipy 라이브러리에서 import 

</br>

## 2. **`mmd`**
    

> Maximum Mean Discrepancy MMD 값을 drift score 값으로 반환
> 
> - path : evidently/src/evidently/metrics/data_drift/embedding_drift_methods.py

</br>

### 2.1 **code flow**

**주요 파라미터**

- `threshold`: 드리프트를 판단할 기준값으로 $mmd2u>threshold$ 이면 드리프트로 판단 (기본값: 0.015)
- `bootstrap`: 부트스트랩 기반 통계 검정을 적용할지 여부 (기본값: `None`)
- `quantile_probability`: 부트스트랩 시 사용되는 퍼센타일 (기본값: 0.05)
- `pca_components`: PCA를 통해 차원을 축소할 경우 유지할 컴포넌트 수

```python
def mmd(
    threshold: float = 0.015,
    bootstrap: Optional[bool] = None,
    quantile_probability: float = 0.05,
    pca_components: Optional[int] = None,
) -> DriftMethod:
    """Returns a function for calculating drift on embeddings using the mmd method with specified parameters
    Args:
        threshold: all values above this threshold means data drift. Applies when bootstrap != True
        bootstrap: boolean parameter to determine whether to apply statistical hypothesis testing
        quantile_probability: applies when bootstrap == True
        pca_components: number of components to keep
    Returns:
        func: a function for calculating drift, which takes in reference and current embeddings data
        and returns a tuple: drift score, whether there is drift, and the name of the drift calculation method.
    """
    return MMDDriftMethod(
        threshold=threshold,
        bootstrap=bootstrap,
        quantile_probability=quantile_probability,
        pca_components=pca_components,
    )
```

</br>

### 2.1 metric code & **mathematical expression**

|  | mathematical expression | feature |
| --- | --- | --- |
| [Maximum Mean Discrepancy](https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions) | $\text{MMD}^2(P, Q) = \mathbb{E}_{x, x' \sim P} [k(x, x')] + \mathbb{E}_{y, y' \sim Q} [k(y, y')] - 2 \mathbb{E}_{x \sim P, y \sim Q} [k(x, y)]$ | 평균과 모양 비교 |
- k
    
    두 데이터 분포의 차이를 커널 함수 값으로 계산 ⇒ 커널 기반으로 하여 두 분포 간의 차이를 유연하게 감지하고 데이터의 특징 공간에서 유사성을 비교
    
</br>

1. reference, current 크기에 맞춰 샘플의 크기 결정
    
    ```python
    x = reference_emb
    y = current_emb
    m = len(x)
    n = len(y)
    ```
    
2. 유클리드 거리 계산 -> RBF커널의 스케일 파라미터 추정
    
    ```python
    pair_dists = pairwise_distances(
    						x.sample(min(m, 1000), random_state=0),
                y.sample(min(n, 1000), random_state=0),
                metric="euclidean",
                n_jobs=-1,
    		        )
    		        
    sigma2 = np.median(pair_dists) ** 2
    xy = np.vstack([x, y])
    
    # RBF 커널 행렬 생성
    K = pairwise_kernels(xy, metric="rbf", gamma=1.0 / sigma2) 
    ```
    
3. MMD 점수 계산
    
    ```python
    mmd2u = MMD2u(K, m, n)
    ```
    
4. drift 결과 반환
    
    ```python
    # MMD 점수를 설정된 임계값과 비교하여 드리프트 여부를 반환
    max(mmd2u, 0), mmd2u > self.threshold, "mmd"
    ```