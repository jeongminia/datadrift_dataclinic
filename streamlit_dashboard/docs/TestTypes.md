# Test Types

대표적인 테스트 종류를 정리하면 다음과 같음

## 임베딩 특성에 맞는 드리프트 탐지 적용

---

| **방법** | **유형** | **특징** |
| --- | --- | --- |
| **MMD** [1][2][3] </br>Maximum Mean Discrepancy | 비모수적 방법, Higher-order measures | 두 데이터셋의 평균 임베딩 간의 차이를 커널 기반으로 계산해 고차원에서도 분포 간 차이를 효과적으로 측정 |
| **LSDD** [3] </br> Least-Squares Density Difference | 비모수적 방법, Higher-order measures | 두 데이터 분포 간 밀도 차이를 최소 제곱 방식으로 계산해 고차원 데이터에서도 효율적 |
| **KL Divergence** [1] </br> Kullback-Leibler Divergence | 분포 비교, Information-Theoretic measures | 두 분포 간의 상대 엔트로피를 계산하해 한 분포가 다른 분포에 대해 얼마나 다른 지에 대한 차이를 측정 |
| **JS Divergence** [1] </br> Jensen-Shannon Divergence | 분포 비교, Information-Theoretic measures | KL Divergence의 대칭적 확장하며 두 분포 간의 차이를 0~1 사이로 정규화하여 계산 |
| **Wasserstein Distance** [4] </br> Earth Mover’s Distance | 비모수적 방법, Geometric measure | 두 확률 분포의 누적 분포 함수(CDF) 간의 최소 이동 거리 계산 |
| **Energy Distance** [3] | 비모수적 방법, Geometric measure | 두 데이터셋 간의 평균 쌍별 거리와 각 데이터셋 내부의 평균 쌍별 거리 차이를 계산 |

### MMD **Maximum Mean Discrepancy**

RKHS 인 $F$에서 분포 $p$, $q$ 의 평균 차이를 측정해 고차원 임베딩 벡터에 사용

$$
\text{MMD}(F, p, q) = \|\mu_p - \mu_q\|_F^2
$$

- $\mu_p, \mu_q$ : 분포 $p, q$의 평균 임베딩
- $\|\mu_p - \mu_q\|_F^2$ : $F$ 공간에서 두 평균 임베딩 간의 제곱 거리
    - RBF 커널을 사용해 MMD 가 두 데이터 분포 간의 비선형적 차이까지 포착
    - 고차원으로 확장해 데이터의 특성과 관계를 더 잘 드러냄

### LSDD Least-Squares Density Difference

두 데이터 분포 간의 차이를 측정하기 위한 비모수적 방법으로 밀도 분포 차이를 계산하는데 밀도 분포 차이를 최소제곱법을 통해 평가

- 분포 간의 밀도 차이의 제곱의 평균을 최소화
- 커널 밀도 추정을 사용해 고차원 데이터에서도 효율적으로 계산

$$
\text{LSDD} = \frac{1}{2} \int (\Delta(x))^2 \, dx = \frac{1}{2} \int \left( p(x) - q(x) \right)^2 \, dx
$$

- 직접적인 분포 추정 대신 밀도 차이 $Δ(x)$를 모델링

### **KL Divergence** Kullback-Leibler Divergence

두 확률 분포 P, Q가 있을 때 Q분포를 사용해 P분포를 얼마나 잘 설명할 수 있는 지를 측정하며 분포 P와 Q가 다를수록 교차 엔트로피 값이 커짐

- 값이 커질수록 두 분포의 차이가 크다는 것을 의미하며, 이를 통해 드리프트 정도를 확인 가능
- 이산 분포

    $$
    D_{KL}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}    
    $$
    
- 연속 분포
    
    $$
    D_{KL}(P \| Q) = \int P(x) \log \frac{P(x)}{Q(x)} \, dx
    $$
    
- KL-발산의 값이 어느 정도 이상이 될 때 드리프트가 발생했다고 판단하는 임계값을 설정할 수 있어, 민감하게 드리프트를 모니터링 가능
- 비대칭적이기 때문에 한쪽 분포를 기준으로 다른 쪽 분포의 변화 정도를 파악 가능

### **JS Divergence J**ensen-**S**hannon Divergence

KL Divergence의 대칭적 변형으로, 두 분포 $P$와 $Q$ 간의 유사성을 측정하며 두 분포 간의 혼합 분포인 $M$를 사용해 각각의 KL Divergence를 계산

$$
JS(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M)
$$

- $M = \frac{1}{2}(P + Q)$ 는 두 분포의 평균
- *KL Divergence*보다 안정적이며, 분포가 서로 다를수록 값이 증가하며 비대칭적이라는 단점을 보완한 형태로 *P*와 *Q*를 바꿔도 동일한 결과를 제공
- 생성 모델의 출력과 실제 분포의 유사성을 비교할 때 유용

### Wasserstein Distance

두 확률 분포 P와 Q 간의 거리를 측정해, 분포 P에서 Q로 변환할 때 얼마나 많은 질량을 얼마의 거리만큼 옮겨야 하는 지에 대한 수치

$$
\mathcal{W}_p(P, Q) = \left( \inf_{\gamma \in \Pi(P, Q)} \int_{\mathbb{R} \times \mathbb{R}} |x - y|^p \, d\gamma(x, y) \right)^{\frac{1}{p}}
$$

- 분포 간의 "재배치 비용"을 거리로 계산하여, 분포 간의 차이를 직관적으로 이해할 수 있음
- *P*와 *Q*를 바꿔도 동일한 결과를 제공하며 학습된 분포와 실제 분포 간의 유사성 평가할 때 사용

### **Energy Distance**

비모수적 방법으로 분포 간의 유사성을 정량화하기 위해 사용되며 에너지 차이를 기반으로 계산

$$
D_E(P, Q) = 2 \mathbb{E}_{X \sim P, Y \sim Q} [\|X - Y\|] - \mathbb{E}_{X, X' \sim P} [\|X - X'\|] - \mathbb{E}_{Y, Y' \sim Q} [\|Y - Y'\|]
$$

- 두 분포간의 평균 거리를 바탕으로 정의
- 세가지 거리 계산(두 분포 간의 평균 거리, reference 내의 평균 거리, current 내의 평균 거리)을 결합

----

### Reference

[1] Domain Divergences: A Survey and Empirical Analysis

[2] Equivalence of Distance-Based and RKHS Based Statistics in Hypothesis Testing

[3] Drift Detection in Text Data with Document Embeddings

[4] Open-Source Drift Detection Tools in Action: Insights from Two Use Cases