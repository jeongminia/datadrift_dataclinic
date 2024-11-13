# README



## Prerequisite



| **Requirement** | **Version** |
| --- | --- |
| Alibi-Detect | 0.12.1.dev0 |
|  |  |

</br>

### Install Alibi-Detect

```bash
git clone https://github.com/SeldonIO/alibi-detect.git
cd alibi-detect
pip install -e .
```

</br>

## [Alibi Detect](https://github.com/SeldonIO/alibi-detect)



<div style="border-left: 4px solid #f39c12; padding: 20px; background-color: #fdf2e9;">
<strong>💡 Alibi Detect </strong> </br> 이상치 탐지, 적대적 공격 탐지, 드리프트 탐지에 초점을 맞춘 오픈소스 python 라이브러리로 테이블형 데이터, 텍스트, 이미지, 시계열 데이터에 대해 온라인 및 오프라인 탐지기를 모두 지원
</div>
</br>

**장점**

- 텍스트 및 임베딩 데이터에 강점
- 유연한 확장성
</br>

**다양한 Drift Detection 제공**

- _<ins>가설 검정<ins>_
    특정 통계치 기준으로 두 분포가 같은지 여부를 평가하며 유의미한 차이가 있는 경우, Drift 발생

- _<ins>Univariate Tests<ins>_
    
    특성별로 독립적인 단변량 검정 수행
    
- _<ins>Multivariate Tests<ins>_
    
    다변량 검정을 통해 모든 특성을 고려해 드리프트 감지
</br>

**예측 결과**
- 감지된 드리프트 여부와 p-value를 출력
- H0는 드리프트가 없음을 가정한 데이터, H1는 드리프트가 발생했을 가능성이 높은 데이터

</br>

## **Methods**



**NLP에서의 고차원 특성과 판별적 모델의 특성을 고려한 드리프트 감지 방법을 실험 시행**
 

</br>

### $KS$ $Test^{[1]}$
> **Kolmogorov-Smirnov test**  
    두 데이터 분포의 누적분포함수 CDF 차이를 계산해 가장 큰 차이를 통한 비모수적 검정으로 특정 차원별 분포 비교

</br>

**Full Flow** 

1. feature별로 reference data와 current data를 비교해 각 feature가 같은 분포를 가지는 지 여부를 독립적으로 확인

2. 이후, 다중테스트보정을 통해 전체 데이터 드리프트 여부 판단

</br>

**KS Test** 

KS 통계량 D가 임계값보다 크면 두 분포가 유의미하게 다름

$$
D = \sup_x |F(x) - G(x)|
$$

- $F(x), G(x)$ : 두 데이터 샘플의 누적분포함수 CDF
- $sup_x$ : $x$에 대해 두 CDF 간의 최대 차이 계산


</br>

**⏩ multiple test correction**
고차원 데이터에서는 모든 피처에 대해 개별적 테스를 수행해 오류 확률 감소를 위해 **보정방법 correction method 적용**

- Bonferroni
    
    모든 p-value을 정렬해, 각 값을 그 순서와 연관지어 허용 가능한 q-value에 맞춰 유의성을 판단해 false positive 비율 허용
    
- FDR False Discovery Rate
    
    유의수준을 테스트 수로 나눈 값인 $\alpha/m$ 을 사용해 각 테스트의 유의수준 조정을 통해 false positive 확률을 줄임

</br>

**For high-dimensional data** 
차원 축소 후 검정을 진행하는 것을 권장

- 고차원 데이터에서 모든 feature에 대해 개별적으로 KS Test를 수행하는 것은 연산 비용이 높음
- PCA, UAE 등을 사용해 차원을 줄여, 축소된 차원에 대해 KS Test 수행

</br>

### $LSDD^{[2]}$
> **Least-Squares Density Difference**  
    두 데이터 분포 간의 밀도 차이를 최소제곱법으로 계산해 전체 임베딩 벡터 비교에 유용

</br>

**Full Flow** 

- 각 feature를 따로 테스트하지 않고 모든 피처를 고려해 데이터 간의 분포 차이 한번에 계산 
- 해당 값에 대해 permutation test 수행해 p-value 계산

→ 여러 피처 간의 상관관계와 상호작용을 반영해 전체적인 분포 차이 측정

</br>

**LSDD** 

분포 간의 밀도 차이를 나타내는 테스트 통계량 

$$
LSDD(p, q) = \int (p(x) - q(x))^2 \, dx
$$

- $p(x), q(x)$ : 두 데이터의 확률 밀도 함수
    
    → 차이의 제곱을 적분해 전체적인 차이 계산

</br>

**⏩ Permutation Test**

LSDD 통계량이 단순한 우연으로 발생한 것인지 아니면 실제로 두 분포가 다르기에 발생한 것인지를 검정하기 위해 p-value 계산

- reference, current data의 LSDD를 계산해 초기 통계량 계산 → $LSDD_0$
- reference, current data를 섞어 랜덤하게 그룹을 나눠 이 과정을 여러 번 반복
- 각 무작위로 분할된 그룹에서 LSDD 값을 계산해 여러 개의 값을 얻음
- $LSDD_0$ 이 랜덤하게 발생한 값들 중에서 얼마나 극단적인지를 확인하기 위해, $LSDD_0$ 보다 큰 값이  랜덤 샘플에서 나오는 비율을 p-value로 정의
    - p-value가 작다면 우연히 발생할 가능성이 낮고 실제로 두 분포가 다르기에 발생한 것으로 판단

</br>


**For high-dimensional data**

고차원 데이터에서 수치적 안정성을 유지하고 계싼 효율을 높이기 위해 차원축소 적용 후, LSDD 수행

- PCA, UAE, BBSD

</br>


### $MMD^{[2]}$
> **Maximum Mean Discrepancy**  
    Reproducing Kernel Hillbert Space 상에서 두 분포의 평균 차이를 측정해 고차원 임베딩 벡터에 사용

</br>


**Full Flow**

- 데이터를 RKHS 공간으로 변환하며, MMD를 계산할 때 커널 함수 사용
- permutation test로 p-value 계산해 두 데이터가 통계적으로 다른지 확인

</br>

**MMD** 

RKHS 인 $F$에서 분포 $p$, $q$ 간의 MMD

$$
\text{MMD}(F, p, q) = \|\mu_p - \mu_q\|_F^2
$$

- $\mu_p, \mu_q$ : 분포 $p, q$의 평균 임베딩
- $\|\mu_p - \mu_q\|_F^2$ : $F$ 공간에서 두 평균 임베딩 간의 제곱 거리
    - RBF 커널을 사용해 MMD 가 두 데이터 분포 간의 비선형적 차이까지 포착
    - 고차원으로 확장해 데이터의 특성과 관계를 더 잘 드러냄

</br>


**⏩ Permutation Test**

- LSDD 와 유사

</br>

**For high-dimensional data**

- LSDD 와 유사

</br>


## Refernece



- **websites**
    
    [1] [Alibi Detect official docs](https://docs.seldon.io/projects/alibi-detect/en/stable/index.html)
    
- **Paper**
    
    [2] [Drift Detection in Text Data with Document Embeddings (2021)](https://github.com/EML4U/Drift-detector-comparison?tab=readme-ov-file)
    
    - [paper review](https://ajmajm2024.notion.site/Drift-Detection-in-Text-Data-with-Document-Embeddings-2021-13b868147227800dbf37d38da0e2b6a1)