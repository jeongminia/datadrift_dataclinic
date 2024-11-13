# README

---

## Prerequisite

---

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

## **Alibi Detect**

---

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

---

**NLP에서의 고차원 특성과 판별적 모델의 특성을 고려한 드리프트 감지 방법을 실험 시행**
 

</br>

### $KS$ $Test^{[1]}$
> **Kolmogorov-Smirnov test**
    두 데이터 분포의 누적분포함수 CDF 차이를 계산해 가장 큰 차이를 통한 비모수적 검정으로 특정 차원별 분포 비교
    

</br>

### $LSDD^{[2]}$
> **Least-Squares Density Difference**
    두 데이터 분포 간의 밀도 차이를 최소제곱법으로 계산해 전체 임베딩 벡터 비교에 유용

</br>

### $MMD^{[2]}$
> **Maximum Mean Discrepancy**
    Reproducing Kernel Hillbert Space 상에서 두 분포의 평균 차이를 측정해 고차원 임베딩 벡터에 사용

</br>



## Refernece

---

- **websites**
    
    [1] [Alibi Detect official docs](https://docs.seldon.io/projects/alibi-detect/en/stable/index.html)
    
- **Paper**
    
    [2] [Drift Detection in Text Data with Document Embeddings (2021)](https://github.com/EML4U/Drift-detector-comparison?tab=readme-ov-file)
    
    - [paper review](https://ajmajm2024.notion.site/Drift-Detection-in-Text-Data-with-Document-Embeddings-2021-13b868147227800dbf37d38da0e2b6a1)