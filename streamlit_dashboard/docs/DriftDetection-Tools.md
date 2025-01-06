# **Open-Source Drift Detection Tools**

---

머신러닝 모델 성능 저하 원인인 데이터 드리프트 탐지를 위해 세가지 오픈소스인 `evidentlyai`,  `nannyml`,  `alibi-detect` 를 알아보고 비교

## **1. [EvidentlyAI](https://github.com/evidentlyai/evidently)**

> 데이터 드리프트 감지와 ML 품질 검토, 모니터링을 포함한 다양한 기능을 갖춘 도구
>

## **2. [NannyML](https://github.com/NannyML/nannyml)**

> 모델 성능 평가와 데이터 드리프트 감지, 그리고 예측 정확성에 미치는 영향을 평가하는 데에 강점
> 

## **3. [Alibi-Detect](https://github.com/SeldonIO/alibi-detect)**

> 이상치, 데이터 드리프트, 적대적 공격 탐지를 위해 설계 되었으며 TensorFlow, PyTorch 등 다양한 백엔드와 호환되는 도구
> 

## 요약

---

* 각 공식 문서 참고

|  | 탐지 범위 | 강점 | 특징 |
| --- | --- | --- | --- |
| **EvidentlyAI** | Feature Drift | 일반적인 데이터 드리프트 감지 | 초기 분석 단계(시각화 가능) |
| **NannyML** | Feature Drift, Concept Drift | 변화의 정확한 시기를 파악 | 모델 성능의 사후 평가(production 환경에 유용) |
| **Alibi**-**Detect** | Covariate Drift, Concept Drift, Label Drift | 고차원 데이터(image, text) | 미세 변화 탐지 |

### 드리프트 탐지 이후,

- 데이터 드리프트만을 모니터링하는 것은 불완전
- 성능 추정 지표를 함께 추적해 모델 성능에 대한 완전한 이해가 가능