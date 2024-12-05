# Metrics of Dimension Reduction

---

차원축소를 평가하기 위해서 크게 두가지 지표로 구분지어 볼 수 있음

- 전역 구조 보존 : 데이터 전반의 큰 변화 확인
- 국소 구조 보존  : 이상치 감지

</br>

| Methods | Structure | Type | Feature |
| --- | --- | --- | --- |
| PCA | Linear | Global | 단순, 계산 빠름, 비선형 데이터에 약함 |
| Kernel PCA | NonLinear | Global | 커널 기반, 비선형 데이터에 유리 |
| UMAP | NonLinear | Local | 군집 표현 및 국소 구조 보존에 강점 |
| t-SNE | NonLinear | Local | 시각화 최적화, 전역 구조 왜곡 가능 |
| SVD | Linear | Global | PCA와 유사, 행렬 분해 방식 |
| GRP | Linear | Global | 빠르고 간단하지만 정확도는 낮을 수 있음 |
| AutoEncoder | NonLinear | Local + Global | 학습 기반, 복잡한 데이터 패턴 모델링 |
- 일반적인 평가 지표로는 Reconstruction Error, Explained Variance Ratio, [Silhouette Coefficient](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.silhouette_score.html) 가 있으나 전역과 국소 구조 보존을 중심으로 차원 축소 품질 지표(QMs)을 선택함

</br>

📜 각 논문에서 언급된 차원 축소 품질 지표(QMs)는 아래와 같음

| **Metrics** | **Type** | **Mathematical Expression** | **Feature** | **Paper** |
| --- | --- | --- | --- | --- |
| Stress | Global | $F = \sqrt{\frac{\sum_{j \neq l} \left( D(j, l) - \xi_{i,j} \right)^2}{\sum_{j \neq l} D(j, l)^2}}$ | 거리가 작을수록 좋음 | [**Analyzing Quality Measurements for Dimensionality Reduction**](https://www.mdpi.com/2504-4990/5/3/56), MDPI, 2023 |
| [Trustworthiness](https://scikit-learn.org/1.5/modules/generated/sklearn.manifold.trustworthiness.html) | Local | $T(k) = 1 - \frac{2}{n \cdot k (2n - 3k - 1)} \sum_{i=1}^n \sum_{j \in U_k(i)} (\text{rank}_X(i, j) - k)$ | 1에 가까울수록 잘 보존 | [**Analyzing Quality Measurements for Dimensionality Reduction**](https://www.mdpi.com/2504-4990/5/3/56), MDPI, 2023 |
| [Pearson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html) | Global | $r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2 \sum_{i=1}^n (y_i - \bar{y})^2}}$ | 거리가 클수록 잘 보존 | - |
| Local Continuity Meta-Criterion | Local | $\text{LCMC}(k) = 1 - \frac{1}{nk} \sum_{i=1}^n \left\| N_k^{\text{orig}}(i) \cap N_k^{\text{reduced}}(i) \right\| + \frac{k}{n}$ | 거리가 작을수록 좋음 | [**Visualizing the quality of dimensionality reduction**](https://pdf.sciencedirectassets.com/271597/1-s2.0-S0925231213X00104/1-s2.0-S0925231213002439/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFQaCXVzLWVhc3QtMSJIMEYCIQClyvzg36NqXgDaAbRdz%2FQchxokQBtM3QgkZRvI2fA%2BKgIhAJohrTNP5DQSeAl1HMWEJ5Z8MIyeYOBgNU1cKSMz9yJ0KrwFCPz%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1IgzBZLHvCL4Q84PbC5UqkAXFpwhBRKwzGGn%2BaZ2FtwhCPPtilcEM%2Bc2eYPv8HtSweUVWzVAP6h1WBwFgLeuuw7jFNzLdNg%2FJTP8OF0bTLhHJ7UlGdVHdZstxgozRNrsUgOfLZn4AIktZfgcWEUiF51wc5tm%2BNE%2F62x%2B0BRXvcOCHkF3txQZ835T%2BNB2%2FaKwKAHJqYD0Nn%2FDj2BiL3UTkVOo%2FF%2F4uceYcjsbSnYMbrZkOtN2EuNSpWkd4%2FG4odXkSKBz4JC6nE3Pysv%2B3BSOv0PPnS8ioWU6DnALB4OdU8d8lTupFA6xtQtmzVEZYP3%2B501UTk2%2FgkTSiyVcZ%2FEamKIqmrooGJ4q0LLJMcdPnnjoKkV1hQ4nziVDBifBFh0WjYQPXgRCUTbgXZDYd3fDjIl00TwC5abCaWVF%2BN3VI%2BTvmApY3ztCwH%2BbquagptHOyeal8QLmjDnGr3iw%2B0kRNx7Eys7XthF48DxSwr%2F%2FvotahW5hRMq75dZNH%2FxrCc%2B1MpuqylM2sDpRP0BeoyqrNM8N5%2FATmy8BibZ7ncN7bXGauk8u7c2HBge3c2xncW5%2BfOR1h1V9X7A7kO%2F49PHcG7L%2BnI6Z3xKrdV0rzvRRtjIBHsF9uWOpROYXz38ZyXJ3ERfrqnfj7yw%2FWo5LGx1Uej4pJZkFwF02RQgiz847C03ZE1KZwAZ6DKIA%2FU7g%2Fe2WSEE5hw2yt3TDV%2F8OeXOwcH2FQ1Xzh32PupMC327F%2BWfn7AjQXwuNIXoSVRyz%2FKBXFD%2BUgttJRO8NUfao12QS2aLpN7IePArGoyp97OvCM0VJs4KNWiMTV3%2B5aNTOUXv1ZtRo1QvEd%2BmLfWaHwZ7lHoo56c7VTqFJz%2FdmF2PIeUwplX2wltZOQ5UKGkIH1oRwECTDotMS6BjqwAaObOTPXB83spO4YTgyXNmfs9ck%2BpxeeKx8lyUXmCWsrdGo1zyezZSz%2BZQeFeS56OW5rRT2SN%2B1rjc4n0PWPBiQY6PMD2OXJrtVIgWSYXqbPYvU6mVuJxrsRKxBtnuEDWS4Kb5t207E4RaR0MAiPH5%2Fi%2BiN2vTrFR4xPmMp2qD55Uzp%2Bu3VSsMY8mbXWIrt04o%2FnBK43u1El72ATf%2B53VETVPE%2Bf%2FV3u1bDUyeXkeqAc&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20241205T034206Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYRIJBO53A%2F20241205%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=73983c7a1b71f9f011edabca8e6ec3c781d09ee9c8c69b8667a1e70bfc2338c8&hash=662c77270508bcecf36d07189fbbf735278e3ba80b8f6f9de85f63f1659185eb&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0925231213002439&tid=spdf-08d8601f-5090-4344-8282-f5860dcf704a&sid=6fd2502a5d4b024d7598020642f352b37af0gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=11145d0a07565f58555854&rr=8ed106371c22ea09&cc=kr), Neurocomputing, 2013 |
- **k 선택 과정** ⇒ 데이터 사이즈는 각각 4078 1935 1000으로 최종적으로 k=200으로 진행
    - 적절한 k값은 데이터 샘플의 개수와 데이터 밀도를 기반으로 설정
    - 일반적으로 전체 데이터 샘플 수의 1-5% 수준에서 설정