
> **✔️ 진행사항 (1/24)**
> 
> 

</br>

## Key Features

| [Data Load](pages/data_load.py) | [Base Visualization](pages/base_visualization.py) | [Embedding Visualization](pages/embedding_visualization.py) | [Detect DataDrift](pages/detect_datadrift.py) |
| ---------- | ---------- | ---------- | ---------- |
|     데이터 로드       |     베이스 시각화      |      임베딩 시각화      |    데이터 드리프트 탐지        |
|      ![alt text](img_files/image.png)      |  ![alt text](img_files/image-1.png)      |     ![alt text](img_files/image-2.png)       | ![alt text](img_files/image-3.png)     |
|  **Dataset**{Preview, Description, Information} | **Class Column Analysis**{Distribution Plot}, </br> **Text Column Analysis**{Length Plot, Length Dataframe, WordCloud}      |     **Original Dimension**{Cosine Similarity, Euclidean Distances}, <br> **Dimension Reduction**{Cosine Similarity, Euclidean Distances, 2D Scatter Plot, 2D Density Plot, 3D Scatter Plot}      | **Evidently AI report**{based MMD}          |

## Documentation
데이터 드리프트 탐지와 관련해 아래 문서를 참고
- [Definition of Data Drift](docs/DataDrift.md)
- [Open-Sourece Tools for Detecting Data Drfit](docs/DriftDetection-Tools.md)
- [Test Type for Detecting Data Drift](docs/TestTypes.md)
- [Dimesionality Reduction for Effective Detecting Drift](docs/Dimensionality-Reduction.md)

## Usage Instructions
1. streamlit 접속
    
    ```
    streamlit run main.py
    ```

2. [데이터셋 업로드](pages/upload_data.py)
3. 데이터 로드에서 결측치 유무 판별 
    - 결측치 없어야 진행 가능, 데이터의 크기가 너무 크면 진행 불가
4. 시각화, 임베딩, 데이터 드리프트 탐지 진행
5. 사이드 바를 통해 각 페이지 확인
6. streamlit 중단

    - (window) `ctrl` + `c`
    - (mac) `pkill -f streamlit`

## Install Prerequisite
- fonts : [나눔고딕 레귤러](https://fonts.google.com/selection)
- library : [requirements](requirements.txt)
- open source : [EvidentlyAI](https://github.com/evidentlyai/evidently/tree/main/examples/integrations/streamlit_dashboard)
- stemming for wordcloud : [Pecab](https://github.com/hyunwoongko/pecab)
- datasets : 법률 관련 문서 데이터셋 (한림대학교 연구팀 구축)

## Directory Structure

```
streamlit_dashboard/
├── pages/
│   ├── upload_data.py
│   ├── data_load.py
│   ├── basic_visualization.py
│   ├── embedding_visualization.py
│   ├── detect_datadrift.py
├── main.py
├── utils.py
├── fonts/
│   ├── NanumGothic.ttf
├── README.md
├── docs/
│   ├── DataDrift.md
│   ├── DriftDetection-Tools.md
│   ├── TestTypes.md
├───├── 
```


## Stacks

<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white"> <img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=for-the-badge&logo=Pytorch&logoColor=white"> <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=HuggingFace&logoColor=white"> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">