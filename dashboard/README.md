## System Flow

- [**app_database/**](./app_database)  : Upload Data → Load Data → Text Visualization → Embedding → Store in DB → Export Report
    
    텍스트 데이터 업로드하여 벡터DB에 저장
    
- [**app_drift/**](./app_drift)  : Load Embeddings → Embeddings Visualization → Detect Drift → Export Report
    
    벡터DB에서 즉시 불러와 시각화해 드리프트 감지


## Documentation
데이터 드리프트 탐지와 관련해 아래 문서를 참고
- [Definition of Data Drift](docs/DataDrift.md)
- [Open-Sourece Tools for Detecting Data Drfit](docs/DriftDetection-Tools.md)
- [Test Type for Detecting Data Drift](docs/TestTypes.md)
- [Dimesionality Reduction for Effective Detecting Drift](docs/Dimensionality-Reduction.md)
- [Text DataDrift](docs/Text-DataDrift.md)

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
dashboard/
├── app_database/
│   ├── pages/
│   │   ├── base_visualization.py
│   │   ├── data_load.py
│   │   ├── export_report.py
│   │   ├── llm_explainer.py
│   │   ├── upload_data.py
│   │   ├── vector_database.py
│   ├── reports/
│   ├── main.py               ◀ app_database 메인 실행 파일
│   ├── utils.py              
│
├── app_drift/
│   ├── pages/
│   │   ├── detect_datadrift.py
│   │   ├── detect_propertydrift.py
│   │   ├── embedding_load.py
│   │   ├── embedding_visualization.py
│   ├── reports/
│   ├── main.py               ◀ app_drift 메인 실행 파일
│   ├── utils.py              
│
├── docs/                     ◀ 기술 문서 정리
├── fonts/                   
├── img_files/               
├── reports/                 
├── inspect-collections.py    ◀ Vector DB 저장 확인
├── README.md
├── requirements.txt          ◀ Streamlit 실행에 필요한 패키지 모음
├── rm-collections.py         ◀ Vector DB 일부 데이터 제거            
```


## Stacks

<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white"> <img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=for-the-badge&logo=Pytorch&logoColor=white"> <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=HuggingFace&logoColor=white"> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/Milvus-00A1EA?style=for-the-badge&logo=Milvus&logoColor=white">