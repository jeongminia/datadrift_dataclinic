# datadrift_dataclinic

## 주소
- https://github.com/keti-datadrift/datadrift_dataclinic.git

## 개요
- 데이터 드리프트 관리 기술의 기반 프레임워크입니다.
- 개발 및 유지 관리 기관 : 한국전자기술연구원(KETI)
- 최종 검토 기관 : __한국전자기술연구원(KETI)__

### Full Flow
```mermaid
flowchart TD
    subgraph app_database["app_database/"]
        A1[Upload Data] --> A2[Load Data]
        A2 --> A3[Text Visualization]
        A3 --> A4[Embedding]
        A4 --> A5[Store in DB]
    end

    subgraph app_drift["app_drift/"]
        B1[Load Embeddings] --> B2[Embeddings Visualization]
        B2 --> B3[Detect Drift]
    end

    subgraph app_report["app_report/"]
        C1[Load Results] --> C2[Build LLM]
        C2 --> C3[Generate Report]
    end
```


#### [1] How to Start DataDrift_Dataclinic 
1. pull this repository
    ```
    git clone https://github.com/keti-datadrift/datadrift_dataclinic.git
    ```
2. change dir
    ```
    cd datadrift_dataclinic/dashboard
    ```
3. make virtual environment
    ```
    python3 -m venv venv
    pip install -r requirements.txt
    ```
4. (option) build Milvus DB
    ```
    cd milvus_db       # cd datadrift_dataclinic/dashboard/milvus_db
    docker compose up -d
    ```

#### [2] Usage Instructions
1. check pwd
    ```
    pwd # datadrift_dataclinic/dashboard
    ```
2. start Streamlit !
    ```
    streamlit run main.py
    ```
3. streamlit 중단

    - (window) `ctrl` + `c`
    - (mac) `pkill -f streamlit`

### Stacks
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white"> <img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=for-the-badge&logo=Pytorch&logoColor=white"> <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=HuggingFace&logoColor=white"> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/Milvus-00A1EA?style=for-the-badge&logo=Milvus&logoColor=white">


## Acknowledgements (사사)
- 이 연구는 2024년도 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구임 (No. RS-2024-00337489, 분석 모델의 성능저하 극복을 위한 데이터 드리프트 관리 기술 개발)
- This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. RS-2024-00337489, Development of data drift management technology to overcome performance degradation of AI analysis models)
