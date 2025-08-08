# datadrift_dataclinic

## 주소
- https://github.com/keti-datadrift/datadrift_dataclinic.git

## 개요
- 데이터 드리프트 관리 기술의 기반 프레임워크입니다.
- 개발 및 유지 관리 기관 : 한국전자기술연구원(KETI)
- 최종 검토 기관 : __한국전자기술연구원(KETI)__

### Full Flow
--------
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white) [![EvidentlyAI](https://img.shields.io/badge/EvidentlyAI-0.5.1-red?logo=EvidentlyAI&logoColor=white)](https://github.com/evidentlyai/evidently/tree/v0.5.1)

```mermaid
flowchart TD
    subgraph app_report["app/report/"]
        A1[Load Results] --> A2[Build LLM]
        A2 --> A3[Generate Report]
    end

    subgraph app_drift["app/drift/"]
        B1[Load Embeddings] --> B2[Embeddings Visualization]
        B2 --> B3[Detect Drift]
    end

    subgraph app_database["app/database/"]
        C1[Upload Data] --> C2[Load Data]
        C2 --> C3[Text Visualization]
        C3 --> C4[Embedding]
        C4 --> C5[Store in DB]
    end
```


#### [1] How to Start DataDrift_Dataclinic 
- This project works properly only in environments where CUDA is available.

1. pull this repository
    ```
    git clone https://github.com/keti-datadrift/datadrift_dataclinic.git
    ```
2. change dir
    ```
    cd datadrift_dataclinic
    ```
3. make virtual environment
    ```
    brew install python@3.11
    python3.10 -m venv venv
    source venv/bin/activate

    pip install -r requirements.txt
    ```
4. intall faiss
    - (only Available GPU) `pip install faiss-gpu`

#### [2] Setup environments

1. build Milvus DB
    ```
    cd milvus_db                   # cd datadrift_dataclinic/db/milvus_db
    docker compose up -d
    ```

2. build Ollama
    
    2.1 [initial settings](docs/build_ollama.md)
    
    2.2 pull models
        
        ```
        cat ollama-models.txt | xargs -n 1 ollama pull
        ```
    
    2.3 (optional) To check the installed models:
        
        ```
        ollama list
        ```

#### [3] Usage Instructions
1. check pwd
    ```
    pwd                           # datadrift_dataclinic
    ```
2. Start Streamlit !
    ```
    streamlit run main.py
    ```
3. END streamlit 

    - (window) `ctrl` + `c`
    - (mac) `pkill -f streamlit`

### Stacks
<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white" height="24"> <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white" height="24"> <img src="https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=HTML5&logoColor=white" height="24"> <img src="https://img.shields.io/badge/CSS-663399?style=flat&logo=CSS&logoColor=white" height="24"> <img src="https://img.shields.io/badge/Milvus-00A1EA?style=flat&logo=Milvus&logoColor=white" height="24"> <img src="https://img.shields.io/badge/Ollama-000000?style=flat&logo=Ollama&logoColor=white" height="24"> <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=LangChain&logoColor=white" height="24"> <img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=flat&logo=Pytorch&logoColor=white" height="24"> <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=HuggingFace&logoColor=white" height="24">  


## Acknowledgements (사사)
- 이 연구는 2024년도 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구임 (No. RS-2024-00337489, 분석 모델의 성능저하 극복을 위한 데이터 드리프트 관리 기술 개발)
- This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. RS-2024-00337489, Development of data drift management technology to overcome performance degradation of AI analysis models)
