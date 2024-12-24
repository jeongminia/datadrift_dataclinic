
## 주요 기능
page 구성은 다음과 같다
- 데이터 로드
- 베이스 시각화
- 임베딩 시각화
- 데이터 드리프트 탐지

## 사용 방법
1. streamlit 접속
    
    ```
    streamlit run main.py
    ```

2. ~~데이터셋 업로드~~(추후 수정 예정)
3. 사이드 바를 통해 각 페이지 확인
4. streamlit 중단

    - (window) `ctrl` + `c`
    - (mac) `pkill -f streamlit`

## Install Prerequisite
- library : [requirements](requirements.txt)
- fonts : [나눔고딕 레귤러](https://fonts.google.com/selection)
- datasets : 법률 관련 문서 데이터셋 (한림대학교 연구팀 구축)

## Dir Structure
```
streamlit_dashboard/
├── pages/
│   ├── data_load.py
│   ├── basic_visualization.py
│   ├── embedding_visualization.py
│   └── detect_datadrift.py
├── main.py
├── data/
│   ├── train_data.csv
│   ├── val_data.csv
│   └── test_data.csv

```

## Stacks

<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white"> <img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=for-the-badge&logo=Pytorch&logoColor=white"> <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=HuggingFace&logoColor=white"> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">