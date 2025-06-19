import requests

# Ollama 서버 주소와 사용할 모델 이름
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "joonoh/HyperCLOVAX-SEED-Text-Instruct-1.5B:latest"  # 또는 설치된 ollama 모델명(exaone3.5 등)

test_prompt = """
총 문서 수: 4078
평균 문장 길이: 13 단어
주요 키워드: 살인, 피고인, 피해자, 증거, 판결
"""

full_prompt = f"""
당신은 데이터 드리프트를 쉽게 이해할 수 있도록 사용자를 돕는 보고서를 작성하는 AI입니다.
데이터 드리프트란 입력 데이터의 변화로 인해 모델의 예측 성능이 저하되는 현상입니다.

train, test, validation 데이터셋을 기반으로 한 데이터 드리프트 분석을 위해 시각화 및 EDA 결과를 요약하세요.
다음 데이터 통계를 보고 분석 결과를 5문장 이내로 자연스럽게 설명하세요.

{test_prompt}

→ 분석 요약:
"""

payload = {
    "model": OLLAMA_MODEL,
    "prompt": full_prompt,
    "stream": False,
    "options": {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 300,
        "repeat_penalty": 1.1
    }
}

response = requests.post(OLLAMA_URL, json=payload, timeout=60)
result = response.json()
print("📌 테스트 응답:", result["response"].strip())