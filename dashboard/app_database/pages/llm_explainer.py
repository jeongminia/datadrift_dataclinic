from gpt4all import GPT4All

# 모델 로딩 (최초 1회)
model = GPT4All("gpt4all-j")  # 또는 절대 경로 지정 가능

def generate_explanation(context: str) -> str:
    prompt = f"""
    다음은 텍스트 데이터셋의 특성 요약입니다. 아래 내용을 기반으로 분석 결과를 이해하기 쉽게 설명해 주세요.

    {context}

    설명:
    """
    return model.generate(prompt, max_tokens=300, temp=0.7).strip()
