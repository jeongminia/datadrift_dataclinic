from gpt4all import GPT4All

model_path = "/home/keti/datadrift_jm/models/gpt4all/ggml-model-Q4_K_M.gguf"
model = GPT4All(model_path)

test_prompt = """
총 문서 수: 4078
평균 문장 길이: 13 단어
주요 키워드: 살인, 피고인, 피해자, 증거, 판결
"""

full_prompt = f"""
당신은 데이터 분석 보고서를 작성하는 AI입니다.
다음 데이터 통계를 보고 분석 결과를 5문장 이내로 자연스럽게 설명하세요.

{test_prompt}

→ 분석 요약:
"""

output = model.generate(full_prompt, max_tokens=300)
print("📌 테스트 응답:", output)
