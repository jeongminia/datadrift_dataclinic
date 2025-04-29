import os
import contextlib
import tempfile
from llama_cpp import Llama

@contextlib.contextmanager
def suppress_stdout_stderr():
    """C-level stdout/stderr 억제용 context manager"""
    with tempfile.TemporaryFile() as fnull:
        fd_stdout = os.dup(1)
        fd_stderr = os.dup(2)
        os.dup2(fnull.fileno(), 1)
        os.dup2(fnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(fd_stdout, 1)
            os.dup2(fd_stderr, 2)

# 모델 경로
model_path = "/home/keti/datadrift_jm/models/gpt4all/ggml-model-Q4_K_M.gguf"

# suppress + verbose=False
with suppress_stdout_stderr():
    model = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=8,
        verbose=False
    )

# 프롬프트 설정
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

# 응답 생성
response = model(
    full_prompt,
    max_tokens=300,
    temperature=0.7,
    top_p=0.9,
    repeat_penalty=1.1
)

# 원하는 출력만!
print("📌 테스트 응답:", response["choices"][0]["text"].strip())
