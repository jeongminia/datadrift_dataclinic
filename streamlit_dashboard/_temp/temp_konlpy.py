import streamlit as st
from konlpy.tag import Mecab
import os

# Streamlit 초기화
st.title("Streamlit Mecab Test")

# Mecab 경로 설정
os.environ["MECAB_PATH"] = "/opt/homebrew/lib/mecab/dic/mecab-ko-dic"

# Mecab 객체 생성 및 테스트
try:
    mecab = Mecab(dicpath=os.environ["MECAB_PATH"])
    test_text = "스트림릿 환경에서 Mecab이 제대로 작동하는지 확인합니다."
    nouns = mecab.nouns(test_text)
    st.write("Mecab 정상 작동!")
    st.write(f"분석된 명사: {nouns}")
except Exception as e:
    st.write("Mecab 초기화 오류:", e)
