import streamlit as st
import subprocess
import json
from langchain_ollama import OllamaLLM

def get_default_drift_prompt():
    """기본 드리프트 분석 프롬프트"""
    return """당신은 데이터 드리프트 분석 전문가입니다. 다음 정보를 바탕으로 드리프트 분석 해설을 작성해주세요.

            데이터셋: {dataset_name}
            드리프트 분석 결과: {drift_summary}
            임베딩 정보: {embedding_info}

            참고 지식:
            {context}

            다음 4단계 구조로 한국어로 작성해주세요:

            [기술적 분석] 각 드리프트 메트릭의 수치적 의미를 해석합니다.
            [현 상황 분석] 현재 드리프트 상황이 모델에 미칠 영향을 평가합니다.
            [시각적 분석] PCA 시각화 결과와 연계하여 데이터 분포 변화를 해석합니다.
            [권장사항] 즉시 조치사항과 모니터링 방안을 제시합니다.

            각 단계는 2-3문장으로 간결하게 작성하세요."""

def generate_drift_explanation_preview():
    """드리프트 해설 미리보기 생성"""
    try:
        # 세션에서 드리프트 데이터 가져오기
        dataset_name = st.session_state.get('selected_dataset', 'Test Dataset')
        drift_summary = st.session_state.get('drift_summary', '드리프트 데이터가 없습니다.')
        
        if drift_summary == '드리프트 데이터가 없습니다.':
            return None
        
        # Custom LLM 사용
        llm = get_custom_llm()
        if not llm:
            return None
        
        # 프롬프트 템플릿 가져오기
        prompt_template = st.session_state.get('custom_prompt_template', get_default_drift_prompt())
        
        # 프롬프트 포맷팅
        formatted_prompt = prompt_template.format(
            dataset_name=dataset_name,
            drift_summary=drift_summary,
            embedding_info="Preview mode - 실제 임베딩 정보",
            context="Preview mode - RAG 검색 결과"
        )
        
        # LLM 호출
        explanation = llm.invoke(formatted_prompt)
        
        return f"""
        <div style="background-color: #e8f5e8; border-left: 4px solid #28a745; padding: 20px; margin: 10px 0;">
            <h4 style="margin-top: 0; color: #28a745;">🤖 AI 드리프트 분석 해설 (미리보기)</h4>
            <div style="line-height: 1.8; font-size: 14px;">
                {explanation.replace(chr(10), '<br>')}
            </div>
            <small style="color: #6c757d; font-style: italic; margin-top: 15px; display: block;">
                * 이는 미리보기입니다. 실제 Generate Report에서 전체 분석이 진행됩니다.
            </small>
        </div>
        """
        
    except Exception as e:
        st.error(f"미리보기 생성 중 오류: {e}")
        return None

def get_ollama_models():
    """설치된 Ollama 모델 목록 가져오기"""
    try:
        # ollama list 명령어 실행
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        
        models = []
        lines = result.stdout.strip().split('\n')
        
        # 헤더 제거하고 모델명만 추출
        for line in lines[1:]:  # 첫 번째 줄은 헤더
            if line.strip():
                model_name = line.split()[0]  # 첫 번째 컬럼이 모델명
                if model_name and not model_name.startswith('NAME'):
                    models.append(model_name)
        
        return models
        
    except subprocess.CalledProcessError:
        st.warning("Ollama가 실행되지 않거나 설치되지 않았습니다.")
        return []
    except FileNotFoundError:
        st.warning("Ollama가 설치되지 않았습니다.")
        return []
    except Exception as e:
        st.warning(f"모델 목록을 가져오는 중 오류: {e}")
        return []

def get_custom_llm():
    """설정된 Custom LLM 객체 반환"""
    if not st.session_state.get('custom_llm_configured'):
        return None
    
    try:
        llm = OllamaLLM(
            model=st.session_state.get('custom_llm_model'),
            temperature=st.session_state.get('custom_llm_temperature', 0.7)
        )
        return llm
    except Exception as e:
        st.error(f"LLM 로드 중 오류: {e}")
        return None

def render():
    """Custom LLM 설정 페이지"""
    st.markdown("""
            <div style="background: linear-gradient(135deg, #c7a2ff 0%, #9b7bff 50%, #b797ff 100%);
                        padding: 15px 25px; border-radius: 15px; margin-bottom: 25px;
                        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
                        border: 1px solid rgba(255, 255, 255, 0.3);
                        transform: scale(0.95);">
                <div style="text-align: center;">
                    <h3 style="color: white; margin: 0; font-weight: 700; font-size: 18px; 
                            text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                        Custom LLM Configuration
                    </h3>
                    <div style="color: rgba(255, 255, 255, 0.95); font-size: 12px; margin-top: 8px;
                            text-shadow: 0 1px 2px rgba(0,0,0,0.2);">
                        드리프트 분석 결과에 대한 맞춤형 해설을 위해 AI 모델을 설정합니다
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # 현재 선택된 데이터셋 확인
    selected_dataset = st.session_state.get('selected_dataset')
    if not selected_dataset:
        st.warning("⚠️ 먼저 결과 데이터를 로드해주세요.")
        return
    
    # ============ 1. 모델 설정 섹션 ============
    st.markdown("#### 1. Select AI Model & Settings")
    
    # Ollama 모델 목록 가져오기
    available_models = get_ollama_models()
    
    if not available_models:
        st.error("❌ Ollama 모델을 찾을 수 없습니다.")
        st.code("ollama list", language="bash")
        return
    
    # 모델 선택 UI
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox(
            "🎯 사용할 AI 모델을 선택하세요:",
            available_models,
            key="model_selection",
            help="드리프트 분석 해설을 생성할 AI 모델을 선택합니다."
        )
    
    with col2:
        temperature = st.slider(
            "🌡️ Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            key="temperature_slider",
            help="높을수록 창의적, 낮을수록 일관된 답변"
        )

    col1, col2 = st.columns(2)
        
    with col1:
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=1000,
            step=100,
            help="생성할 최대 토큰 수"
            )
            
    with col2:
        top_p = st.slider(
            "Top P",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.1,
            help="누적 확률 임계값"
            )
    
    # LLM 설정 저장
    if st.button("✅ LLM 설정 완료", type="primary", key="save_llm_config"):
        try:
            # LLM 객체 생성 및 테스트
            llm = OllamaLLM(
                model=selected_model,
                temperature=temperature
            )
            
            # 간단한 테스트
            test_response = llm.invoke("Hello")
            
            # 세션에 저장
            st.session_state.custom_llm_model = selected_model
            st.session_state.custom_llm_temperature = temperature
            st.session_state.custom_llm_max_tokens = max_tokens
            st.session_state.custom_llm_top_p = top_p
            st.session_state.custom_llm_configured = True
        
        except Exception as e:
            st.error(f"❌ 모델 설정 중 오류가 발생했습니다: {e}")
    
    # 모델이 설정되지 않았으면 여기서 종료
    if not st.session_state.get('custom_llm_configured'):
        return

    
    # ============ 2. 프롬프트 커스터마이징 섹션 ============
    st.markdown("#### 2. Customize Prompt Template")
    
    # 프리셋 선택
    preset_options = {
        "기본 드리프트 분석": get_default_drift_prompt(),
        "사용자 정의": ""
    }
    
    selected_preset = st.selectbox(
        "📋 프롬프트 프리셋 선택:",
        list(preset_options.keys()),
        help="미리 정의된 프롬프트 템플릿을 선택하거나 직접 작성하세요."
    )
    
    # 프롬프트 변수 설명
    with st.expander("📖 사용 가능한 변수"):
        st.code("""
            {dataset_name} - 선택된 데이터셋 이름
            {drift_summary} - 드리프트 분석 결과 요약
            {embedding_info} - 임베딩 정보
            {context} - RAG에서 검색된 관련 지식
                    """)

    # 프롬프트 템플릿 편집
    if selected_preset == "사용자 정의":
        custom_prompt = st.text_area(
            "사용자 정의 프롬프트:",
            height=250,
            placeholder="여기에 원하는 프롬프트 템플릿을 작성하세요...",
            help="변수: {dataset_name}, {drift_summary}, {embedding_info}"
        )
    else:
        custom_prompt = st.text_area(
            f"{selected_preset} 프롬프트:",
            value=preset_options[selected_preset],
            height=250,
            help="프롬프트를 수정하거나 그대로 사용하세요."
        )
    
    # 프롬프트 저장
    if st.button("💾 프롬프트 저장", key="save_prompt"):
        st.session_state.custom_prompt_template = custom_prompt
    
    # 프롬프트가 저장되지 않았으면 제한적으로 표시
    if not st.session_state.get('custom_prompt_template'):
        return
    
    # ============ 3. 설정 확인 및 미리보기 섹션 ============
    st.markdown("#### 3. Check Settings")
    
    # 현재 설정 표시
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        st.metric("Model Name", st.session_state.get('custom_llm_model', 'N/A'))
        
    with config_col2:
        st.metric("Temperature", st.session_state.get('custom_llm_temperature', 'N/A'))
        
    with config_col3:
        st.metric("State", "✅")
    
    # 모든 설정 완료 상태 확인
    all_configured = (
        st.session_state.get('custom_llm_configured') and 
        st.session_state.get('custom_prompt_template')
    )
    
    if all_configured:
        st.success("🎉 모든 설정이 완료되었습니다! Generate Report에서 최종 리포트를 생성하세요!")
        
    else:
        st.info("⚠️ 모든 설정을 완료하면 Generate Report로 이동할 수 있습니다.")