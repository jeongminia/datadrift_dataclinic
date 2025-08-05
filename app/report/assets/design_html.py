from bs4 import BeautifulSoup
import streamlit as st
import base64
import os
from datetime import datetime


# ------------------------------------- CSS -------------------------------------
def get_report_css():
    """리포트용 CSS 스타일 반환"""
    return """
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
        font-family: 'Malgun Gothic', sans-serif; 
        line-height: 1.6; color: #2c3e50; 
        background: #f8f9fa; padding: 30px;
    }
    .container { 
        max-width: 1000px; margin: 0 auto; 
        background: white; padding: 30px; 
        border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .header { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 25px; border-radius: 8px; 
        margin-bottom: 25px; text-align: center;
    }
    .title { font-size: 2em; margin-bottom: 5px; }
    .subtitle { font-size: 1.1em; opacity: 0.9; }
    .section { 
        margin: 25px 0; padding: 20px; 
        border: 1px solid #e9ecef; border-radius: 8px;
    }
    .section-title { 
        font-size: 1.4em; color: #495057; 
        margin-bottom: 15px; padding-bottom: 8px;
        border-bottom: 2px solid #dee2e6;
    }
    table { 
        width: 100%; border-collapse: collapse; margin: 15px 0;
        border-radius: 5px; overflow: hidden;
    }
    th { 
        background: #6c757d; color: white; 
        padding: 10px; text-align: left;
    }
    td { padding: 8px; border-bottom: 1px solid #dee2e6; }
    img { max-width: 100%; height: auto; margin: 10px 0; }
    pre { 
        background: #f8f9fa; padding: 15px; 
        border-radius: 5px; overflow-x: auto;
    }
    .footer { 
        text-align: center; margin-top: 30px; 
        padding: 15px; background: #f8f9fa; 
        border-radius: 5px; color: #6c757d;
    }
    .comment-box {
        background-color: #f4f4f4;
        padding: 15px;
        margin: 10px 0 30px 0;
        border-radius: 8px;
    }
    """

# ------------------------------------- HTML Head와 Body에 해당하는 template -------------------------------------
def head_footer_html(dataset_name, database_content, drift_content, llm_content):
    timestamp = datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')
    css = get_report_css()
    
    return f"""<!DOCTYPE html>
            <html lang="ko">
            <head>
                <meta charset="utf-8">
                <title>{dataset_name} - 통합 분석 리포트</title>
                <style>{css}</style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <div class="title">{dataset_name} 통합 분석 리포트</div>
                        <div class="subtitle">데이터 드리프트 분석 보고서</div>
                        <div style="margin-top: 10px; font-size: 0.9em;">생성일시: {timestamp}</div>
                    </div>
                    
                    <div class="section">
                        <div class="section-title">📊 Dataset Information & Statistics</div>
                        {database_content}
                    </div>
                    
                    <div class="section">
                        <div class="section-title">🔍 Data Drift Analysis Results</div>
                        {drift_content}
                    </div>

                    <div class="section">
                        <div class="section-title">🔍 LLM Explanation</div>
                        {llm_content}
                    </div>
                    
                    <div class="footer">
                        <strong>
                            <a href="https://github.com/keti-datadrift/datadrift_dataclinic" target="_blank" style="color: #3498db; text-decoration: none;">
                            DataDrift Dataclinic System
                            </a>
                        </strong><br>
                        @2025 KETI, Korea Electronics Technology Institute<br>
                    </div>
                </div>
            </body>
            </html>"""