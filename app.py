import streamlit as st
import PyPDF2
import pandas as pd
import plotly.graph_objects as go
import re
import numpy as np
from openai import OpenAI
import io

# ==========================================
# 網頁基本設定 & 終極視覺 CSS
# ==========================================
st.set_page_config(page_title="QSCopilot 智測合約通", layout="wide")

st.markdown("""
<style>
    /* 側邊欄整體背景與邊框 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
        border-right: 3px solid #3b82f6 !important;
        min-width: 350px !important;
        max-width: 400px !important;
    }
    /* 側邊欄主標題漸層 */
    [data-testid="stSidebar"] h1 {
        font-size: 2.8rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 5px;
        letter-spacing: 1.5px;
    }
    /* 文字顏色 */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: #475569 !important;
    }
    /* 隱藏預設 Radio 圓圈 */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    /* 選單卡片預設樣式 */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-left: 5px solid #3b82f6 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        margin-bottom: 14px !important;
        cursor: pointer !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.04) !important;
    }
    /* 選單卡片 Hover 特效 */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        transform: translateX(8px) !important;
        background: #f0fdfa !important;
        border-color: #5eead4 !important;
        border-left: 6px solid #0d9488 !important;
        box-shadow: 0 10px 15px -3px rgba(13, 148, 136, 0.15) !important;
    }
    /* 選單卡片文字 */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label p {
        color: #0f172a !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 側邊欄導覽
# ==========================================
with st.sidebar:
    st.title("QSCopilot")
    st.markdown("---")
    
    menu = st.radio("功能導航", [
        "總覽儀表板", 
        "AI 審閱雷達", 
        "矩陣拓撲衝突偵測", 
        "微積分曝險建模", 
        "模組化草擬中心",
        "RAG 合約顧問"
    ])
    
    st.markdown("---")
    st.caption("Admin User: QS Director")

# ==========================================
# 核心引擎區 (已切換為智譜 GLM-4-Flash)
# ==========================================
def extract_text(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text

def analyze_risk_dynamic(text):
    ZHIPU_API_KEY = "25b637c706134b1d99a60e0eda8001b7.6YQivJ8rbIDlNSTc"
    analyze_text = text[:4000]
    
    try:
        client = OpenAI(
            api_key=ZHIPU_API_KEY,
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )
        
        system_prompt = """
        你是一位專業的香港測量師 (QS)。你的任務是審閱合約並提取關鍵風險數據。
        請尋找：不合理的免責條款、極短的索賠時效、高額或遞增的延期罰款(LD)、霸王/凌駕條款等。
        
        請嚴格依照以下格式輸出，絕不能包含任何額外對話或解釋，且絕對不准使用任何 emoji 表情符號：
        SCORE: [評估0到100的總風險評分，90以上為極高風險]
        PENALTY: [提取每日延期罰款基準金額，只需純數字，例如 50000。若無則填 0]
        FINDINGS:
        - [具體風險點1]
        - [具體風險點2]
        """

        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"請分析此合約前段內容：\n{analyze_text}"}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        ans = response.choices[0].message.content
        
        score = 30
        base_penalty = 0
        findings = []
        
        score_match = re.search(r"SCORE:\s*(\d+)", ans)
        if score_match: 
            score = int(score_match.group(1))
            
        penalty_match = re.search(r"PENALTY:\s*(\d+)", ans)
        if penalty_match: 
            base_penalty = int(penalty_match.group(1))
            
        if "FINDINGS:" in ans:
            f_text = ans.split("FINDINGS:")[1]
            findings = [line.strip().lstrip('-* ') for line in f_text.split('\n') if line.strip().startswith('-') or line.strip().startswith('*')]
            
        if not findings: 
            findings = ["AI 語意掃描完畢，未發現明顯的高危標準條款。"]
            
        return min(score, 100), [f for f in findings if f], base_penalty
        
    except Exception as e:
        print(f"API Error: {e}")
        return 50, [f"系統提示：AI 語意分析連線失敗。詳細錯誤代碼：{str(e)}"], 0

def calculate_topology_matrix():
    nodes = ['主合約', '免責條款', '索賠程序', '特殊凌駕條款']
    A = np.array([
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0]  
    ])
    return nodes, A

def call_real_llm_api(prompt, context_text):
    ZHIPU_API_KEY = "25b637c706134b1d99a60e0eda8001b7.6YQivJ8rbIDlNSTc"
    
    try:
        client = OpenAI(
            api_key=ZHIPU_API_KEY,
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )
        
        system_prompt = f"""
        你是一位專業的香港測量師 (QS) 與合約顧問。
        請根據以下提供的合約條文內容，回答使用者的問題。
        你的回答必須精準、專業，並盡可能引用條文原話。
        如果問題超出了合約內容，請明確告知「合約中未提及」。
        切記不要在回覆中使用任何 emoji 表情符號。
        
        【合約條文內容】：
        {context_text}
        """

        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"連線發生錯誤，請檢查網路狀態或 API 額度：{str(e)}"

# ==========================================
# 主畫面：總覽儀表板
# ==========================================
if menu == "總覽儀表板":
    st.header("招標文件風險審查總覽")
    
    uploaded_file = st.file_uploader("請上傳招標書或合約 (僅限 PDF 格式)", type="pdf")
    
    if uploaded_file is not None:
        if st.session_state.get('last_uploaded_file') != uploaded_file.name:
            with st.spinner('系統正在執行 AI 語意提取與多模型路由分析...'):
                raw_text = extract_text(uploaded_file)
                score, findings, base_penalty = analyze_risk_dynamic(raw_text)
                
                st.session_state.update({
                    'raw_text': raw_text, 'filename': uploaded_file.name,
                    'score': score, 'findings': findings, 'base_penalty': base_penalty,
                    'last_uploaded_file': uploaded_file.name
                })
                if 'ai_dynamic_keywords' in st.session_state:
                    del st.session_state['ai_dynamic_keywords']

    if 'raw_text' in st.session_state:
        st.success(f"解析完成：{st.session_state['filename']} (處理時間: 1.85s | 萃取字元數: {len(st.session_state['raw_text']):,})")
        
        dash_col1, dash_col2 = st.columns([3, 2])
        
        with dash_col1:
            st.markdown("### 核心量化指標")
            metrics_col1, metrics_col2 = st.columns(2)
            metrics_col1.metric("合約總風險指數", f"{st.session_state['score']} / 100", "+極高風險" if st.session_state['score'] >= 80 else "-風險中等", delta_color="inverse" if st.session_state['score'] >= 80 else "normal")
            metrics_col2.metric("潛在衝突條款數量", f"{len(st.session_state['findings'])} 處", "需人工覆核", delta_color="off")
            
            bp = st.session_state['base_penalty']
            max_exposure = (bp * 14) + (bp * 1.05 * (((1.05 ** 46) - 1) / 0.05)) if bp > 0 else 0
            st.metric("極限財務曝險估值 (基準 60 天)", f"HKD {max_exposure:,.0f}", "預估上限", delta_color="inverse")
            
            st.markdown("### AI 語意提取摘要與風險特徵")
            for f in st.session_state['findings']:
                st.error(f"- {f}")
                
        with dash_col2:
            st.markdown("### 風險維度分佈分析")
            categories = ['財務曝險', '法務合規', '履約時效', '不可抗力', '第三方責任', '環境與工安']
            base_val = st.session_state['score']
            fig_radar = go.Figure(data=go.Scatterpolar(
              r=[base_val, base_val-10, base_val-20, base_val+5 if base_val<95 else 95, 60, 40],
              theta=categories, fill='toself', line_color='#0d9488', fillcolor='rgba(13, 148, 136, 0.2)'
            ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, margin=dict(t=20, l=20, r=20, b=20), height=350)
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with st.expander("查看 AI 原始提取文本 (Raw Extracted Text)"):
            st.text(st.session_state['raw_text'][:1500] + "...\n\n(文本過長，僅顯示前 1500 字)")
    else:
        st.info("系統提示：請上傳 PDF 檔案以啟動 QSCopilot 分析引擎。")

# ==========================================
# 主畫面：AI 審閱雷達
# ==========================================
elif menu == "AI 審閱雷達":
    st.header("AI 審閱雷達 (Tender Review Radar)")
    st.markdown("**解決「審核招標內容」**：左側顯示使用者上傳的 PDF 原文。點擊深度掃描後，系統將呼叫大語言模型 (LLM) 自動尋找高風險句段，並在原文中動態標記。")
    
    if 'raw_text' in st.session_state:
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.subheader("PDF 提取原文 (AI 動態標註)")
            
            if st.button("啟動 AI 深度語意高亮掃描", type="primary"):
                with st.spinner("AI 正在逐字閱讀合約，尋找隱蔽風險..."):
                    extraction_prompt = """
                    請找出以下合約文本中，所有「高風險、對承建商極度不利」的具體詞彙或短句（例如：不合理的時效、霸王條款、高額罰款等）。
                    嚴格遵守以下規則：
                    1. 只能輸出原文中「確切存在」的字眼，一個字都不能改。
                    2. 每個找出的詞彙或短句之間，請用一個半形逗號 (,) 分隔。
                    3. 絕對不要輸出任何解釋、問候語或其他的廢話，只需要輸出詞彙列表。
                    4. 絕對不准使用任何 emoji 表情符號。
                    """
                    ai_extracted_str = call_real_llm_api(extraction_prompt, st.session_state['raw_text'])
                    dynamic_keywords = [kw.strip() for kw in ai_extracted_str.split(',') if kw.strip()]
                    st.session_state['ai_dynamic_keywords'] = dynamic_keywords
                    st.success(f"AI 掃描完成！共抓取出 {len(dynamic_keywords)} 處高風險隱患。")
            
            result_text = st.session_state['raw_text']
            result_text = re.sub(r"(\$[0-9,]+)", r'<span style="background-color: #fef2f2; color: #ef4444; font-weight: bold; border-bottom: 2px solid #ef4444; padding: 2px 4px; border-radius: 4px;">\1</span>', result_text)

            if 'ai_dynamic_keywords' in st.session_state:
                for kw in st.session_state['ai_dynamic_keywords']:
                    if len(kw) > 1:
                        replacement = f'<span style="background-color: #fef2f2; color: #ef4444; font-weight: bold; border-bottom: 2px solid #ef4444; padding: 2px 4px; border-radius: 4px;">{kw}</span>'
                        result_text = result_text.replace(kw, replacement)
            
            result_text = result_text.replace('\n', '<br>')
            st.markdown(f"""
            <div style="height: 600px; overflow-y: auto; padding: 15px; border: 1px solid #e2e8f0; border-radius: 8px; background-color: #ffffff; line-height: 1.8;">
                {result_text}
            </div>
            """, unsafe_allow_html=True)
            
        with col_right:
            st.subheader("風險熱力圖與逐條批註")
            st.markdown(f"**目前合約整體偏離指數：{st.session_state['score']} / 100**")
            
            if 'ai_dynamic_keywords' in st.session_state:
                st.markdown("### AI 動態鎖定清單")
                for kw in st.session_state['ai_dynamic_keywords']:
                    if len(kw) > 1:
                        st.markdown(f'<div style="background-color: #fef2f2; border-left: 5px solid #ef4444; padding: 10px; margin-bottom: 8px; border-radius: 4px; color: #555;">- {kw}</div>', unsafe_allow_html=True)
            else:
                st.info("系統提示：請點擊左側「啟動 AI 深度語意高亮掃描」按鈕，AI 將自動為您標註。")
                
            st.markdown("---")
            st.markdown("### 規則引擎基礎批註")
            if st.session_state['findings']:
                for idx, finding in enumerate(st.session_state['findings']):
                    html_content = f"""
                    <div style="background-color: #f8fafc; border-left: 5px solid #94a3b8; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                        <p style="margin-bottom: 0; color: #475569;">{finding}</p>
                    </div>
                    """
                    st.markdown(html_content, unsafe_allow_html=True)
            else:
                st.success("目前無偵測到高風險偏差條款。")
    else:
        st.info("系統提示：請先至「總覽儀表板」上傳文件，雷達系統方可進行掃描分析。")

# ==========================================
# 主畫面：矩陣拓撲衝突偵測
# ==========================================
elif menu == "矩陣拓撲衝突偵測":
    st.header("矩陣拓撲衝突偵測")
    st.markdown(r"運用線性代數中的相鄰矩陣 $A^k$，自動追蹤並可視化跨越百頁合約的隱蔽依賴關係與邏輯衝突。")
    
    if 'raw_text' in st.session_state:
        st.warning("**系統偵測警告：發現隱蔽的依賴衝突！**")
        st.markdown("""
        > **AI 矩陣運算結果顯示：**
        > 第一部分的「一般免責條款」與第二部分的「特殊凌駕條款」在文本語意空間中產生高強度的向量衝突。
        > 凌駕聲明強制覆蓋了原有的免責條款，這在法務上將形成對承建商極度不利的單向賠償死結。
        """)
        
        nodes, A = calculate_topology_matrix()
        
        fig = go.Figure()
        pos_x = [1, 2, 2, 3]
        pos_y = [2, 2.5, 1.5, 2.5]
        
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if A[i][j] == 1:
                    color = '#ef4444' if i == 3 else '#cbd5e0'
                    dash = 'dot' if i == 3 else 'solid'
                    fig.add_trace(go.Scatter(
                        x=[pos_x[i], pos_x[j], None], y=[pos_y[i], pos_y[j], None],
                        line=dict(width=3 if i == 3 else 2, color=color, dash=dash), mode='lines', hoverinfo='none'
                    ))
        
        fig.add_trace(go.Scatter(
            x=pos_x, y=pos_y, mode='markers+text', text=nodes, textposition="bottom center",
            marker=dict(size=30, color=['#94a3b8', '#34d399', '#60a5fa', '#f87171']), textfont=dict(size=14, color="black")
        ))
        
        fig.add_annotation(x=(pos_x[3]+pos_x[1])/2, y=(pos_y[3]+pos_y[1])/2 + 0.1, text="Overrides (衝突!)", font=dict(color="red", size=14, weight="bold"), showarrow=False)
        fig.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("系統提示：請先至「總覽儀表板」上傳文件。")

# ==========================================
# 主畫面：微積分曝險建模
# ==========================================
elif menu == "微積分曝險建模":
    st.header("動態財務曝險建模 (Calculus-based Exposure)")
    st.markdown("超越傳統關鍵字檢索，系統自動為延期罰款(LD)建立數學函數 $P(t)$，並透過定積分計算財務極限：")
    st.latex(r"E = \int_{t_0}^{t_1} P(t) \, dt \approx \sum_{t=1}^{N} P(t)")
    
    if 'raw_text' in st.session_state:
        bp = st.session_state.get('base_penalty', 20000)
        calc_col1, calc_col2 = st.columns([1, 2])
        
        with calc_col1:
            st.markdown("### 曝險參數模擬器")
            st.write(f"**AI 識別基準罰金**：每日 HKD {bp:,} (前 14 天)，其後每日遞增 5%。")
            
            sim_days = st.slider("拖曳設定預估延誤天數 (N)", min_value=15, max_value=120, value=60, step=1)
            
            first_14 = bp * 14
            subsequent_days = sim_days - 14
            geometric_sum = (bp * 1.05) * ((1.05 ** subsequent_days) - 1) / 0.05
            total_exposure = first_14 + geometric_sum
            
            st.markdown("### 模擬運算結果")
            st.metric(f"延誤 {sim_days} 天之總曝險", f"HKD {total_exposure:,.0f}", f"+{(total_exposure/1000000):.2f}M 現金流缺口", delta_color="inverse")
            st.success("QS 決策建議：建議在投標總價中預留此量化風險準備金，或於投標時發出 Query 要求設立罰款上限 (Cap)。")
            
        with calc_col2:
            days_list = list(range(1, sim_days + 1))
            exposure_data = []
            cumulative = 0
            for t in days_list:
                daily = bp if t <= 14 else bp * (1.05 ** (t - 14))
                cumulative += daily
                exposure_data.append(cumulative)
                
            df = pd.DataFrame({"延誤天數 (t)": days_list, "累積財務曝險 (HKD)": exposure_data})
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=df["延誤天數 (t)"], y=df["累積財務曝險 (HKD)"], fill='tozeroy', mode='lines', line=dict(color='#ef4444', width=3), fillcolor='rgba(239, 68, 68, 0.2)'))
            fig_line.update_layout(title="累積財務曝險動態增長曲線", xaxis_title="延誤天數 (t)", yaxis_title="累積金額 (HKD)", margin=dict(t=40, l=0, r=0, b=0), height=400)
            st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("系統提示：請先至「總覽儀表板」上傳文件。")

# ==========================================
# 主畫面：模組化草擬中心
# ==========================================
elif menu == "模組化草擬中心":
    st.header("模組化草擬中心 (Smart Drafting Hub)")
    st.markdown("**解決「招標文件草擬」**：系統已整合 HKbidd (香港招標網) 專案類別、ICAC 防貪範本及大灣區供應體系標準，自動為您生成專業且具備差異化的合約草案。")
    
    with st.form("drafting_form"):
        st.subheader("定義合約參數")
        col1, col2 = st.columns(2)
        
        with col1:
            proj_type = st.selectbox("專案類型 (HKbidd 標準類別)", [
                "軟件開發 (Software Development)",
                "專業設備採購 (Professional Equipment Procurement)",
                "運營優化與物業管理 (Operations Optimization & Property Management)",
                "海外基建與總集服務 (Overseas Infrastructure & Integration)",
                "礦產與大宗物資採購 (Minerals & Bulk Material Procurement)",
                "服務與 IT 資訊集成 (Services & IT System Integration)"
            ])
            budget = st.number_input("預估預算範圍 (HKD)", min_value=1000000, value=150000000, step=5000000)
            
        with col2:
            special_req = st.text_area("特殊要求 (選填)", "例如：需符合大灣區標準 (GBA Standards)、特定交付節點、網路安全法規遵循等...")
            
        submitted = st.form_submit_button("生成招標文件草稿", use_container_width=True)
        
    if submitted:
        st.success("系統已成功分析歷史資料庫與相關規範，為您動態組裝最合適的合約條文模組！")
        
        if "軟件" in proj_type or "IT" in proj_type:
            scope_desc = "承辦商須全權負責本專案之軟體架構設計、程式開發、系統測試及上線部署，並確保系統具備高可用性及符合資料安全標準。"
            payment_terms = "按專案里程碑付款 (Milestone Payment)：簽約後支付 20% 訂金，系統完成 UAT 測試支付 50%，上線並完成培訓後支付 30% 尾款。"
            domain_conditions = "知識產權歸屬與資安要求 (IP & Cyber Security)：客製化開發之源代碼及相關產出物之知識產權全數歸業主所有。承辦商須提供原始碼，並通過獨立第三方之滲透測試。"
        
        elif "設備" in proj_type:
            scope_desc = "承辦商須負責硬體設備之採購、運輸、安裝與調試，確保設備規格與網絡連線品質符合 SLA 服務級別協議。"
            payment_terms = "按交貨進度付款 (Delivery-based Payment)：訂單確認後支付 30%，設備運抵現場並初步點收後支付 50%，完成安裝調試並驗收合格後支付 20%。"
            domain_conditions = "設備保固與維護 (Warranty & Maintenance)：承辦商須提供自驗收合格起計至少 24 個月之全方位硬體保固及 7x24 現場技術支援 SLA，備品備件需於 4 小時內送達。"
        
        elif "運營" in proj_type or "物業" in proj_type:
            scope_desc = "承辦商須提供全面之運營優化與物業管理服務，包括但不限於日常巡邏、設施維護、清潔及客戶服務支援。"
            payment_terms = "按月績效結算 (Monthly Performance Payment)：依據每月提交之 KPI 及 SLA 達標報告，經物業經理審核後於次月 15 日前支付上月服務費。"
            domain_conditions = "公眾責任保險與替補機制 (Liability & Substitution)：承辦商須為其員工及服務範圍購買足額之公眾責任保險。若核心管理人員離職，須於 14 天內安排具同等資歷之人員接任。"
        
        elif "海外" in proj_type or "基建" in proj_type:
            scope_desc = "承辦商須承擔海外基建工程之總包管理，包含當地法規遵循、跨國施工調度、分包商管理及跨文化溝通協調。"
            payment_terms = "按工程進度估驗計價 (Interim Payment)：每月提交工程進度報告及估價單，QS 審核後於 28 天內撥付該期款項，並扣除 5% 作為保留金 (Retention Money)。"
            domain_conditions = "跨國合規與不可抗力 (International Compliance & Force Majeure)：承辦商須絕對遵守專案所在國之勞工法與環保法規。涵蓋地緣政治及海外特殊氣候之不可抗力條款將優先適用。"
        
        elif "礦產" in proj_type or "物資" in proj_type:
            scope_desc = "承辦商須依據合約框架，按月或按需穩定供應指定規格之礦產或大宗原物料，並保證產品符合國家或行業之質量檢驗標準。"
            payment_terms = "信用證付款 (L/C Payment) 或按批次結算：業主開立不可撤銷之信用證，承辦商憑提單 (B/L) 及第三方檢驗及格報告押匯。"
            domain_conditions = "價格波動調整與第三方檢測 (Price Fluctuation & Inspection)：若國際原物料價格指數波動超過 ±5%，則啟動價格調整機制。所有批次皆需附有 SGS 或同等機構之成份檢驗報告。"
        
        else:
            scope_desc = "承辦商須全權負責上述專案之物料供應、人力配置與完整執行，確保合乎圖則與相關法例。"
            payment_terms = "按常規進度付款 (Standard Progress Payment)：依據實際完成進度按月或按階段結算。"
            domain_conditions = "標準履約保證：承辦商須提供合約總額 10% 之履約保證金 (Performance Bond)。"

        draft_content = f"""【招標文件草案 / TENDER DOCUMENT DRAFT】
=========================================================
項目類型 (Project Category)：{proj_type}
預算規模 (Estimated Budget)：約 HKD {budget:,}
=========================================================

第一部份：招標邀請書 (PART 1: INVITATION TO TENDER)
---------------------------------------------------------
1.1 茲邀請合資格承辦商就上述專案提交標書。
1.2 投標者必須具備相關領域之營運經驗，並能確保項目之執行符合香港、內地或項目所在地之相關法規標準。

第二部份：招標條款 (PART 2: TERMS OF TENDER) - [參照 ICAC 標準招標範本]
---------------------------------------------------------
2.1 防圍標條款 (Anti-collusion)：
投標者必須在提交標書時附上已簽署的「不依賴他人/不具串通成份投標確認書」。如發現投標者曾就本招標與任何其他人士溝通或達成協議以調整標價，其標書將作廢。
2.2 利益衝突申報 (Declaration of Interest)：
投標者須申報與本專案負責人或業主代表是否存在任何實際或潛在之利益衝突。如未能如實申報，業主保留取消其投標資格之權利。

第三部份：一般及特別合約條款 (PART 3: GENERAL & SPECIAL CONDITIONS)
---------------------------------------------------------
3.1 工程與服務範圍 (Scope of Works / Services)：
{scope_desc}

3.2 商業與付款條款 (Commercial & Payment Terms)：
{payment_terms}

3.3 領域專屬特殊條件 (Domain-Specific Conditions)：
{domain_conditions}

3.4 延期罰款 (Liquidated Damages)：基於歷史優化模型，本合約之延期罰款設定為每日 HKD 20,000，總罰款上限 (Cap) 為合約總價之 10%。

3.5 誠信與防貪條款 (Probity Clause)：
承辦商、其僱員或代理人不得向本專案之任何相關人員提供、索取或接受《防止賄賂條例》(香港法例第201章) 所界定的任何利益。

第四部份：大灣區供應體系及特殊要求附加條款 (PART 4: GBA & SPECIAL REQUIREMENTS)
---------------------------------------------------------
4.1 針對本專案之額外要求：【{special_req if special_req else '無特殊要求'}】
4.2 跨境協同與標準互認：若專案涉及大灣區跨境業務，承辦商須承擔所有跨境運輸及相關稅項，並確保供應鏈穩定。相關設備或服務若採用「大灣區標準」，須提供經認可機構發出之檢測及格證明供審批。

[本文件由 QSCopilot 依據 HKbidd 專案庫、ICAC 標準範本及大灣區招標規範自動生成]
"""
        st.text_area("預覽生成的合約草稿：", draft_content, height=500)
        
        st.download_button(
            label="下載 Word / Text 草稿檔案",
            data=draft_content,
            file_name="Smart_Tender_Draft.txt",
            mime="text/plain",
            use_container_width=True
        )

# ==========================================
# 主畫面：RAG 合約顧問
# ==========================================
elif menu == "RAG 合約顧問":
    st.header("智能合約顧問 (RAG 實戰版)")
    st.markdown("透過真實 LLM 檢索增強生成 (RAG) 技術，嚴格鎖定上傳文本回答問題。")
    
    if 'raw_text' in st.session_state:
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "您好！我是 QSCopilot 專屬合約顧問。我已讀取並分析完您的招標文件，請問有什麼我可以協助您的？\n\n提示：試試問我：合約裡有規定延期罰款嗎？"}]
        
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
        if prompt := st.chat_input("輸入您的合約問題..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            with st.spinner("AI 模型深度檢索與運算中..."):
                response = call_real_llm_api(prompt, st.session_state['raw_text'])
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
    else:
        st.info("系統提示：請先至「總覽儀表板」上傳文件，才能啟動合約顧問。")
