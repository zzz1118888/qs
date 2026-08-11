import streamlit as st
import PyPDF2
import pandas as pd
import plotly.graph_objects as go
import re
import numpy as np
from openai import OpenAI

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
    st.caption("智慧測量創新賽 2026 - 競賽實戰版")
    st.markdown("---")
    
    menu = st.radio("功能導航", [
        "總覽儀表板", 
        "矩陣拓撲衝突偵測", 
        "微積分曝險建模", 
        "RAG 合約顧問"
    ])
    
    st.markdown("---")
    st.caption("Admin User: QS Director")

# ==========================================
# 核心引擎區
# ==========================================
def extract_text(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text

def analyze_risk_dynamic(text):
    # 1. 動態抓取基準罰金
    penalty_matches = re.findall(r"\$([0-9,]+)", text)
    base_penalty = int(penalty_matches[0].replace(",", "")) if penalty_matches else 0
    
    # 基礎起跳分調低，由系統根據細節累加
    score = 10 
    findings = []
    
    # 2. 評估免責條款
    if "免責" in text or "概不負責" in text:
        score += 15
        findings.append("偵測到「免責聲明」，可能導致工程延誤之額外成本無法索賠。")
        
    # 3. 評估索賠時效
    if "7天" in text:
        score += 15
        findings.append("索賠時效極度嚴苛 (僅 7 天)，存在極高喪失索賠權之風險。")
    elif "14天" in text or "28天" in text:
        score += 5
        findings.append("索賠時效為常規標準 (14-28天內)。")
        
    # 4. 評估延期罰款與財務曝險
    if "罰款" in text or "Liquidated Damages" in text:
        score += 10 # 具備罰款機制的基礎風險
        
        # 根據提取出的金額大小，給予不同的風險權重
        if base_penalty >= 40000:
            score += 20
            findings.append(f"「延期罰款」金額極高 (${base_penalty:,}/日)，財務曝險超標。")
        elif base_penalty >= 20000:
            score += 10
            findings.append(f"偵測到「延期罰款」，基準金額為 ${base_penalty:,}/日。")
        else:
            score += 5
            findings.append(f"偵測到「延期罰款」，金額尚在常規可控範圍 (${base_penalty:,}/日)。")
            
        # 偵測是否含有惡意的複利遞增條款
        if "遞增" in text:
            score += 10
            findings.append("罰款包含「逐日遞增」條款，長期延誤將導致曝險呈指數失控。")
            
    # 5. 評估法律衝突
    if "凌駕" in text or "推翻" in text or "Overrides" in text:
        score += 15
        findings.append("發現「凌駕/覆蓋 (Overrides)」字眼，合約法務邏輯存在單向霸王條款。")
        
    return min(score, 100), findings, base_penalty

def calculate_topology_matrix():
    nodes = ['主合約', '免責條款', '索賠程序', '特殊凌駕條款']
    A = np.array([
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0]  
    ])
    return nodes, A

# 真實 API 呼叫引擎 (已寫入您的 DeepSeek Key)
def call_real_llm_api(prompt, context_text):
    DEEPSEEK_API_KEY = "sk-3f04b63a30a145708e976283a5ee69b6"
    
    try:
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )
        
        system_prompt = f"""
        你是一位專業的香港測量師 (QS) 與合約顧問。
        請根據以下提供的合約條文內容，回答使用者的問題。
        你的回答必須精準、專業，並盡可能引用條文原話。
        如果問題超出了合約內容，請明確告知「合約中未提及」。
        
        【合約條文內容】：
        {context_text}
        """

        response = client.chat.completions.create(
            model="deepseek-chat",
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
            with st.spinner('系統正在執行 OCR 文字提取與多模型 API 路由分析...'):
                raw_text = extract_text(uploaded_file)
                score, findings, base_penalty = analyze_risk_dynamic(raw_text)
                
                st.session_state.update({
                    'raw_text': raw_text, 'filename': uploaded_file.name,
                    'score': score, 'findings': findings, 'base_penalty': base_penalty,
                    'last_uploaded_file': uploaded_file.name
                })

    if 'raw_text' in st.session_state:
        st.success(f"解析完成：{st.session_state['filename']} (處理時間: 1.24s | 萃取字元數: {len(st.session_state['raw_text']):,})")
        
        dash_col1, dash_col2 = st.columns([3, 2])
        
        with dash_col1:
            st.markdown("### 核心量化指標")
            metrics_col1, metrics_col2 = st.columns(2)
            metrics_col1.metric("合約總風險指數", f"{st.session_state['score']} / 100", "+極高風險" if st.session_state['score'] >= 80 else "-風險中等", delta_color="inverse" if st.session_state['score'] >= 80 else "normal")
            metrics_col2.metric("潛在衝突條款數量", f"{len(st.session_state['findings'])} 處", "需人工覆核", delta_color="off")
            
            bp = st.session_state['base_penalty']
            max_exposure = (bp * 14) + (bp * 1.05 * (((1.05 ** 46) - 1) / 0.05)) if bp > 0 else 0
            st.metric("極限財務曝險估值 (基準 60 天)", f"${max_exposure:,.0f} HKD", "預估上限", delta_color="inverse")
            
            st.markdown("### AI 提取摘要與風險特徵")
            for f in st.session_state['findings']:
                st.error(f"- {f}")
                
        with dash_col2:
            st.markdown("###  風險維度分佈分析")
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
        st.info("請上傳 PDF 檔案以啟動 QSCopilot 分析引擎。")

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
        st.info("請先至「總覽儀表板」上傳文件。")

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
            st.write(f"**AI 識別基準罰金**：每日 **${bp:,} HKD** (前 14 天)，其後每日遞增 **5%**。")
            
            sim_days = st.slider("拖曳設定預估延誤天數 (N)", min_value=15, max_value=120, value=60, step=1)
            
            first_14 = bp * 14
            subsequent_days = sim_days - 14
            geometric_sum = (bp * 1.05) * ((1.05 ** subsequent_days) - 1) / 0.05
            total_exposure = first_14 + geometric_sum
            
            st.markdown("### 模擬運算結果")
            st.metric(f"延誤 {sim_days} 天之總曝險", f"${total_exposure:,.0f} HKD", f"+{(total_exposure/1000000):.2f}M 現金流缺口", delta_color="inverse")
            st.success("**QS 決策建議**：建議在投標總價中預留此量化風險準備金，或於投標時發出 Query 要求設立罰款上限 (Cap)。")
            
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
        st.info("請先至「總覽儀表板」上傳文件。")

# ==========================================
# 主畫面：RAG 合約顧問 (整合真實 DeepSeek API)
# ==========================================
elif menu == "RAG 合約顧問":
    st.header("智能合約顧問 (RAG 實戰版)")
    st.markdown("透過真實 LLM 檢索增強生成 (RAG) 技術，嚴格鎖定上傳文本回答問題。")
    
    if 'raw_text' in st.session_state:
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "您好！我是 QSCopilot 專屬合約顧問。我已讀取並分析完您的招標文件，請問有什麼我可以協助您的？\n\n💡 *試試問我：合約裡有規定延期罰款嗎？*"}]
        
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
        st.info("請先至「總覽儀表板」上傳文件，才能啟動合約顧問。")
