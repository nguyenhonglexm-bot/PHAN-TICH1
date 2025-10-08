import streamlit as st
from docx import Document
import pandas as pd
import numpy as np
import io
import re
# Import thư viện Gemini/Google AI
from google import genai
from google.genai.errors import APIError

# Thay thế bằng API key thực tế của bạn
# Khuyến nghị: Sử dụng st.secrets để lưu API key trong môi trường production
# client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
# Giả định: Sử dụng biến môi trường hoặc key trực tiếp cho ví dụ này
try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except AttributeError:
    # Fallback nếu không dùng st.secrets (Chỉ dùng cho mục đích Demo/Local)
    st.error("Lỗi: Không tìm thấy GEMINI_API_KEY. Vui lòng thiết lập key trong st.secrets hoặc biến môi trường.")
    client = None

# --- Hàm tiện ích ---

def extract_text_from_docx(uploaded_file):
    """Đọc và trích xuất toàn bộ văn bản từ file Word đã tải lên."""
    try:
        doc = Document(uploaded_file)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        st.error(f"Lỗi khi đọc file Word: {e}")
        return None

def parse_and_clean_number(text):
    """Chuyển đổi văn bản thành số, loại bỏ ký tự không phải số/dấu phẩy/dấu chấm."""
    if isinstance(text, (int, float)):
        return text
    if not text:
        return 0.0

    # Xử lý các định dạng tiền tệ phổ biến (tỷ, triệu, k)
    text = text.lower().replace(',', '.')
    multiplier = 1.0

    if 'tỷ' in text or 't' in text:
        multiplier = 1_000_000_000
    elif 'triệu' in text or 'tr' in text:
        multiplier = 1_000_000
    elif 'nghìn' in text or 'k' in text:
        multiplier = 1_000

    # Lọc chỉ giữ lại số và dấu chấm (dùng cho số thập phân)
    cleaned_text = re.sub(r'[^\d.]', '', text)
    try:
        number = float(cleaned_text)
        return number * multiplier
    except ValueError:
        return 0.0

def calculate_financial_metrics(initial_investment, ncf_yearly, wacc, project_life):
    """Tính toán NPV, IRR, PP, DPP."""
    cash_flows = [-initial_investment] + [ncf_yearly] * project_life
    wacc_rate = wacc / 100.0

    # 1. NPV
    npv = np.npv(wacc_rate, cash_flows)

    # 2. IRR
    try:
        irr = np.irr(cash_flows)
    except Exception:
        irr = np.nan

    # 3. PP (Payback Period - Thời gian hoàn vốn)
    pp = initial_investment / ncf_yearly

    # 4. DPP (Discounted Payback Period - Thời gian hoàn vốn có chiết khấu)
    cumulative_discounted_cf = 0
    dpp = project_life
    remaining_investment = initial_investment
    
    # Tính dòng tiền chiết khấu
    discounted_cash_flows = [ncf_yearly / ((1 + wacc_rate)**t) for t in range(1, project_life + 1)]

    for t, dcf in enumerate(discounted_cash_flows):
        remaining_investment -= dcf
        if remaining_investment <= 0:
            # Năm hoàn vốn: t + 1 (vì t bắt đầu từ 0)
            # Hoàn vốn trong năm (t+1):
            dpp = (t + 1) + (remaining_investment + dcf) / dcf
            break

    return npv, irr, pp, dpp

# --- Định nghĩa Prompt cho AI (Rất quan trọng) ---

PROMPT_TEMPLATE = """
Bạn là một trợ lý tài chính chuyên nghiệp. Nhiệm vụ của bạn là phân tích văn bản dự án kinh doanh được cung cấp dưới đây và trích xuất các thông số tài chính chính.

Văn bản dự án:
---
{text_content}
---

Hãy trả lời chỉ bằng một chuỗi JSON HỢP LỆ (Không có bất kỳ ký tự nào khác ngoài JSON) với cấu trúc sau, đảm bảo các giá trị được đưa ra dưới dạng số (nếu là tiền tệ, không cần đơn vị):

{{
  "Vốn đầu tư": <Tổng vốn đầu tư ban đầu>,
  "Dòng đời dự án (năm)": <Số năm hoạt động của dự án>,
  "Doanh thu hàng năm": <Mức doanh thu hàng năm>,
  "Chi phí hàng năm": <Mức chi phí hoạt động hàng năm>,
  "WACC (%)": <Tỷ lệ WACC/Chi phí vốn bình quân>,
  "Thuế suất (%)": <Tỷ lệ thuế thu nhập doanh nghiệp>
}}

Nếu không tìm thấy thông số nào, hãy điền giá trị 0.
"""

# --- Giao diện Streamlit ---

st.set_page_config(
    page_title="Đánh giá Dự án Kinh doanh",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Ứng dụng Đánh giá Dự án Kinh doanh bằng AI 🤖")
st.markdown("---")

# Khởi tạo session state
if 'project_data' not in st.session_state:
    st.session_state.project_data = None
if 'file_content' not in st.session_state:
    st.session_state.file_content = None

# --- SIDEBAR: Tải File và Cấu hình ---
with st.sidebar:
    st.header("1. Tải File & Trích xuất")
    uploaded_file = st.file_uploader(
        "Tải file Word (.docx) chứa Phương án Kinh doanh",
        type="docx"
    )

    if uploaded_file:
        st.session_state.file_content = extract_text_from_docx(uploaded_file)
        st.success("Tải file thành công!")

    if st.session_state.file_content and client:
        if st.button("LỌC DỮ LIỆU TỪ AI", type="primary"):
            with st.spinner("AI đang phân tích và trích xuất thông số..."):
                try:
                    # Gửi prompt tới Gemini
                    prompt = PROMPT_TEMPLATE.format(text_content=st.session_state.file_content)
                    
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                        config={"response_mime_type": "application/json"}
                    )
                    
                    # Chuyển đổi JSON thành dict
                    data_dict = pd.read_json(io.StringIO(response.text), typ='series').to_dict()
                    
                    # Chuẩn hóa dữ liệu
                    st.session_state.project_data = {
                        "Vốn đầu tư": parse_and_clean_number(data_dict.get("Vốn đầu tư")),
                        "Dòng đời dự án (năm)": int(parse_and_clean_number(data_dict.get("Dòng đời dự án (năm)"))),
                        "Doanh thu hàng năm": parse_and_clean_number(data_dict.get("Doanh thu hàng năm")),
                        "Chi phí hàng năm": parse_and_clean_number(data_dict.get("Chi phí hàng năm")),
                        "WACC (%)": parse_and_clean_number(data_dict.get("WACC (%)")),
                        "Thuế suất (%)": parse_and_clean_number(data_dict.get("Thuế suất (%)")),
                    }
                    st.success("Trích xuất thông số hoàn tất!")
                
                except APIError as e:
                    st.error(f"Lỗi API Gemini: {e}")
                except Exception as e:
                    st.error(f"Lỗi trong quá trình AI xử lý/phân tích: {e}")
    elif uploaded_file and not client:
        st.warning("Không thể sử dụng chức năng AI do thiếu API Key.")

# --- MAIN CONTENT ---

# 1. Hiển thị thông tin đã lọc
if st.session_state.project_data:
    st.header("2. Thông Tin Dự Án Đã Lọc")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("💰 Vốn Đầu Tư Ban Đầu", f"{st.session_state.project_data['Vốn đầu tư']:,.0f} VNĐ")
        st.metric("📅 Dòng Đời Dự Án", f"{st.session_state.project_data['Dòng đời dự án (năm)']} năm")
        st.metric("💸 Chi Phí Hàng Năm", f"{st.session_state.project_data['Chi phí hàng năm']:,.0f} VNĐ")
        
    with col2:
        st.metric("💵 Doanh Thu Hàng Năm", f"{st.session_state.project_data['Doanh thu hàng năm']:,.0f} VNĐ")
        st.metric("⚖️ WACC (Chi phí vốn)", f"{st.session_state.project_data['WACC (%)']:.2f} %")
        st.metric("🧾 Thuế Suất TNDN", f"{st.session_state.project_data['Thuế suất (%)']:.0f} %")

    # Tính toán NCF hàng năm
    VON_DAU_TU = st.session_state.project_data["Vốn đầu tư"]
    DONG_DOI = st.session_state.project_data["Dòng đời dự án (năm)"]
    DOANH_THU = st.session_state.project_data["Doanh thu hàng năm"]
    CHI_PHI = st.session_state.project_data["Chi phí hàng năm"]
    WACC = st.session_state.project_data["WACC (%)"]
    THUE_SUAT = st.session_state.project_data["Thuế suất (%)"] / 100.0

    # Giả định khấu hao đường thẳng
    try:
        KHAU_HAO = VON_DAU_TU / DONG_DOI
    except ZeroDivisionError:
        KHAU_HAO = 0

    EBIT = DOANH_THU - CHI_PHI - KHAU_HAO
    THUE = EBIT * THUE_SUAT if EBIT > 0 else 0
    EAT = EBIT - THUE
    NCF_NAM = EAT + KHAU_HAO
    
    st.metric("Net Cash Flow (NCF) Hàng năm", f"{NCF_NAM:,.0f} VNĐ", delta_color="normal")
    
    st.markdown("---")

    # 2. Bảng Dòng Tiền
    st.header("3. Bảng Dòng Tiền Dự Án (Cash Flow)")
    
    if DONG_DOI > 0:
        # Tạo DataFrame
        years = list(range(0, DONG_DOI + 1))
        cash_flow_data = {
            "Năm": years,
            "Doanh Thu": [0] + [DOANH_THU] * DONG_DOI,
            "Chi Phí": [0] + [CHI_PHI] * DONG_DOI,
            "Khấu Hao": [0] + [KHAU_HAO] * DONG_DOI,
            "EBIT (LN trước thuế & lãi)": [0] + [EBIT] * DONG_DOI,
            "Thuế TNDN": [0] + [THUE] * DONG_DOI,
            "EAT (LN sau thuế)": [0] + [EAT] * DONG_DOI,
            "Dòng Tiền Thuần (NCF)": [-VON_DAU_TU] + [NCF_NAM] * DONG_DOI,
            "NCF Cộng Dồn": [0] * (DONG_DOI + 1),
            "NCF Chiết Khấu": [0] * (DONG_DOI + 1),
            "NCF Chiết Khấu Cộng Dồn": [0] * (DONG_DOI + 1),
        }
        
        df_cf = pd.DataFrame(cash_flow_data)
        
        # Tính toán cộng dồn và chiết khấu
        df_cf.loc[0, "NCF Cộng Dồn"] = -VON_DAU_TU
        for i in range(1, DONG_DOI + 1):
            df_cf.loc[i, "NCF Cộng Dồn"] = df_cf.loc[i-1, "NCF Cộng Dồn"] + df_cf.loc[i, "Dòng Tiền Thuần (NCF)"]
            
            discount_factor = 1 / ((1 + WACC / 100.0)**i)
            df_cf.loc[i, "NCF Chiết Khấu"] = df_cf.loc[i, "Dòng Tiền Thuần (NCF)"] * discount_factor
            
            df_cf.loc[i, "NCF Chiết Khấu Cộng Dồn"] = df_cf.loc[i-1, "NCF Chiết Khấu Cộng Dồn"] + df_cf.loc[i, "NCF Chiết Khấu"]

        df_cf.loc[0, "NCF Chiết Khấu Cộng Dồn"] = -VON_DAU_TU # Gán lại vốn đầu tư
        
        # Hiển thị bảng
        st.dataframe(df_cf.style.format(
            {col: "{:,.0f}" for col in df_cf.columns if col not in ["Năm"]}
        ), use_container_width=True)
    else:
        st.warning("Dòng đời dự án phải lớn hơn 0 để xây dựng bảng dòng tiền.")

    st.markdown("---")

    # 3. Tính toán các chỉ số
    st.header("4. Các Chỉ Số Đánh Giá Hiệu Quả")

    if DONG_DOI > 0:
        NPV, IRR, PP, DPP = calculate_financial_metrics(VON_DAU_TU, NCF_NAM, WACC, DONG_DOI)

        # Định dạng kết quả
        npv_text = f"{NPV:,.0f} VNĐ"
        irr_text = f"{IRR*100:.2f} %" if not np.isnan(IRR) else "Không xác định"
        pp_text = f"{PP:.2f} năm"
        dpp_text = f"{DPP:.2f} năm"

        col3, col4, col5, col6 = st.columns(4)
        
        col3.metric("Net Present Value (NPV)", npv_text, delta_color="off")
        col4.metric("Internal Rate of Return (IRR)", irr_text, delta_color="off")
        col5.metric("Payback Period (PP)", pp_text, delta_color="off")
        col6.metric("Discounted Payback Period (DPP)", dpp_text, delta_color="off")

        st.session_state.financial_metrics = {
            "NPV": NPV, "IRR": IRR, "PP": PP, "DPP": DPP,
            "WACC": WACC, "DONG_DOI": DONG_DOI
        }
    
    st.markdown("---")

    # 4. Phân tích của AI
    st.header("5. Phân Tích Chuyên Sâu của AI")

    if st.session_state.project_data and client:
        if st.button("YÊU CẦU AI PHÂN TÍCH HIỆU QUẢ", key="analyze_button"):
            with st.spinner("AI đang phân tích các chỉ số tài chính..."):
                try:
                    # Tạo prompt chi tiết cho AI
                    analysis_prompt = f"""
                    Bạn là một nhà phân tích tài chính cao cấp. Hãy đánh giá hiệu quả của dự án kinh doanh này dựa trên các chỉ số sau (Đơn vị tiền tệ là VNĐ):

                    - Vốn Đầu Tư Ban Đầu: {VON_DAU_TU:,.0f}
                    - Dòng đời dự án: {DONG_DOI} năm
                    - WACC (Chi phí vốn): {WACC:.2f}%
                    - NPV: {NPV:,.0f}
                    - IRR: {IRR*100:.2f}%
                    - PP (Hoàn vốn): {PP:.2f} năm
                    - DPP (Hoàn vốn chiết khấu): {DPP:.2f} năm

                    Phân tích của bạn cần tập trung vào:
                    1. Đánh giá chung: Dự án có khả thi về mặt tài chính không? (Dựa trên NPV và so sánh IRR với WACC).
                    2. Rủi ro về thời gian: Thời gian hoàn vốn có phù hợp với dòng đời dự án và rủi ro không?
                    3. Kiến nghị: Đề xuất hành động tiếp theo (Chấp nhận, Từ chối, hoặc Yêu cầu điều chỉnh).
                    4. Trình bày dưới dạng văn bản chuyên nghiệp.
                    """

                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=analysis_prompt
                    )
                    st.success("Phân tích hoàn tất!")
                    st.info(response.text)

                except APIError as e:
                    st.error(f"Lỗi API Gemini: {e}")
                except Exception as e:
                    st.error(f"Lỗi khi AI thực hiện phân tích: {e}")
    elif st.session_state.project_data:
         st.warning("Không thể thực hiện phân tích chuyên sâu của AI do thiếu API Key.")
