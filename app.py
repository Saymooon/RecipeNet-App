import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import joblib
import re
import math
from typing import List, Dict, Optional
from skimage import color # requirements.txt에 'scikit-image'가 있어야 합니다.
import matplotlib.pyplot as plt
from datetime import datetime # DATE 표시를 위해 import
from io import BytesIO # 엑셀 다운로드를 위해 import

# ==========================================================
# 0. CONFIG (Jupyter Notebook에서 정확하게 복사)
# ==========================================================
CONFIG = {
    'embed_dim': 64, # ⭐️ 모델 뼈대 생성을 위해 필수

    # 필수 컬럼 매핑
    'condition_col': 'COLOR',
    'name_col':      'COLOR',
    'lab_cols':      ['L*(10°/D65)', 'a*(10°/D65)', 'b*(10°/D65)'],
    'total_col' : "TOTAL_LOAD",

    # 스펙트럼 사용 (수동 지정)
    'spectrum_prefixes': [],
    'spectrum_cols':   ['400[nm]', '410[nm]', '420[nm]', '430[nm]', '440[nm]', '450[nm]',
       '460[nm]', '470[nm]', '480[nm]', '490[nm]', '500[nm]', '510[nm]',
       '520[nm]', '530[nm]', '540[nm]', '550[nm]', '560[nm]', '570[nm]',
       '580[nm]', '590[nm]', '600[nm]', '610[nm]', '620[nm]', '630[nm]',
       '640[nm]', '650[nm]', '660[nm]', '670[nm]', '680[nm]', '690[nm]',
       '700[nm]'],

    # 레시피(56 안료)
    'recipe_cols': [
       '1/10 BLUE 2000/S100', '1/10 BROWN 3001/S100', '1/10 CARBON',
       '1/10 CARBON/S100', '1/10 CO BLUE/R350', '1/10 CO BLUE/S100',
       '1/10 GREEN K8730/S100', '1/10 MK4535/R350', '1/10 MK4535/S100',
       '1/10 RED B/S100', '1/10 YELLOW300/S100', '1/100 BLUE7000/S100',
       '1/100 CARBON', '1/100 CARBON/S100', '1/100 MK4535/S100',
       '1/100 RED B/R350', '1/100 RED B/S100', '1/100 YELLOW 300/S100',
       '1/50 CARBON/S100', '1/5000 CARBON/S100', '10550 BROWN', '2000 BLUE',
       '214 BLUE', '23 VIOLET', '7000 BLUE', 'BLUE 424', 'BROWN 216',
       'BROWN 3001', 'CARBON', 'CO BLUE', 'GREEN 9361', 'GREEN K8730',
       'HQ BLUE', 'HQ GREEN', 'HQ MAGENTA', 'HQ ORANGE', 'HQ ORANGE+RED',
       'HQ ORANGE+YELLOW', 'HQ PINK', 'HQ RED', 'HQ VIOLET', 'HQ YELLOW',
       'MK 4535', 'ORANGE K2890(2G)', 'ORANGE K2960', 'RED B', 'RED BNP',
       'RED K3840', 'RED K4035(2B)', 'TIO2-R350', 'VIOLET 21', 'VIOLET 42',
       'YELLOW 10401', 'YELLOW 300', 'YELLOW H3R', 'YELLOW NG'
    ],

    "tio2_name": "TIO2-R350",
}


# ==========================================================
# 1. 텍스트 인코더 클래스 정의 (SimpleNameEncoder)
# ==========================================================
class SimpleNameEncoder:
    def __init__(self, max_tokens: int = 512, embed_dim: int = 64, seed: int = 42):
        self.max_tokens = max_tokens
        self.embed_dim = embed_dim
        self.seed = seed
        self.token2id: Dict[str, int] = {}
        self.id2token: List[str] = []
        self.emb: Optional[np.ndarray] = None
        self._tok_pat = re.compile(r"[A-Za-z0-9가-힣\+\-_/]+")
    def _tokenize(self, s: str) -> List[str]:
        if not isinstance(s, str): return []
        s = s.strip().lower()
        return self._tok_pat.findall(s)
    def fit(self, names: List[str]):
        from collections import Counter
        cnt = Counter()
        for n in names:
            cnt.update(self._tokenize(n))
        most = cnt.most_common(self.max_tokens)
        self.id2token = [t for t, _ in most]
        self.token2id = {t:i for i, t in enumerate(self.id2token)}
        rng = np.random.default_rng(self.seed)
        self.emb = rng.standard_normal(size=(len(self.id2token), self.embed_dim)).astype(np.float32) / math.sqrt(self.embed_dim)
    def encode(self, names: List[str]) -> np.ndarray:
        assert self.emb is not None, "Encoder가 fit되지 않았거나, pkl 파일이 잘못 로드되었습니다."
        X = np.zeros((len(names), self.embed_dim), dtype=np.float32)
        for i, n in enumerate(names):
            toks = self._tokenize(n)
            idxs = [self.token2id[t] for t in toks if t in self.token2id]
            if idxs: X[i] = self.emb[idxs].mean(axis=0)
        return X

# ==========================================================
# 2. PyTorch 모델 클래스 정의 (CrossAttentionBlock, RecipeNet3Head)
# ==========================================================
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.ReLU(),
            nn.Linear(d_model*2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x, context):
        h,_ = self.attn(x, context, context)
        x = self.norm(x+h)
        h = self.ff(x)
        x = self.norm2(x+h)
        return x

class RecipeNet3Head(nn.Module):
    def __init__(self, in_dim, num_pigments, d_model=128, nhead=4, nlayers=2):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, d_model)
        self.pigment_emb = nn.Parameter(torch.randn(num_pigments, d_model))
        self.layers = nn.ModuleList([CrossAttentionBlock(d_model,nhead) for _ in range(nlayers)])
        self.base_head = nn.Linear(d_model, 1)
        self.chroma_head = nn.Linear(d_model, num_pigments-1)
        self.total_head = nn.Linear(d_model, 1)
    def forward(self, x):
        B = x.size(0)
        q = self.proj_in(x).unsqueeze(1)
        context = self.pigment_emb.unsqueeze(0).expand(B,-1,-1)
        for layer in self.layers:
            q = layer(q, context)
        q = q.squeeze(1)
        b = torch.sigmoid(self.base_head(q))
        chroma = torch.softmax(self.chroma_head(q),dim=-1)
        total = F.relu(self.total_head(q))
        return b, chroma, total

# ==========================================================
# 3. 유틸리티 함수 (DeltaE, 색상 시각화, 엑셀 변환)
# ==========================================================
def lab_to_rgb(lab):
    """Lab -> RGB 변환 (skimage 활용)"""
    lab = np.array(lab).reshape(1,1,3)
    rgb = color.lab2rgb(lab)
    return rgb[0,0,:]

def show_color_patches(lab_true, lab_pred):
    """Streamlit용 색상 비교차트 생성 (True vs Pred)"""
    fig, ax = plt.subplots(1, 2, figsize=(6,3))
    rgb_true = lab_to_rgb(lab_true)
    rgb_pred = lab_to_rgb(lab_pred)
    ax[0].imshow([[rgb_true]]); ax[0].set_title("Target (True)"); ax[0].axis("off")
    ax[1].imshow([[rgb_pred]]); ax[1].set_title("Predicted (Surrogate)"); ax[1].axis("off")
    return fig

def show_single_color_patch(lab_color, title="Color"):
    """Streamlit용 단일 색상 차트 생성"""
    fig, ax = plt.subplots(figsize=(2.5, 1.8))
    rgb_color = lab_to_rgb(lab_color)
    ax.imshow([[rgb_color]])
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return fig

def deltaE_00(y_true, y_pred, kL=1, kC=1, kH=1):
    """CIEDE2000 DeltaE 계산"""
    L1, a1, b1 = y_true[:,0], y_true[:,1], y_true[:,2]
    L2, a2, b2 = y_pred[:,0], y_pred[:,1], y_pred[:,2]
    C1 = np.sqrt(a1*a1 + b1*b1); C2 = np.sqrt(a2*a2 + b2*b2)
    C_bar = 0.5 * (C1 + C2); C_bar7 = C_bar**7
    G = 0.5 * (1 - np.sqrt(C_bar7 / (C_bar7 + 25**7 + 1e-12)))
    a1p = (1+G)*a1; a2p = (1+G)*a2
    C1p = np.sqrt(a1p*a1p + b1*b1); C2p = np.sqrt(a2p*a2p + b2*b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0; h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    dLp = L2 - L1; dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(C1p*C2p==0,0.0,dhp); dhp = np.where(dhp>180,dhp-360,dhp); dhp = np.where(dhp<-180,dhp+360,dhp)
    dHp = 2*np.sqrt(C1p*C2p)*np.sin(np.radians(dhp)/2.0)
    Lp_bar = 0.5*(L1+L2); Cp_bar=0.5*(C1p+C2p)
    hp_sum = h1p+h2p
    hp_bar = np.where((C1p*C2p)==0,hp_sum, np.where(np.abs(h1p-h2p)>180,(hp_sum+360.0)/2.0-360.0*(hp_sum>=360.0),hp_sum/2.0))
    T=(1-0.17*np.cos(np.radians(hp_bar-30)) +0.24*np.cos(np.radians(2*hp_bar)) +0.32*np.cos(np.radians(3*hp_bar+6)) -0.20*np.cos(np.radians(4*hp_bar-63)))
    Sl=1+0.015*(Lp_bar-50)**2/np.sqrt(20+(Lp_bar-50)**2)
    Sc=1+0.045*Cp_bar; Sh=1+0.015*Cp_bar*T
    delta_theta=30*np.exp(-((hp_bar-275)/25)**2)
    Rc=2*np.sqrt(C_bar**7/(C_bar**7+25**7+1e-12))
    Rt=-Rc*np.sin(2*np.radians(delta_theta))
    dE00=np.sqrt((dLp/(kL*Sl))**2+(dCp/(kC*Sc))**2+(dHp/(kH*Sh))**2 +Rt*(dCp/(kC*Sc))*(dHp/(kH*Sh)))
    return dE00

# 엑셀 다운로드를 위한 헬퍼 함수
def to_excel_with_header(recipe_df, color_name, date_str):
    """Pandas DataFrame(레시피)과 헤더 정보(색상명, 날짜)를 엑셀 파일(BytesIO)로 변환합니다."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 1. 헤더 정보 DataFrame 생성
        header_data = {'Info': ['COLOR', 'DATE'], 'Value': [color_name, date_str]}
        header_df = pd.DataFrame(header_data)
        # 헤더 DataFrame 쓰기 (인덱스 및 헤더 제외)
        header_df.to_excel(writer, index=False, header=False, sheet_name='Predicted_Recipe', startrow=0)

        # 2. 레시피 DataFrame 쓰기 (헤더 포함, 헤더 정보 아래에 위치)
        recipe_df.to_excel(writer, index=False, header=True, sheet_name='Predicted_Recipe', startrow=3)

        # (선택사항) 컬럼 너비 자동 조절
        worksheet = writer.sheets['Predicted_Recipe']
        for idx, col in enumerate(recipe_df): # recipe_df의 컬럼 기준
            series = recipe_df[col]
            max_len = max((
                series.astype(str).map(len).max(),
                len(str(series.name))
            )) + 2 # 약간의 여유 공간
            worksheet.column_dimensions[chr(65 + idx)].width = max_len # A열부터 시작
        # 헤더 정보 컬럼 너비 수동 조절 (필요시)
        # worksheet.column_dimensions['A'].width = 15 # Info 컬럼
        # worksheet.column_dimensions['B'].width = 30 # Value 컬럼

    processed_data = output.getvalue()
    return processed_data

# ==========================================================
# 4. 추론 함수 (test_new_swatch -> Streamlit용으로 수정)
# ==========================================================
def run_inference(model, cfg, surrogate, spectrum, lab, color_name, name_encoder):
    """
    Streamlit 입력값을 받아 예측을 수행하고 결과를 출력/반환합니다.
    레시피는 0.01 g/K 이상 값만 표시하며, 6개를 초과하면 상위 6개만 표시합니다.
    다운로드 엑셀은 화면 표시 내용과 동일하며, COLOR/DATE 정보를 포함합니다.
    """
    device = torch.device("cpu") # Streamlit Cloud는 CPU 기반
    model = model.to(device)
    model.eval()

    # ---- 텍스트 임베딩
    X_text = name_encoder.encode([color_name]) # shape (1, embed_dim)

    # ---- 입력 feature
    feat = np.hstack([spectrum, lab, X_text[0]])
    xb = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(device)

    # ---- 레시피 예측
    with torch.no_grad():
        b, q, t = model(xb)
        others = (1-b)*q
        chunks=[];k=0
        for j in range(len(cfg['recipe_cols'])):
            if j == cfg['recipe_cols'].index(cfg['tio2_name']):
                chunks.append(b)
            else:
                chunks.append(others[:,k:k+1]); k+=1
        p = torch.cat(chunks,dim=1)
        P_g = (p*t).cpu().numpy() # g 단위 (g/K로 가정)

    # ---- Surrogate 예측
    lab_pred = surrogate.predict(P_g)

    # ---- ΔE00
    lab_true = lab.reshape(1,3)
    de00 = deltaE_00(lab_true, lab_pred)

    # ---- Streamlit 출력 ----

    st.subheader("🔬 예측된 레시피")

    # --- 테이블 1: 정보 (COLOR, DATE 통합) ---
    current_date = datetime.now().strftime('%Y-%m-%d')
    st.markdown(f"""
    <style>
        .info-table {{ border-collapse: collapse; width: 60%; margin-bottom: 1rem; }}
        .info-table td, .info-table th {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        .info-table th {{ background-color: #f2f2f2; font-weight: bold; }}
    </style>
    <table class="info-table">
      <tr>
        <th>COLOR</th>
        <td>{color_name}</td>
        <th>DATE</th>
        <td>{current_date}</td>
      </tr>
    </table>
    """, unsafe_allow_html=True)

    st.write("") # 약간의 간격

    # --- 테이블 2: 안료 (PIGMENT, 함량) ---
    recipe_g_series = pd.Series(P_g.flatten(), index=cfg['recipe_cols'])

    # 1. 0.01 이상 필터링 & 내림차순 정렬
    recipe_filtered = recipe_g_series[recipe_g_series >= 0.01].sort_values(ascending=False)

    # 2. 표시할 레시피 결정 (상위 6개 또는 전체)
    if len(recipe_filtered) > 6:
        recipe_to_display_series = recipe_filtered.head(6)
        # st.caption(f"함량이 0.01 g/K 이상인 {len(recipe_filtered)}개의 안료 중 상위 6개만 표시됩니다.")
    else:
        recipe_to_display_series = recipe_filtered

    # 3. 화면 표시 및 다운로드용 데이터 준비
    if recipe_to_display_series.empty:
        st.warning("예측된 레시피 중 함량이 0.01 g/K 이상인 안료가 없습니다.")
        # 다운로드용 DataFrame도 비어 있게 만듦
        recipe_df_final = pd.DataFrame({'PIGMENT': [], '함량 (g/K)': []})
    else:
        # DataFrame으로 변환 (화면 표시 및 다운로드 공통 사용)
        recipe_df_final = pd.DataFrame({
            'PIGMENT': recipe_to_display_series.index,
            '함량 (g/K)': recipe_to_display_series.values
        }).reset_index(drop=True)

        # 소수점 4자리까지만 화면에 표시
        st.dataframe(
            recipe_df_final.style.format({'함량 (g/K)': '{:.4f}'}),
            hide_index=True,
            use_container_width=True
        )

    # 엑셀 다운로드 버튼
    excel_data = to_excel_with_header(recipe_df_final, color_name, current_date)
    st.download_button(
        label="📄 표시된 레시피 엑셀 다운로드 (.xlsx)",
        data=excel_data,
        file_name=f'predicted_recipe_{color_name.replace(" ", "_")}.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )

    st.divider() # 가로줄 추가

    # --- 예측 결과 및 색상 비교 ---
    st.subheader("📊 예측 결과")

    col1_res, col2_res = st.columns(2)
    with col1_res:
        st.metric(label="Predicted ΔE00", value=f"{de00.mean():.3f}")
        st.write(f"**True Lab:** {np.round(lab_true.flatten(), 2)}")
        st.write(f"**Pred Lab:** {np.round(lab_pred.flatten(), 2)}")

    with col2_res:
        st.write("**색상 비교:**")
        fig = show_color_patches(lab_true.flatten(), lab_pred.flatten())
        st.pyplot(fig)


# ==========================================================
# 5. 모델 로드 (Streamlit 캐시 사용)
# ==========================================================
@st.cache_resource
def load_all_models(config):
    # ... (이전과 동일) ...
    device = torch.device("cpu")
    try: name_encoder = joblib.load("name_encoder.pkl")
    except FileNotFoundError: st.error("`name_encoder.pkl`..."); return None, None, None
    try: surrogate = joblib.load("xgb_surrogate_2.pkl")
    except FileNotFoundError: st.error("`xgb_surrogate_2.pkl`..."); return None, None, None
    try:
        in_dim = len(config['spectrum_cols']) + len(config['lab_cols']) + config['embed_dim']
        num_pigments = len(config['recipe_cols'])
        model = RecipeNet3Head(in_dim, num_pigments, d_model=128)
        model.load_state_dict(torch.load("recipe_model.pth", map_location=device))
        model.eval()
    except FileNotFoundError: st.error("`recipe_model.pth`..."); return None, None, None
    except Exception as e: st.error(f"PyTorch 모델 로드 오류: {e}..."); return None, None, None
    return model, name_encoder, surrogate

# ==========================================================
# 6. Streamlit UI (메인 앱 로직)
# ==========================================================

# --- 엑셀 파일 처리 함수 ---
@st.cache_data
def parse_excel(uploaded_file, config):
    # ... (이전과 동일, NaN 처리 포함) ...
    try:
        df = pd.read_excel(uploaded_file)
        filter_col = '정반사광 처리'; name_col = '데이터 이름'
        if filter_col not in df.columns: st.error(f"'{filter_col}' 없음"); return None
        sce_df = df[df[filter_col] == 'SCE'].copy()
        if sce_df.empty: st.error("'SCE' 행 없음"); return None
        if name_col not in sce_df.columns: st.error(f"'{name_col}' 없음"); return None
        sce_df['Color Name'] = sce_df[name_col].astype(str).str[4:].str.strip()
        required_cols = config['lab_cols'] + config['spectrum_cols']
        missing_cols = [col for col in required_cols if col not in sce_df.columns]
        if missing_cols: st.error(f"필수 컬럼 없음: {missing_cols}"); return None
        final_cols = ['Color Name'] + config['lab_cols'] + config['spectrum_cols']
        for col in config['lab_cols'] + config['spectrum_cols']:
             sce_df[col] = pd.to_numeric(sce_df[col], errors='coerce')
        if sce_df[final_cols].isnull().values.any():
            nan_rows = sce_df[sce_df[final_cols].isnull().any(axis=1)]['Color Name'].tolist()
            st.warning(f"숫자 오류 행 제외: {nan_rows}")
            sce_df = sce_df.dropna(subset=final_cols)
        if sce_df.empty: st.error("유효 'SCE' 행 없음"); return None
        return sce_df[final_cols].reset_index(drop=True)
    except Exception as e: st.error(f"엑셀 처리 오류: {e}"); return None

# --- 메인 UI ---
st.set_page_config(layout="wide")
st.title("🧪 레시피 예측 모델")

# 모델 로드
model, name_encoder, surrogate = load_all_models(CONFIG)

if model and name_encoder and surrogate:
    # st.success(f"모델 로드 완료! (안료 개수: {len(CONFIG['recipe_cols'])})")

    st.header("1. 목표 색상 정보 업로드")

    uploaded_file = st.file_uploader(
        "목표 색상 엑셀 파일 업로드 (xlsx)",
        type=["xlsx"],
        help="파일 내 '정반사광 처리' 컬럼의 'SCE' 행 데이터를 모두 불러옵니다."
    )

    # 세션 상태 초기화
    if 'sce_data' not in st.session_state: st.session_state.sce_data = None
    if 'selected_color' not in st.session_state: st.session_state.selected_color = None
    if 'prediction_output' not in st.session_state: st.session_state.prediction_output = None

    # 파일 업로드 시 처리
    if uploaded_file is not None:
        new_sce_data = parse_excel(uploaded_file, CONFIG)
        if new_sce_data is not None and not new_sce_data.empty:
            st.session_state.sce_data = new_sce_data
            # Selectbox의 기본값을 첫 번째 항목으로 설정
            st.session_state.selected_color = st.session_state.sce_data['Color Name'][0]
            # 새 파일 업로드 시 이전 예측 결과 삭제
            if 'prediction_output' in st.session_state: del st.session_state.prediction_output
        else: # 파싱 실패 시 초기화
             st.session_state.sce_data = None; st.session_state.selected_color = None
             if 'prediction_output' in st.session_state: del st.session_state.prediction_output

    # --- 데이터 선택 및 표시 ---
    if st.session_state.sce_data is not None:
        df_sce = st.session_state.sce_data
        if not df_sce.empty:
            st.header("2. 목표 색상 선택")
            selected_color_name_from_box = st.selectbox(
                f"'SCE' 기준 총 {len(df_sce)}개의 색상이 로드되었습니다. 예측할 색상을 선택하세요",
                options=df_sce['Color Name'], key='color_selector',
                index=list(df_sce['Color Name']).index(st.session_state.selected_color) if st.session_state.selected_color in list(df_sce['Color Name']) else 0
            )
            # Selectbox 변경 시 세션 상태 업데이트 및 예측 결과 초기화
            if st.session_state.selected_color != selected_color_name_from_box:
                 st.session_state.selected_color = selected_color_name_from_box
                 if 'prediction_output' in st.session_state: del st.session_state.prediction_output

            current_selected_color = st.session_state.selected_color
            if current_selected_color and current_selected_color in list(df_sce['Color Name']):
                selected_row = df_sce[df_sce['Color Name'] == current_selected_color].iloc[0]
                st.subheader(f"'{current_selected_color}' 데이터 확인")
                lab_true_np = selected_row[CONFIG['lab_cols']].values.astype(float)
                spectrum_true_np = selected_row[CONFIG['spectrum_cols']].values.astype(float)
                col1, col2, col3 = st.columns([0.45, 0.4, 0.15])
                with col1: # Lab 정보
                    st.write("**목표 색상 정보:**")
                    st.text_input("Color Name", value=current_selected_color, disabled=True, key=f"name_display_{current_selected_color}")
                    st.text_input(f"{CONFIG['lab_cols'][0]}", value=f"{lab_true_np[0]:.2f}", disabled=True, key=f"l_display_{current_selected_color}")
                    st.text_input(f"{CONFIG['lab_cols'][1]}", value=f"{lab_true_np[1]:.2f}", disabled=True, key=f"a_display_{current_selected_color}")
                    st.text_input(f"{CONFIG['lab_cols'][2]}", value=f"{lab_true_np[2]:.2f}", disabled=True, key=f"b_display_{current_selected_color}")
                with col2: # 스펙트럼 정보
                    st.write("**스펙트럼 정보:**")
                    spectrum_df = pd.DataFrame({'파장 (Wavelength)': CONFIG['spectrum_cols'], '값 (Value)': spectrum_true_np})
                    st.dataframe(spectrum_df, height=320)
                with col3: # 색상 시각화
                    st.write("**Target Color:**")
                    fig = show_single_color_patch(lab_true_np, title="Target (True)")
                    st.pyplot(fig)
                

                # --- 예측 버튼 ---
                st.header("3. 예측 실행")
                if st.button(f"🚀 '{current_selected_color}' 레시피 예측 실행", type="primary", key=f"predict_btn_{current_selected_color}"):
                    with st.spinner('모델 예측 중...'):
                        st.session_state.prediction_output = {
                             "model": model, "cfg": CONFIG, "surrogate": surrogate, # ⭐️ "cfg" 키 사용
                             "spectrum": spectrum_true_np, "lab": lab_true_np,
                             "color_name": current_selected_color, "name_encoder": name_encoder
                        }

    # --- 파일 없음 또는 초기화 ---
    elif uploaded_file is None:
        st.info("⬆️ 예측을 시작하려면 엑셀 파일을 업로드해주세요.")
        if 'sce_data' in st.session_state: del st.session_state.sce_data
        if 'selected_color' in st.session_state: del st.session_state.selected_color
        if 'prediction_output' in st.session_state: del st.session_state.prediction_output

    # --- 예측 결과 표시 ---
    if 'prediction_output' in st.session_state and st.session_state.prediction_output is not None:
         if st.session_state.selected_color == st.session_state.prediction_output['color_name']:
              output_args = st.session_state.prediction_output
              run_inference(**output_args)

# 모델 로드 실패 시
else:
    st.error("‼️ 모델 로딩 실패.")
    st.code("""
    [필수 파일 목록]
    1. app.py (지금 이 파일)
    2. recipe_model.pth
    3. name_encoder.pkl
    4. xgb_surrogate_2.pkl (⭐️ 이름 확인!)
    5. requirements.txt (openpyxl 포함 총 9줄)
    """)

