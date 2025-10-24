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
from datetime import datetime # ⭐️ [추가] DATE 표시를 위해 import

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
# 3. 유틸리티 함수 (DeltaE, 색상 시각화)
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
    return fig # ⭐️ st.pyplot()을 위해 fig 객체 반환

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
    C1 = np.sqrt(a1*a1 + b1*b1)
    C2 = np.sqrt(a2*a2 + b2*b2)
    C_bar = 0.5 * (C1 + C2); C_bar7 = C_bar**7
    G = 0.5 * (1 - np.sqrt(C_bar7 / (C_bar7 + 25**7 + 1e-12)))
    a1p = (1+G)*a1; a2p = (1+G)*a2
    C1p = np.sqrt(a1p*a1p + b1*b1); C2p = np.sqrt(a2p*a2p + b2*b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    dLp = L2 - L1; dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(C1p*C2p==0,0.0,dhp)
    dhp = np.where(dhp>180,dhp-360,dhp)
    dhp = np.where(dhp<-180,dhp+360,dhp)
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

# ==========================================================
# 4. 추론 함수 (test_new_swatch -> Streamlit용으로 수정)
# ==========================================================
# ⭐️ [수정됨] 이 함수 전체가 수정되었습니다.
def run_inference(model, cfg, surrogate, spectrum, lab, color_name, name_encoder):
    """
    Streamlit 입력값을 받아 예측을 수행하고 결과를 출력/반환합니다.
    """
    device = torch.device("cpu") # Streamlit Cloud는 CPU 기반
    model = model.to(device)
    model.eval()

    # ---- 텍스트 임베딩
    X_text = name_encoder.encode([color_name]) # shape (1, embed_dim)

    # ---- 입력 feature (스케일러 없음! Jupyter 학습 코드 기준)
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

    # ---- Streamlit 출력
    recipe_g_series = pd.Series(P_g.flatten(), index=cfg['recipe_cols'])
    
    # ⭐️ [UI 변경] 요청: 엑셀 스타일로 레시피 출력 (요청 2: 순서 변경)
    st.subheader("🔬 예측된 레시피")

    # --- 테이블 1: 정보 (COLOR, DATE) ---
    col1_info, col2_info, col_spacer = st.columns([0.4, 0.4, 0.2])
    with col1_info:
        # st.text_input("COLOR", value=color_name, disabled=True)
        st.markdown("**COLOR**")
        st.markdown(f"<div style='font-size: 1.25rem; font-weight: bold; border: 1px solid #eee; padding: 8px; border-radius: 0.25rem; background-color: #fafafa;'>{color_name}</div>", unsafe_allow_html=True)
    with col2_info:
        # st.text_input("DATE", value=datetime.now().strftime('%Y-%m-%d'), disabled=True)
        st.markdown("**DATE**")
        st.markdown(f"<div style='font-size: 1.25rem; font-weight: bold; border: 1px solid #eee; padding: 8px; border-radius: 0.25rem; background-color: #fafafa;'>{datetime.now().strftime('%Y-%m-%d')}</div>", unsafe_allow_html=True)

    st.divider() # 가로줄 추가

    # --- 테이블 2: 안료 (PIGMENT, 함량) ---
    
    # ⭐️ [필터링 로직 수정]
    # 1. 0.01 이상 필터링 & 내림차순 정렬
    recipe_filtered = recipe_g_series[recipe_g_series >= 0.01].sort_values(ascending=False)

    # 2. 표시할 레시피 결정 (상위 6개 또는 전체)
    if len(recipe_filtered) > 6:
        recipe_to_display = recipe_filtered.head(6)
        # st.caption(f"함량이 0.01 g/K 이상인 {len(recipe_filtered)}개의 안료 중 상위 6개만 표시됩니다.")
    else:
        recipe_to_display = recipe_filtered

    # 3. 화면 표시 및 다운로드용 데이터 준비
    if recipe_to_display.empty:
        st.warning("예측된 레시피 중 함량이 0.01 g/K 이상인 안료가 없습니다.")
        # 다운로드용 DataFrame도 비어 있게 만듦
        recipe_df_for_download = pd.DataFrame({'PIGMENT': [], '함량 (g/K)': []})
    else:
        # DataFrame으로 변환 (화면 표시용)
        recipe_df_display = pd.DataFrame({
            'PIGMENT': recipe_to_display.index,
            '함량 (g/K)': recipe_to_display.values
        }).reset_index(drop=True)

        # 소수점 4자리까지만 화면에 표시
        st.dataframe(
            recipe_df_display.style.format({'함량 (g/K)': '{:.4f}'}),
            hide_index=True,
            use_container_width=True
        )
        # 다운로드용 DataFrame은 0.01 이상 필터링된 전체 데이터 사용
        recipe_df_for_download = pd.DataFrame({
            'PIGMENT': recipe_filtered.index,
            '함량 (g/K)': recipe_filtered.head(6).values
        }).reset_index(drop=True)

    st.divider() # 가로줄 추가

    # ⭐️ [순서 변경] 예측 결과 및 색상 비교를 나중에 표시
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
    """
    앱 실행 시 한 번만 모델, 인코더, 서로게이트를 로드합니다.
    """
    device = torch.device("cpu")
    
    # 1. NameEncoder 로드
    try:
        name_encoder = joblib.load("name_encoder.pkl")
    except FileNotFoundError:
        st.error("`name_encoder.pkl` 파일을 찾을 수 없습니다. GitHub에 업로드했는지 확인하세요.")
        return None, None, None

    # 2. Surrogate 모델 로드
    try:
        # ⭐️⭐️⭐️ [중요] 우리 대화에서 'xgb_surrogate_2.pkl'로 확인했습니다. ⭐️⭐️⭐️
        surrogate = joblib.load("xgb_surrogate_2.pkl")
    except FileNotFoundError:
        st.error("`xgb_surrogate_2.pkl` 파일을 찾을 수 없습니다. GitHub에 업로드했는지 확인하세요.")
        return None, None, None
        
    # 3. PyTorch 모델 (RecipeNet3Head) 로드
    try:
        in_dim = len(config['spectrum_cols']) + len(config['lab_cols']) + config['embed_dim']
        num_pigments = len(config['recipe_cols'])
        
        model = RecipeNet3Head(in_dim, num_pigments, d_model=128)
        model.load_state_dict(torch.load("recipe_model.pth", map_location=device))
        model.eval()
    except FileNotFoundError:
        st.error("`recipe_model.pth` 파일을 찾을 수 없습니다. GitHub에 업로드했는지 확인하세요.")
        return None, None, None
    except Exception as e:
        st.error(f"PyTorch 모델 로드 중 오류 발생: {e}")
        st.error(f"CONFIG의 embed_dim({config['embed_dim']}), spectrum_cols({len(config['spectrum_cols'])}), lab_cols({len(config['lab_cols'])}) 개수가 학습 시점과 동일한지 확인하세요.")
        return None, None, None

    return model, name_encoder, surrogate

# ==========================================================
# 6. Streamlit UI (메인 앱 로직)
# ==========================================================

# --- 엑셀 파일 처리 함수 (이전과 동일) ---
@st.cache_data # 👈 파일을 다시 올리지 않는 한, 파싱 결과를 캐시합니다.
def parse_excel(uploaded_file, config):
    """
    업로드된 엑셀 파일에서 '정반사광 처리' == 'SCE'인 *모든* 행을 찾아
    DataFrame으로 반환합니다.
    """
    try:
        df = pd.read_excel(uploaded_file)
        
        filter_col = '정반사광 처리'
        if filter_col not in df.columns:
            st.error(f"엑셀 파일 오류: '{filter_col}' 컬럼을 찾을 수 없습니다.")
            return None
            
        sce_df = df[df[filter_col] == 'SCE'].copy()
        
        if sce_df.empty:
            st.error(f"엑셀 파일 오류: '{filter_col}' 컬럼에 'SCE' 값을 가진 행이 없습니다.")
            return None
            
        name_col = '데이터 이름'
        if name_col not in sce_df.columns:
            st.error(f"엑셀 파일 오류: '{name_col}' 컬럼을 찾을 수 없습니다.")
            return None
        
        sce_df['Color Name'] = sce_df[name_col].astype(str).str[4:]
        
        required_cols = config['lab_cols'] + config['spectrum_cols']
        missing_cols = [col for col in required_cols if col not in sce_df.columns]
        if missing_cols:
            st.error(f"엑셀 파일 오류: 다음 필수 컬럼을 찾을 수 없습니다: {', '.join(missing_cols)}")
            return None

        final_cols = ['Color Name'] + config['lab_cols'] + config['spectrum_cols']
        return sce_df[final_cols].reset_index(drop=True)

    except Exception as e:
        st.error(f"엑셀 파일 처리 중 오류 발생: {e}")
        return None

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

    if 'sce_data' not in st.session_state:
        st.session_state.sce_data = None

    if uploaded_file is not None:
        st.session_state.sce_data = parse_excel(uploaded_file, CONFIG)

    # --- 업로드된 데이터 목록 및 선택 UI ---
    if st.session_state.sce_data is not None:
        df_sce = st.session_state.sce_data
        st.header("2. 목표 색상 선택")
        
        selected_color_name = st.selectbox(
            f"'SCE' 기준 총 {len(df_sce)}개의 데이터가 로드되었습니다. 예측할 색상을 선택하세요.",
            options=df_sce['Color Name']
        )
        
        if selected_color_name:
            selected_row = df_sce[df_sce['Color Name'] == selected_color_name].iloc[0]
            
            # --- 선택된 데이터 확인 (3단 레이아웃 - 이전과 동일) ---
            st.subheader(f"'{selected_color_name}' 데이터 확인")
            
            lab_true_np = selected_row[CONFIG['lab_cols']].values.astype(float)
            spectrum_true_np = selected_row[CONFIG['spectrum_cols']].values.astype(float)
            
            # ⭐️ [레이아웃 변경] 3단 컬럼 레이아웃으로 수정
            col1, col2, col3 = st.columns([0.25, 0.5, 0.15]) # 40% / 20% / 40% 비율
            
            with col1:
                st.write("**목표 색상 정보:**")
                st.text_input("Color Name", value=selected_color_name, disabled=True, key=f"name_{selected_color_name}")
                st.text_input(f"{CONFIG['lab_cols'][0]}", value=f"{lab_true_np[0]:.2f}", disabled=True, key=f"l_{selected_color_name}")
                st.text_input(f"{CONFIG['lab_cols'][1]}", value=f"{lab_true_np[1]:.2f}", disabled=True, key=f"a_{selected_color_name}")
                st.text_input(f"{CONFIG['lab_cols'][2]}", value=f"{lab_true_np[2]:.2f}", disabled=True, key=f"b_{selected_color_name}")
            
            with col2:
                st.write("**스펙트럼 정보:**")
                spectrum_df = pd.DataFrame({
                    '파장 (Wavelength)': CONFIG['spectrum_cols'],
                    '값 (Value)': spectrum_true_np
                })
                # ⭐️ Lab 값 표시부(col1)와 높이를 맞추기 위해 height=270 (조절 가능)
                st.dataframe(spectrum_df, height=320) 

            with col3:
                st.write("**Target Color:**")
                # ⭐️ 크기 조절 함수(show_single_color_patch)는 이전 버전을 그대로 사용
                fig = show_single_color_patch(lab_true_np, title="Target (True)")
                st.pyplot(fig)

            # --- 3. 예측 실행 버튼 ---
            st.header("3. 예측 실행")
            if st.button(f"🚀 '{selected_color_name}' 레시피 예측 실행", type="primary"):
                with st.spinner('모델이 예측을 수행 중입니다...'):
                    run_inference(
                        model,
                        CONFIG,
                        surrogate,
                        spectrum=spectrum_true_np,
                        lab=lab_true_np,
                        color_name=selected_color_name,
                        name_encoder=name_encoder
                    )
    
    elif uploaded_file is None:
        st.info("⬆️ 예측을 시작하려면 엑셀 파일을 업로드해주세요.")
        
else:
    st.error("‼️ 모델 로딩 실패. GitHub 레파토리에 파일이 모두 있는지 확인하세요.")
    st.code("""
    [필수 파일 목록]
    1. app.py (지금 이 파일)
    2. recipe_model.pth
    3. name_encoder.pkl
    4. xgb_surrogate_2.pkl (⭐️ 이름 확인!)
    5. requirements.txt (openpyxl 포함 총 9줄)
    """)