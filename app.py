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

# ==========================================================
# 0. CONFIG (Jupyter Notebook에서 정확하게 복사)
# ==========================================================
# ⚠️ 이 CONFIG 변수는 Jupyter Notebook의 것과 100% 동일해야 합니다.
# ⚠️ 'embed_dim'은 모델 로드를 위해 수동으로 추가했습니다.
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
    
    # 아래 값들은 추론 시에는 직접 사용되지 않음
    # "n_splits": 5,
    # "batch_size": 32,
    # "epochs": 50,
    # "lr": 1e-3,
    # "random_state": 42,
    # "use_aug": True
}


# ==========================================================
# 1. 텍스트 인코더 클래스 정의 (SimpleNameEncoder)
# (Jupyter Notebook에서 그대로 복사)
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

    def fit(self, names: List[str]): # (app.py에서는 이 함수를 호출하지 않음)
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
# (Jupyter Notebook에서 그대로 복사)
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
# (Jupyter Notebook에서 그대로 복사 및 수정)
# ==========================================================
def lab_to_rgb(lab):
    """Lab -> RGB 변환 (skimage 활용)"""
    lab = np.array(lab).reshape(1,1,3)
    rgb = color.lab2rgb(lab)
    return rgb[0,0,:]

def show_color_patches(lab_true, lab_pred):
    """Streamlit용 색상 비교차트 생성"""
    fig, ax = plt.subplots(1, 2, figsize=(6,3))
    rgb_true = lab_to_rgb(lab_true)
    rgb_pred = lab_to_rgb(lab_pred)

    ax[0].imshow([[rgb_true]])
    ax[0].set_title("Target (True)")
    ax[0].axis("off")

    ax[1].imshow([[rgb_pred]])
    ax[1].set_title("Predicted (Surrogate)")
    ax[1].axis("off")
    
    return fig # ⭐️ st.pyplot()을 위해 fig 객체 반환

def deltaE_00(y_true, y_pred, kL=1, kC=1, kH=1):
    """CIEDE2000 DeltaE 계산"""
    L1, a1, b1 = y_true[:,0], y_true[:,1], y_true[:,2]
    L2, a2, b2 = y_pred[:,0], y_pred[:,1], y_pred[:,2]
    C1 = np.sqrt(a1*a1 + b1*b1)
    C2 = np.sqrt(a2*a2 + b2*b2)
    C_bar = 0.5 * (C1 + C2)
    C_bar7 = C_bar**7
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
    hp_bar = np.where((C1p*C2p)==0,hp_sum,
        np.where(np.abs(h1p-h2p)>180,(hp_sum+360.0)/2.0-360.0*(hp_sum>=360.0),hp_sum/2.0))
    T=(1-0.17*np.cos(np.radians(hp_bar-30))
       +0.24*np.cos(np.radians(2*hp_bar))
       +0.32*np.cos(np.radians(3*hp_bar+6))
       -0.20*np.cos(np.radians(4*hp_bar-63)))
    Sl=1+0.015*(Lp_bar-50)**2/np.sqrt(20+(Lp_bar-50)**2)
    Sc=1+0.045*Cp_bar; Sh=1+0.015*Cp_bar*T
    delta_theta=30*np.exp(-((hp_bar-275)/25)**2)
    Rc=2*np.sqrt(C_bar**7/(C_bar**7+25**7+1e-12))
    Rt=-Rc*np.sin(2*np.radians(delta_theta))
    dE00=np.sqrt((dLp/(kL*Sl))**2+(dCp/(kC*Sc))**2+(dHp/(kH*Sh))**2
       +Rt*(dCp/(kC*Sc))*(dHp/(kH*Sh)))
    return dE00

# ==========================================================
# 4. 추론 함수 (test_new_swatch -> Streamlit용으로 수정)
# ==========================================================
def run_inference(model, cfg, surrogate, spectrum, lab, color_name, name_encoder):
    """
    Streamlit 입력값을 받아 예측을 수행하고 결과를 출력/반환합니다.
    (Jupyter의 test_new_swatch 함수 기반)
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
        P_g = (p*t).cpu().numpy() # g 단위

    # ---- Surrogate 예측
    lab_pred = surrogate.predict(P_g)

    # ---- ΔE00
    lab_true = lab.reshape(1,3)
    de00 = deltaE_00(lab_true, lab_pred)

    # ---- Streamlit 출력
    recipe_g_series = pd.Series(P_g.flatten(), index=cfg['recipe_cols'])
    
    st.subheader("📊 예측 결과")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted ΔE00", value=f"{de00.mean():.3f}")
        st.write(f"**True Lab:** {np.round(lab_true.flatten(), 2)}")
        st.write(f"**Pred Lab:** {np.round(lab_pred.flatten(), 2)}")

    with col2:
        st.write("**색상 비교:**")
        fig = show_color_patches(lab_true.flatten(), lab_pred.flatten())
        st.pyplot(fig)

    st.subheader("🔬 예측된 레시피 (g 단위, Top 10)")
    # 0이 아닌 값만 필터링 후 상위 10개
    recipe_nonzero = recipe_g_series[recipe_g_series > 1e-4]
    st.dataframe(recipe_nonzero.sort_values(ascending=False).head(10))

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
        # ⭐️ 경로가 Jupyter와 다릅니다. GitHub 루트에 있다고 가정합니다.
        surrogate = joblib.load("xgb_surrogate_2.pkl")
    except FileNotFoundError:
        st.error("`xgb_surrogate.pkl` 파일을 찾을 수 없습니다. GitHub에 업로드했는지 확인하세요.")
        return None, None, None
        
    # 3. PyTorch 모델 (RecipeNet3Head) 로드
    # 뼈대(구조)를 먼저 만들고, 가중치(state_dict)를 덮어씁니다.
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
st.set_page_config(layout="wide")
st.title("🧪 레시피 예측 모델 (RecipeNet3Head)")

# 모델 로드
model, name_encoder, surrogate = load_all_models(CONFIG)

if model and name_encoder and surrogate:
    st.success(f"모델 로드 완료! (안료 개수: {len(CONFIG['recipe_cols'])})")

    # --- 사용자 입력 UI ---
    st.header("1. 목표 색상 정보 입력")
    
    # 예시: 'DK MARINA BLUE'
    color_name_input = st.text_input("Color Name", "DK MARINA BLUE")
    
    c1, c2, c3 = st.columns(3)
    # L* a* b* 컬럼명을 CONFIG에서 가져옴
    l_input = c1.number_input(f"Target {CONFIG['lab_cols'][0]}", value=35.72, format="%.2f")
    a_input = c2.number_input(f"Target {CONFIG['lab_cols'][1]}", value=-6.12, format="%.2f")
    b_input = c3.number_input(f"Target {CONFIG['lab_cols'][2]}", value=-24.54, format="%.2f")
    lab_input_np = np.array([l_input, a_input, b_input])

    st.header("2. 스펙트럼 정보 입력")
    st.write(f"총 {len(CONFIG['spectrum_cols'])}개 파장대({CONFIG['spectrum_cols'][0]} ~ {CONFIG['spectrum_cols'][-1]}) 값을 쉼표(,)로 구분하여 입력하세요.")

    # 예시: 'DK MARINA BLUE'의 스펙트럼
    default_spectrum_str = (
        "19.55, 19.24, 19.04, 18.67, 18.36, 18.30, 18.48, 18.79, 18.78, 17.78, "
        "15.81, 13.49, 11.37, 9.76, 8.40, 7.23, 6.39, 5.69, 5.00, 4.63, 4.60, "
        "4.68, 4.48, 4.14, 4.18, 4.85, 7.10, 12.25, 21.00, 32.86, 46.82"
    )
    spectrum_str_input = st.text_area("Spectrum (쉼표로 구분)", default_spectrum_str, height=150)

    # --- 예측 버튼 ---
    if st.button("🚀 레시피 예측 실행", type="primary"):
        try:
            # 1. 스펙트럼 입력값 파싱
            spectrum_values = [float(s.strip()) for s in spectrum_str_input.split(',')]
            
            # 2. 개수 검증
            if len(spectrum_values) != len(CONFIG['spectrum_cols']):
                st.error(f"스펙트럼 입력 오류: {len(CONFIG['spectrum_cols'])}개가 필요하지만 {len(spectrum_values)}개가 입력되었습니다.")
            else:
                spectrum_input_np = np.array(spectrum_values)
                
                # 3. 예측 함수 실행
                with st.spinner('모델이 예측을 수행 중입니다...'):
                    run_inference(
                        model,
                        CONFIG,
                        surrogate,
                        spectrum=spectrum_input_np,
                        lab=lab_input_np,
                        color_name=color_name_input,
                        name_encoder=name_encoder
                    )
        except Exception as e:
            st.error(f"예측 중 오류 발생: {e}")
            st.error("입력값을 확인해주세요. (예: 스펙트럼에 숫자 아닌 값이 포함됨)")
else:
    st.error("‼️ 모델 로딩 실패. GitHub 레파토리에 5개 파일이 모두 있는지 확인하세요.")
    st.code("""
    [필수 파일 목록]
    1. app.py (지금 이 파일)
    2. recipe_model.pth (PyTorch 모델 가중치)
    3. name_encoder.pkl (SimpleNameEncoder 객체)
    4. xgb_surrogate.pkl (Surrogate 모델)
    5. requirements.txt (라이브러리 목록)
    """)