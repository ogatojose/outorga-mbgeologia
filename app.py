import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm
import io

# --- FUNÇÕES ---
def modelo_logaritmico(x, a, b):
    return a * np.log(x) + b

def format_padrao(valor):
    if valor is None: return "-"
    return f"{valor:.2f}".replace('.', ',')

def format_transmissividade(valor):
    if valor is None: return "-"
    return f"{valor:.9f}".replace('.', ',')

def format_cap_especifica(valor):
    if valor is None: return "-"
    return f"{valor:.6f}".replace('.', ',')

def analisar_dados_log(x_data, y_data, x1=10, x2=100):
    try:
        popt, pcov = curve_fit(modelo_logaritmico, x_data, y_data)
        a_calc, b_calc = popt
        y_pred = modelo_logaritmico(x_data, *popt)
        r2 = r2_score(y_data, y_pred)
        y_10 = modelo_logaritmico(x1, a_calc, b_calc)
        y_100 = modelo_logaritmico(x2, a_calc, b_calc)
        delta_s = abs(y_100 - y_10)
        return a_calc, b_calc, r2, delta_s
    except:
        return None, None, 0, 0

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="Central de Outorgas", layout="wide")
st.title("🌊 Central de Outorgas e Projetos")

# --- SIDEBAR ---
st.sidebar.title("1. Dados Gerais")
cliente = st.sidebar.text_input("Cliente", "Cliente Exemplo")
municipio = st.sidebar.text_input("Município", "Tapera/RS")
uploaded_file = st.file_uploader("Arquivo Excel (.xlsx)", type=["xlsx"])

q = ne = nd_final = transmissividade = s_total = 0

if uploaded_file:
    df_full = pd.read_excel(uploaded_file, header=None)
    try:
        q_auto = float(df_full.iloc[2, 4]) 
        ne_auto = float(df_full.iloc[2, 1])
        st.sidebar.success(f"Lido: Q={q_auto} | NE={ne_auto}")
    except:
        q_auto = 6.0
        ne_auto = 0.0
        
    q = st.sidebar.number_input("Vazão (m³/h)", value=q_auto)
    ne = st.sidebar.number_input("Nível Estático (m)", value=ne_auto)

    # --- PROJETO ---
    st.sidebar.markdown("---")
    st.sidebar.title("2. Regime e Equipamento")
    
    tempo_op = st.sidebar.slider("Horas/dia", 1, 24, 20)
    vazao_diaria = q * tempo_op
    st.sidebar.info(f"Q Diária: {vazao_diaria:.2f} m³/dia")

    modelo_bomba = st.sidebar.text_input("Modelo Bomba", "Ebara 4BPS")
    potencia = st.sidebar.text_input("Potência", "1.5 cv")
    num_estagios = st.sidebar.text_input("Estágios", "12")
    diametro_edutor = st.sidebar.text_input("Diâmetro Edutor", "1 1/2 pol")
    prof_bomba = st.sidebar.number_input("Profundidade Instalação (m)", value=ne + 20.0)

    # --- PROCESSAMENTO ---
    df_reb = df_full.iloc[3:58, [0, 1]].copy()
    df_reb.columns = ['t', 'nd']
    df_reb = df_reb.apply(pd.to_numeric, errors='coerce').dropna()
    df_reb = df_reb[df_reb['t'] > 0]
    
    try:
        df_rec = df_full.iloc[3:58, [12, 9]].copy()
        df_rec.columns = ['ratio', 'na']
        df_rec = df_rec.apply(pd.to_numeric, errors='coerce').dropna()
        df_rec = df_rec[df_rec['ratio'] > 0]
        df_rec['res'] = df_rec['na'] - ne
    except:
        df_rec = pd.DataFrame()

    # Cálculos
    a_reb, b_reb, r2_reb, ds_reb = analisar_dados_log(df_reb['t'], df_reb['nd'])
    
    if ds_reb > 0:
        T_reb_h = (0.183 * q) / ds_reb
        T_reb_s = T_reb_h / 3600
        cap_esp_reb = T_reb_h * 0.8
        nd_final = df_reb['nd'].max()
        s_total = nd_final - ne
        vazao_otima = cap_esp_reb * s_total
    else
