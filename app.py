import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm
import io
from datetime import timedelta, date

# --- FUNÇÕES AUXILIARES ---
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

def gerar_excel_cabecalho(dados):
    """Gera um buffer Excel com os dados do cabeçalho organizados"""
    # Organização visual em 3 blocos de colunas para ficar igual ao layout
    data = [
        ["IDENTIFICAÇÃO", "", "DATAS", "", "DADOS TÉCNICOS", ""],
        ["Cliente:", dados['cliente'], "Data Início:", dados['data_ini'], "Nível Estático:", f"{dados['ne']} m"],
        ["Município:", dados['municipio'], "Data Fim:", dados['data_fim'], "Nível Dinâmico:", f"{dados['nd']} m"],
        ["Aquífero:", dados['aquifero'], "Profundidade:", f"{dados['prof']} m", "Vazão:", f"{dados['q']} m³/h"],
        ["Execução:", dados['execucao'], "Crivo:", f"{dados['crivo']} m", "Tempo Bomb.:", dados['tempo']]
    ]
    df = pd.DataFrame(data)
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, header=False, sheet_name='Cabeçalho')
    output.seek(0)
    return output

def gerar_imagem_tabela(df, titulo):
    """Converte DataFrame em imagem PNG (Versão Estreita)"""
    altura = max(2, len(df) * 0.25 + 1.5)
    fig, ax = plt.subplots(figsize=(4.5, altura)) 
    
    ax.axis('off')
    ax.set_title(titulo, fontweight="bold", fontsize=11, pad=10)
    
    tabela = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(9)
    tabela.scale(1.0, 1.3)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    return buf

def gerar_imagem_cabecalho(dados):
    """Gera uma imagem com os dados do cabeçalho"""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis('off')
    
    rect = plt.Rectangle((0.01, 0.01), 0.98, 0.98, fill=False, color="black", lw=1.5)
    ax.add_patch(rect)
    
    ax.text(0.5, 0.9, "RESUMO TÉCNICO - TESTE DE BOMBEAMENTO", 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax.plot([0.05, 0.95], [0.82, 0.82], color='black', lw=0.5)

    # Coluna 1
    y_start = 0.75
    espaco = 0.12
    ax.text(0.05, y_start, f"Cliente: {dados['cliente']}", fontsize=11, fontweight='bold')
    ax.text(0.05, y_start - espaco, f"Município: {dados['municipio']}", fontsize=11)
    ax.text(0.05, y_start - 2*espaco, f"Aquífero: {dados['aquifero']}", fontsize=11)
    ax.text(0.05, y_start - 3*espaco, f"Execução: {dados['execucao']}", fontsize=11)
    
    # Coluna 2
    ax.text(0.45, y_start, f"Data Início: {dados['data_ini']}", fontsize=11)
    ax.text(0.45, y_start - espaco, f"Data Fim: {dados['data_fim']}", fontsize=11)
    ax.text(0.45, y_start - 2*espaco, f"Profundidade: {dados['prof']} m", fontsize=11)
    ax.text(0.45, y_start - 3*espaco, f"Crivo: {dados['crivo']} m", fontsize=11)
    
    # Coluna 3
    ax.text(0.75, y_start, f"NE: {dados['ne']} m", fontsize=11)
    ax.text(0.75, y_start - espaco, f"ND: {dados['nd']} m", fontsize=11)
    ax.text(0.75, y_start - 2*espaco, f"Vazão: {dados['q']} m³/h", fontsize=11)
    ax.text(0.75, y_start - 3*espaco, f"Tempo: {dados['tempo']}", fontsize=11)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    return buf

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
st.sidebar.title("1. Dados do Ensaio")
uploaded_file = st.file_uploader("Arquivo Excel (.xlsx)", type=["xlsx"])

q = ne = nd_final = transmissividade = s_total = 0

if uploaded_file:
    df_full = pd.read_excel(uploaded_file, header=None)
    try:
        q_auto = float(df_full.iloc[2, 4]) 
        ne_auto = float(df_full.iloc[2, 1])
    except:
        q_auto = 6.0
        ne_auto = 0.0
    
    st.sidebar.subheader("Identificação")
    cliente = st.sidebar.text_input("Cliente", "Cliente Exemplo")
    municipio = st.sidebar.text_input("Município", "Tapera/RS")
    aquifero = st.sidebar.text_input("Aquífero", "Serra Geral")
    execucao = st.sidebar.text_input("Execução", "Bruna Koppe Kronhardt")
    
    st.sidebar.subheader("Datas")
    data_inicio = st.sidebar.date_input("Data Início", date.today())
    data_fim = data_inicio + timedelta(days=1)
    st.sidebar.caption(f"Término Automático: {data_fim.strftime('%d/%m/%Y')}")

    st.sidebar.subheader("Dados Técnicos")
    profundidade_poco = st.sidebar.text_input("Profundidade Poço (m)", "100")
    crivo_bomba = st.sidebar.text_input("Crivo da Bomba (m)", "80")
    tipo_equipamento = st.sidebar.text_input("Equipamento", "Bomba Submersa")
    tempo_bombeamento = st.sidebar.text_input("Tempo Bombeamento", "24 h")
    
    st.sidebar.markdown("---")
    q = st.sidebar.number_input("Vazão (m³/h)", value=q_auto)
    ne = st.sidebar.number_input("Nível Estático (m)", value=ne_auto)
    
    # --- PROJETO ---
    st.sidebar.markdown("---")
    st.sidebar.title("2. Projeto Operacional")
    tempo_op = st.sidebar.number_input("Tempo Operação (h/dia)", 0.1, 24.0, 20.0)
    vazao_diaria = q * tempo_op
    
    modelo_bomba = st.sidebar.text_input("Modelo Bomba", "Ebara 4BPS")
    potencia = st.sidebar.text_input("Potência", "1.5 cv")
    num_estagios = st.sidebar.text_input("Estágios", "12")
    diametro_edutor = st.sidebar.text_input("Diâmetro Edutor", "1 1/2 pol")
    prof_bomba = st.sidebar.number_input("Prof. Instalação (m)", value=ne + 20.0)

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

    a_reb, b_reb, r2_reb, ds_reb = analisar_dados_log(df_reb['t'], df_reb['nd'])
    
    if ds_reb > 0:
        T_reb_h = (0.183 * q) / ds_reb
        T_reb_s = T_reb_h / 3600
        cap_esp_reb = T_reb_h * 0.8
        nd_final = df_reb['nd'].max()
        s_total = nd_final - ne
        vazao_otima = cap_esp_reb * s_total
    else:
        T_reb_h = T_reb_s = cap_esp_reb = vazao_otima = 0
        nd_final = ne

    if not df_rec.empty:
        a_rec, b_rec, r2_rec, ds_rec = analisar_dados_log(df_rec['ratio'], df_rec['res'])
        if ds_rec > 0:
            T_rec_h = (0.183 * q) / ds_rec
            T_rec_s = T_rec_h / 3600
            cap_esp_rec = T_rec_h * 0.8
        else:
            T_rec_h = T_rec_s = cap_esp_rec = 0
    else:
        T_rec_h = T_rec_s = cap_esp_rec = ds_rec = 0
        a_rec = None

    submergencia = prof_bomba - nd_final

    # ================= FORMATAÇÃO =================
    df_bomb_clean = df_full.iloc[3:58, 0:4].copy()
    df_bomb_clean.columns = ["t (min)", "N.D (m)", "s (m)", "
