import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm
import io
from scipy.stats import linregress

# --- BLOCO DE SEGURANÇA (Para garantir gráficos) ---
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

# --- FUNÇÃO DE CÁLCULO DE INCLINAÇÃO ---
def calcular_slope_log(x_array, y_array):
    """
    Calcula a inclinação (Delta S) por ciclo logarítmico (base 10).
    A regressão linear é feita com ln(x), então o slope real é slope_ln * 2.303.
    """
    # Filtrar valores válidos (x > 0 e não nulos)
    mask = (x_array > 0) & (pd.notnull(x_array)) & (pd.notnull(y_array))
    x_clean = x_array[mask]
    y_clean = y_array[mask]
    
    if len(x_clean) < 2:
        return 0, 0, 0, x_clean, y_clean
    
    # Regressão Linear: y = slope * ln(x) + intercept
    slope_ln, intercept, r_value, p_value, std_err = linregress(np.log(x_clean), y_clean)
    
    # Delta S (variação por ciclo logarítmico na base 10)
    delta_s = abs(slope_ln * np.log(10)) 
    
    return delta_s, slope_ln, intercept, x_clean, y_clean

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Gestão de Recursos Hídricos", layout="wide")
st.title("🌊 Automação Completa: Teste de Bombeamento")

# --- BARRA LATERAL ---
st.sidebar.header("Dados do Projeto")
cliente = st.sidebar.text_input("Nome do Cliente", "Cliente Exemplo Ltda")
municipio = st.sidebar.text_input("Município", "Tapera/RS")
uploaded_file = st.file_uploader("Carregue a planilha de campo (Excel)", type=["xlsx"])

if uploaded_file:
    # Ler a planilha (Assumindo que os dados começam na linha 2, após cabeçalhos)
    # Ajuste: header=1 significa que a linha 2 do Excel é o cabeçalho real
    df = pd.read_excel(uploaded_file, sheet_name=0, header=1)
    
    # --- EXTRAÇÃO DE COLUNAS (Baseado no padrão Jéssica) ---
    # Colunas Rebaixamento: A=t(min), B=ND, C=s(m) -> Indices 0, 1, 2
    # Colunas Recuperação: M=t/t', J=N.A -> Indices 12, 9 (Verificar seu Excel)
    
    # Pegando NE da primeira linha de dados válida
    ne_inicial = df.iloc[0, 1] # Coluna ND, primeira linha costuma ser o estático
    q_teste = 6.0 # Vazão padrão ou ler da planilha se tiver campo fixo
    
    # --- ABAS DE ANÁLISE ---
    tab1, tab2 = st.tabs(["📉 Rebaixamento (Bombeamento)", "📈 Recuperação"])
    
    with tab1:
        st.header("Análise de Rebaixamento")
        col_a, col_b = st.columns([2, 1])
        
        with col_a:
            # Dados
            t_reb = pd.to_numeric(df.iloc[:, 0], errors='coerce') # Tempo
            s_reb = pd.to_numeric(df.iloc[:, 2], errors='coerce') # Rebaixamento
            
            # Cálculo Automático
            delta_s_reb, slope_reb, inter_reb, x_r, y_r = calcular_slope_log(t_reb, s_reb)
            
            # Gráfico
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            ax1.semilogx(x_r, y_r, 'o', label='Dados de Campo', alpha=0.6)
            
            # Linha de Tendência
            x_fit = np.logspace(np.log10(min(x_r)), np.log10(max(x_r)), 100)
            y_fit = slope_reb * np.log(x_fit) + inter_reb
            ax1.semilogx(x_fit, y_fit, 'r--', label=f'Ajuste (ΔS = {delta_s_reb:.2f})')
            
            ax1.set_xlabel('Tempo (min)')
            ax1.set_ylabel('Rebaixamento (m)')
            ax1.grid(True, which="both", ls="--", alpha=0.4)
            ax1.legend()
            st.pyplot(fig1)
            
        with col_b:
            st.subheader("Resultados Calculados")
            st.info(f"Equação: s = {slope_reb:.3f} * ln(t) + {inter_reb:.3f}")
            
            q_user = st.number_input("Vazão (Q) - m³/h", value=q_teste, key="q_reb")
            ds_user = st.number_input("ΔS (Ciclo Log)", value=float(f"{delta_s_reb:.3f}"), format="%.3f", key="ds_reb")
            
            if ds_user > 0:
                t_calc = (0.183 * q_user) / ds_user
                nd_max = max(s_reb.dropna()) + ne_inicial
                s_total = nd_max - ne_inicial
                q_otima = 0.8 * t_calc * s_total
                
                st.metric("Transmissividade (T)", f"{t_calc:.4f} m²/h")
                st.metric("Vazão Ótima", f"{q_otima:.2f} m³/h")
            else:
                st.warning("ΔS inválido para cálculo.")

    with tab2:
        st.header("Análise de Recuperação")
        col_c, col_d = st.columns([2, 1])
        
        with col_c:
            # Dados Recuperação
            # Residual Drawdown (s') = Nível Medido (NA) - Nível Estático (NE)
            # Cooper-Jacob usa t/t' no eixo X vs Residual Drawdown no eixo Y
            t_ratio = pd.to_numeric(df.iloc[:, 12], errors='coerce') # Coluna t/t'
            na_rec = pd.to_numeric(df.iloc[:, 9], errors='coerce')   # Coluna N.A
            s_residual = na_rec - ne_inicial
            
            # Cálculo Automático
            delta_s_rec, slope_rec, inter_rec, x_rec, y_rec = calcular_slope_log(t_ratio, s_residual)
            
            # Gráfico
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.semilogx(x_rec, y_rec, 'o', color='green', label='Dados Recuperação', alpha=0.6)
            
            # Linha de Tendência
            if len(x_rec) > 0:
                x_fit_rec = np.logspace(np.log10(min(x_rec)), np.log10(max(x_rec)), 100)
                y_fit_rec = slope_rec * np.log(x_fit_rec) + inter_rec
                ax2.semilogx(x_fit_rec, y_fit_rec, 'r--', label=f'Ajuste (ΔS = {delta_s_rec:.2f})')
            
            ax2.set_xlabel("Razão t/t' (Adimensional)")
            ax2.set_ylabel("Rebaixamento Residual (m)")
            ax2.grid(True, which="both", ls="--", alpha=0.4)
            ax2.legend()
            st.pyplot(fig2)
            
        with col_d:
            st.subheader("Resultados Recuperação")
            st.info(f"Equação: s' = {slope_rec:.3f} * ln(t/t') + {inter_rec:.3f}")
            
            ds_rec_user = st.number_input("ΔS Recuperação", value=float(f"{delta_s_rec:.3f}"), format="%.3f", key="ds_rec")
            
            if ds_rec_user > 0:
                t_rec_calc = (0.183 * q_user) / ds_rec_user
                st.metric("Transmissividade (T)", f"{t_rec_calc:.4f} m²/h")


    # --- GERAÇÃO DE DOCUMENTO ---
    st.divider()
    if st.button("📄 Gerar Relatório Word"):
        # Lógica para preencher o template usando 't_calc' e 'q_otima' calculados acima
        # (Aqui entra a mesma lógica do docxtpl que te passei antes)
        st.success("Cálculos concluídos! Implementar integração com template aqui.")
