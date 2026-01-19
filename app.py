import streamlit as st
import subprocess
import sys

# --- BLOCO DE EMERGÊNCIA (O HACK) ---
# Isso obriga o servidor a instalar o matplotlib se ele não encontrar
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    st.warning("Instalando biblioteca gráfica... aguarde um momento.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt
# ------------------------------------

import pandas as pd
import numpy as np

st.set_page_config(page_title="Gestão de Recursos Hídricos", layout="wide")

st.title("🌊 Automação de Relatórios: Memorial e Projeto")

uploaded_file = st.file_uploader("Carregue a planilha de campo (Excel)", type=["xlsx"])

if uploaded_file:
    # Lendo a planilha (Planilha1)
    df = pd.read_excel(uploaded_file, sheet_name=0) 
    
    # --- CÁLCULOS HIDRÁULICOS ---
    st.header("📊 Análise do Teste de Bombeamento")
    
    # Extraindo dados conforme o padrão do teu Excel
    ne = 41.89  # Nível Estático
    nd = 45.34  # Nível Dinâmico
    q = 6.0     # Vazão m3/h
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gráfico de Rebaixamento")
        # Criando o gráfico (Figura 2 do teu modelo)
        fig, ax = plt.subplots()
        
        # Simulação simples para o gráfico aparecer (depois ligamos aos dados reais)
        # Se a planilha tiver colunas 't (min)' e 's (m)', usamos elas:
        if 't (min)' in df.columns and 's (m)' in df.columns:
            # Filtra zeros para log não dar erro
            df_chart = df[df['t (min)'] > 0]
            ax.plot(df_chart['t (min)'], df_chart['s (m)'], 'o-', label='Rebaixamento')
        else:
            # Dados fictícios só para mostrar que o gráfico funciona
            ax.plot([1, 10, 100, 1000], [0.5, 1.5, 2.5, 3.5], 'o-')
            st.info("Avisos: Colunas 't (min)' e 's (m)' não detectadas automaticamente. Mostrando exemplo.")

        ax.set_xscale('log')
        ax.set_xlabel('Tempo (min) - Escala Log')
        ax.set_ylabel('Rebaixamento (m)')
        ax.grid(True, which="both", ls="-")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Parâmetros Calculados")
        ds_linha = st.number_input("Inclinação da Reta (ΔS')", value=1.07)
        transmissividade = (0.183 * q) / ds_linha
        vazao_otima = 0.8 * transmissividade * (nd - ne)
        
        st.metric("Transmissividade (T)", f"{transmissividade:.4f} m²/h")
        st.metric("Vazão Ótima", f"{vazao_otima:.2f} m³/h")

    # --- USOS E QUALIDADE ---
    st.divider()
    st.header("📝 Definições do Projeto")
    
    potavel = st.radio("A água é potável?", ["Sim", "Não"], index=1)
    if potavel == "Não":
        params = st.text_input("Parâmetros fora do padrão:", "coliformes totais e bactérias")
        st.warning(f"Texto automático: '...novas análises serão feitas devido a {params}.'")

    st.success("Sistema Operacional!")
