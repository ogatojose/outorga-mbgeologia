import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm

# --- BLOCO DE SEGURANÇA (Instalação Automática) ---
try:
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
except ModuleNotFoundError:
    import subprocess
    import sys
    st.warning("Instalando bibliotecas matemáticas... aguarde.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "scikit-learn", "scipy"])
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

# --- FUNÇÕES MATEMÁTICAS ---
def modelo_logaritmico(x, a, b):
    return a * np.log(x) + b

def calcular_parametros_exatos(df_raw):
    """
    Realiza a regressão exata nas linhas 4 a 58 (índices 3 a 57)
    Considerando Coluna A = Tempo (t), Coluna B = Nível Dinâmico (ND)
    """
    # Selecionar intervalo fixo conforme solicitado (Linha 4 a 58 do Excel)
    # Pandas usa índice 0, então linha 4 é índice 3. Fim 58 é índice 57 (slice 3:58)
    try:
        df_crop = df_raw.iloc[3:58, :2].copy() # Pega colunas A e B
        df_crop.columns = ['t', 'nd']
        
        # Limpeza e conversão
        df_crop = df_crop.apply(pd.to_numeric, errors='coerce').dropna()
        df_crop = df_crop[df_crop['t'] > 0] # Evitar log(0)
        
        X = df_crop['t'].values
        y = df_crop['nd'].values
        
        if len(X) < 5:
            return None, None, None, None, "Dados insuficientes nas linhas 4-58"

        # Ajuste da Curva Logarítmica: y = a*ln(x) + b
        # Onde 'a' é a inclinação relacionada ao ciclo log
        popt, pcov = curve_fit(modelo_logaritmico, X, y)
        a_calc, b_calc = popt
        
        # Calcular R²
        y_pred = modelo_logaritmico(X, *popt)
        r2 = r2_score(y, y_pred)
        
        # Calcular Delta S usando x1=10 e x2=100 na equação ajustada
        y_10 = modelo_logaritmico(10, a_calc, b_calc)
        y_100 = modelo_logaritmico(100, a_calc, b_calc)
        delta_s = abs(y_100 - y_10) # Rebaixamento por ciclo logarítmico
        
        dados_grafico = {
            'X': X, 'y': y, 
            'X_fit': np.sort(X), 
            'y_fit': modelo_logaritmico(np.sort(X), a_calc, b_calc),
            'eq_label': f"y = {a_calc:.4f}*ln(x) + {b_calc:.4f}",
            'r2': r2
        }
        
        return delta_s, a_calc, b_calc, dados_grafico, None
        
    except Exception as e:
        return None, None, None, None, f"Erro no processamento: {str(e)}"

# --- INTERFACE DO STREAMLIT ---
st.set_page_config(page_title="Gestão de Recursos Hídricos", layout="wide")
st.title("🌊 Automação de Teste de Bombeamento (Ajuste Fino)")

# --- INPUTS ---
st.sidebar.header("Configurações")
cliente = st.sidebar.text_input("Cliente", "Cliente Padrão")
municipio = st.sidebar.text_input("Município", "Tapera/RS")

uploaded_file = st.file_uploader("Carregue a planilha (Excel)", type=["xlsx"])

if uploaded_file:
    # Ler sem cabeçalho para pegar por coordenadas fixas (A, B, E, etc)
    df_full = pd.read_excel(uploaded_file, header=None)
    
    # 1. PEGAR VAZÃO (Linha 3, Coluna E -> índice [2, 4])
    try:
        q_auto = float(df_full.iloc[2, 4]) # Linha 3 (idx 2), Col E (idx 4)
        st.success(f"✅ Vazão detectada na célula E3: {q_auto} m³/h")
    except:
        st.warning("⚠️ Não foi possível ler a vazão em E3. Usando valor padrão.")
        q_auto = 6.0
        
    vazao = st.number_input("Confirmar Vazão (m³/h)", value=q_auto)

    # 2. CALCULAR PARÂMETROS
    delta_s, a, b, graf_data, erro = calcular_parametros_exatos(df_full)
    
    if erro:
        st.error(erro)
    else:
        # Cálculos Finais
        # Transmissividade T = (0.183 * Q) / DeltaS
        transmissividade = (0.183 * vazao) / delta_s
        
        # Vazão Ótima (Estimativa baseada no NE original e ND máximo projetado ou real)
        # Pegando NE da Célula B9 (Linha 9, Col B) se existir, ou input manual
        try:
            ne_val = float(df_full.iloc[8, 1]) # Linha 9 é idx 8
        except:
            ne_val = 41.89
            
        ne = st.number_input("Nível Estático (NE)", value=ne_val)
        # O rebaixamento total para Q ótima geralmente usa o s_max do teste ou disponível
        s_max_teste = max(graf_data['y']) - min(graf_data['y']) # Estimativa simples
        
        vazao_otima = 0.8 * transmissividade * (graf_data['y'][-1] - graf_data['y'][0]) # Exemplo usando range do teste

        # --- EXIBIÇÃO ---
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Gráfico: Nível Dinâmico vs Tempo")
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Pontos Reais
            ax.scatter(graf_data['X'], graf_data['y'], color='blue', alpha=0.6, label='Dados de Campo')
            
            # Linha de Tendência
            ax.plot(graf_data['X_fit'], graf_data['y_fit'], 'r--', linewidth=2, label='Ajuste Logarítmico')
            
            # Anotação da Equação e R²
            texto_box = f"{graf_data['eq_label']}\n$R^2$ = {graf_data['r2']:.4f}\n$\Delta S$ (ciclo) = {delta_s:.4f}m"
            ax.text(0.05, 0.95, texto_box, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xscale('log') # Eixo X em escala logarítmica
            ax.set_xlabel('Tempo (min) - Escala Log')
            ax.set_ylabel('Nível Dinâmico (m)')
            ax.grid(True, which="both", ls="--", alpha=0.4)
            ax.legend()
            st.pyplot(fig)
            
        with col2:
            st.subheader("Resultados")
            st.metric("ΔS (Calculado)", f"{delta_s:.4f} m")
            st.metric("Transmissividade (T)", f"{transmissividade:.4f} m²/h")
            # Ajuste manual opcional se quiser forçar outro valor
            st.caption("Fórmula: T = (0,183 * Q) / ΔS")
            
            st.divider()
            if st.button("📄 Gerar Word"):
                st.info("O download iniciará em breve (Requer template atualizado).")
                # Aqui iria o bloco do docxtpl (mesmo do anterior)
