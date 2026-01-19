import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm
import io

# --- FUNÇÕES MATEMÁTICAS ---
def modelo_logaritmico(x, a, b):
    # Evita log de números negativos ou zero
    return a * np.log(x) + b

def format_numero(valor):
    """1.23 -> 1,23"""
    if valor is None: return "-"
    return f"{valor:.2f}".replace('.', ',')

def format_cientifico(valor):
    """0.000123 -> 1,23E-04 (formato científico para m²/s)"""
    if valor is None: return "-"
    return f"{valor:.2e}".replace('.', ',')

def analisar_dados_log(x_data, y_data, x1=10, x2=100):
    """
    Realiza a regressão y = a*ln(x) + b e calcula métricas
    """
    try:
        # Ajuste da curva
        popt, pcov = curve_fit(modelo_logaritmico, x_data, y_data)
        a_calc, b_calc = popt
        
        # R²
        y_pred = modelo_logaritmico(x_data, *popt)
        r2 = r2_score(y_data, y_pred)
        
        # Delta S (Diferença entre x=10 e x=100 na reta ajustada)
        y_10 = modelo_logaritmico(x1, a_calc, b_calc)
        y_100 = modelo_logaritmico(x2, a_calc, b_calc)
        delta_s = abs(y_100 - y_10)
        
        return a_calc, b_calc, r2, delta_s
    except:
        return None, None, 0, 0

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Hidrogeologia Profissional", layout="wide")
st.title("🌊 Sistema de Análise de Teste de Bombeamento")

# --- SIDEBAR: CONFIGURAÇÕES ---
st.sidebar.header("Dados de Entrada")
cliente = st.sidebar.text_input("Cliente", "Cliente Exemplo")
municipio = st.sidebar.text_input("Município", "Tapera/RS")
uploaded_file = st.file_uploader("Arquivo Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    # Ler planilha completa (sem cabeçalho para usar índices fixos)
    df_full = pd.read_excel(uploaded_file, header=None)
    
    # --- 1. LEITURA DE PARÂMETROS FIXOS ---
    try:
        q_auto = float(df_full.iloc[2, 4]) # Célula E3
        ne_auto = float(df_full.iloc[2, 1]) # Célula B3
        st.sidebar.success(f"Vazão (E3): {q_auto} m³/h | NE (B3): {ne_auto} m")
    except:
        st.sidebar.warning("Não foi possível ler E3 ou B3 automaticamente.")
        q_auto = 6.0
        ne_auto = 0.0
        
    q = st.sidebar.number_input("Confirmar Vazão (m³/h)", value=q_auto)
    ne = st.sidebar.number_input("Confirmar Nível Estático (m)", value=ne_auto)

    # --- 2. PREPARAÇÃO DOS DADOS ---
    # REBAIXAMENTO: Linhas 4-58, Col A(t) e B(ND)
    df_reb = df_full.iloc[3:58, [0, 1]].copy()
    df_reb.columns = ['t', 'nd']
    df_reb = df_reb.apply(pd.to_numeric, errors='coerce').dropna()
    df_reb = df_reb[df_reb['t'] > 0] # Remove t=0 para log
    
    # RECUPERAÇÃO: Linhas 4-58. 
    # Assumindo: Coluna M (Index 12) = t/t' e Coluna J (Index 9) = N.A.
    try:
        df_rec = df_full.iloc[3:58, [12, 9]].copy()
        df_rec.columns = ['ratio', 'na']
        df_rec = df_rec.apply(pd.to_numeric, errors='coerce').dropna()
        df_rec = df_rec[df_rec['ratio'] > 0]
        # Calcular Rebaixamento Residual (s') = NA - NE
        df_rec['res'] = df_rec['na'] - ne
    except:
        df_rec = pd.DataFrame() # Vazio se falhar

    # --- 3. CÁLCULOS E GRÁFICOS ---
    tab1, tab2 = st.tabs(["📉 Rebaixamento", "📈 Recuperação"])
    
    # --- ABA 1: REBAIXAMENTO ---
    with tab1:
        col1a, col1b = st.columns([2, 1])
        
        # Processamento Matemática
        a_reb, b_reb, r2_reb, ds_reb = analisar_dados_log(df_reb['t'], df_reb['nd'])
        
        # Cálculos Hidráulicos
        if ds_reb > 0:
            T_reb_h = (0.183 * q) / ds_reb  # m²/h
            T_reb_s = T_reb_h / 3600        # m²/s
            
            # Capacidade Específica (Q / s_total)
            s_max_reb = df_reb['nd'].max() - ne
            cap_esp_reb = q / s_max_reb if s_max_reb > 0 else 0
            
            vazao_otima = 0.8 * T_reb_h * s_max_reb
        else:
            T_reb_h = T_reb_s = cap_esp_reb = vazao_otima = 0

        with col1a:
            # Gráfico Rebaixamento
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            ax1.scatter(df_reb['t'], df_reb['nd'], color='navy', s=20, label='Dados')
            
            if a_reb is not None:
                x_fit = np.logspace(np.log10(min(df_reb['t'])), np.log10(max(df_reb['t'])), 100)
                y_fit = modelo_logaritmico(x_fit, a_reb, b_reb)
                ax1.plot(x_fit, y_fit, 'r--', label='Ajuste Log')
                
                texto = (f"y = {a_reb:.4f}ln(x) + {b_reb:.4f}\n"
                         f"R² = {r2_reb:.4f}\n"
                         f"ΔS = {ds_reb:.4f} m")
                ax1.text(0.02, 0.98, texto, transform=ax1.transAxes, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            ax1.set_xscale('log')
            ax1.set_xlabel('Tempo (min)')
            ax1.set_ylabel('Nível Dinâmico (m)')
            ax1.grid(True, which="both", ls="--", alpha=0.4)
            st.pyplot(fig1)
            
        with col1b:
            st.subheader("Resultados Rebaixamento")
            st.metric("Transmissividade (m²/h)", f"{T_reb_h:.4f}")
            st.metric("Transmissividade (m²/s)", f"{T_reb_s:.2e}")
            st.metric("Cap. Específica (m³/h/m)", f"{cap_esp_reb:.4f}")
            st.metric("Vazão Ótima", f"{vazao_otima:.2f}")

    # --- ABA 2: RECUPERAÇÃO ---
    with tab2:
        col2a, col2b = st.columns([2, 1])
        
        # Processamento Matemática
        if not df_rec.empty:
            a_rec, b_rec, r2_rec, ds_rec = analisar_dados_log(df_rec['ratio'], df_rec['res'])
            
            # Cálculos Hidráulicos Recuperação
            if ds_rec > 0:
                T_rec_h = (0.183 * q) / ds_rec
                T_rec_s = T_rec_h / 3600
                # Para recuperação, Cap Específica costuma ser replicada ou recalculada
                # Aqui usaremos a mesma lógica do rebaixamento para consistência
                cap_esp_rec = cap_esp_reb 
            else:
                T_rec_h = T_rec_s = cap_esp_rec = 0
                
            with col2a:
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                ax2.scatter(df_rec['ratio'], df_rec['res'], color='green', s=20, label='Dados')
                
                if a_rec is not None:
                    x_fit2 = np.logspace(np.log10(min(df_rec['ratio'])), np.log10(max(df_rec['ratio'])), 100)
                    y_fit2 = modelo_logaritmico(x_fit2, a_rec, b_rec)
                    ax2.plot(x_fit2, y_fit2, 'r--', label='Ajuste Log')
                    
                    texto2 = (f"y = {a_rec:.4f}ln(x) + {b_rec:.4f}\n"
                              f"R² = {r2_rec:.4f}\n"
                              f"ΔS' = {ds_rec:.4f} m")
                    ax2.text(0.02, 0.98, texto2, transform=ax2.transAxes, 
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

                ax2.set_xscale('log')
                ax2.set_xlabel("Razão t/t' (Adimensional)")
                ax2.set_ylabel("Rebaixamento Residual (m)")
                ax2.grid(True, which="both", ls="--", alpha=0.4)
                st.pyplot(fig2)
                
            with col2b:
                st.subheader("Resultados Recuperação")
                st.metric("Transmissividade (m²/h)", f"{T_rec_h:.4f}")
                st.metric("Transmissividade (m²/s)", f"{T_rec_s:.2e}")
                st.metric("Cap. Específica (m³/h/m)", f"{cap_esp_rec:.4f}")
        else:
            st.warning("Dados de recuperação não encontrados nas colunas M e J.")
            T_rec_h = T_rec_s = ds_rec = 0

    # --- GERAR RELATÓRIO WORD ---
    st.divider()
    if st.button("📄 Baixar Relatório Completo"):
        try:
            doc = DocxTemplate("template_memorial.docx")
            
            # Salvar Gráfico 1 (Rebaixamento)
            img_reb = io.BytesIO()
            fig1.savefig(img_reb, format='png', dpi=150)
            img_reb.seek(0)
            
            # Salvar Gráfico 2 (Recuperação)
            img_rec = io.BytesIO()
            if not df_rec.empty:
                fig2.savefig(img_rec, format='png', dpi=150)
                img_rec.seek(0)
                img_rec_word = InlineImage(doc, img_rec, width=Mm(150))
            else:
                img_rec_word = "Gráfico não gerado"

            contexto = {
                'cliente': cliente,
                'municipio': municipio,
                'ne': format_numero(ne),
                'q': format_numero(q),
                # Rebaixamento
                'nd': format_numero(df_reb['nd'].max()),
                's_total': format_numero(s_max_reb),
                'ds_linha': format_numero(ds_reb),
                'transmissividade': format_numero(T_reb_h), # m²/h
                't_reb_s': format_cientifico(T_reb_s),      # m²/s
                'ce_reb': format_numero(cap_esp_reb),
                'vazao_otima': format_numero(vazao_otima),
                'grafico_rebaixamento': InlineImage(doc, img_reb, width=Mm(150)),
                # Recuperação
                'ds_rec': format_numero(ds_rec),
                't_rec_h': format_numero(T_rec_h),
                't_rec_s': format_cientifico(T_rec_s),
                'ce_rec': format_numero(cap_esp_rec),
                'grafico_recuperacao': img_rec_word
            }
            
            doc.render(contexto)
            
            bio = io.BytesIO()
            doc.save(bio)
            
            st.download_button(
                label="⬇️ Download .docx",
                data=bio.getvalue(),
                file_name=f"Memorial_{cliente}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        except Exception as e:
            st.error(f"Erro no template: {e}")
            st.info("Verifique se as etiquetas {{ grafico_recuperacao }} etc. estão no Word.")
