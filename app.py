import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm
import io

# --- FUNÇÕES DE FORMATAÇÃO ---
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

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Central de Outorgas", layout="wide")
st.title("🌊 Central de Outorgas e Projetos")

# ==================================================
# BARRA LATERAL - ENTRADA DE DADOS
# ==================================================
st.sidebar.title("1. Dados Gerais")
cliente = st.sidebar.text_input("Cliente", "Cliente Exemplo")
municipio = st.sidebar.text_input("Município", "Tapera/RS")
uploaded_file = st.file_uploader("Arquivo Excel (.xlsx)", type=["xlsx"])

# Variáveis globais de inicialização
q = ne = nd_final = transmissividade = s_total = 0

if uploaded_file:
    df_full = pd.read_excel(uploaded_file, header=None)
    
    # --- Leitura Automática ---
    try:
        q_auto = float(df_full.iloc[2, 4]) # E3
        ne_auto = float(df_full.iloc[2, 1]) # B3
        st.sidebar.success(f"Lido: Q={q_auto} | NE={ne_auto}")
    except:
        q_auto = 6.0
        ne_auto = 0.0
        
    q = st.sidebar.number_input("Vazão do Poço (m³/h)", value=q_auto)
    ne = st.sidebar.number_input("Nível Estático (m)", value=ne_auto)

    # --- INPUTS DO PROJETO OPERACIONAL ---
    st.sidebar.markdown("---")
    st.sidebar.title("2. Regime de Operação")
    
    # 1. Tempo e Vazão Diária
    tempo_op = st.sidebar.slider("Tempo de Operação Diária (horas)", 1, 24, 20)
    vazao_diaria = q * tempo_op
    st.sidebar.info(f"Volume Diário Calculado: **{vazao_diaria:.2f} m³/dia**")

    # 2. Definição da Bomba
    st.sidebar.markdown("---")
    st.sidebar.subheader("Equipamento")
    modelo_bomba = st.sidebar.text_input("Modelo da Bomba", "Ebara 4BPS")
    potencia = st.sidebar.text_input("Potência (cv)", "1.5 cv")
    num_estagios = st.sidebar.text_input("Nº de Estágios", "12")
    diametro_edutor = st.sidebar.text_input("Diâmetro Edutor", "1 1/2 pol")
    prof_bomba = st.sidebar.number_input("Profundidade Instalação (m)", value=ne + 20.0)

    # ==================================================
    # PROCESSAMENTO HIDRÁULICO (Igual ao anterior)
    # ==================================================
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

    submergencia = prof_bomba - nd_final
    cor_alerta = "green" if submergencia > 2 else "red"

    # ==================================================
    # INTERFACE PRINCIPAL
    # ==================================================
    
    tab1, tab2, tab3 = st.tabs(["📉 Análise Hidráulica", "📝 Detalhes do Uso", "📥 Downloads"])
    
    # --- ABA 1: GRÁFICOS ---
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.scatter(df_reb['t'], df_reb['nd'], color='navy', s=20, label='Dados')
            if a_reb is not None:
                x_fit = np.logspace(np.log10(min(df_reb['t'])), np.log10(max(df_reb['t'])), 100)
                y_fit = modelo_logaritmico(x_fit, a_reb, b_reb)
                ax1.plot(x_fit, y_fit, 'r--', label='Ajuste Log')
            ax1.set_xscale('log')
            ax1.set_xlabel('Tempo (min)')
            ax1.set_ylabel('Nível (m)')
            ax1.invert_yaxis()
            ax1.grid(True, which="both", ls="--", alpha=0.4)
            st.pyplot(fig1)

        with col2:
            st.metric("T (m²/h)", f"{T_reb_h:.9f}")
            st.metric("C. Específica", f"{cap_esp_reb:.6f}")
            st.metric("Submergência", f"{submergencia:.2f} m", delta_color="normal" if submergencia > 2 else "inverse")

    # --- ABA 2: USOS E JUSTIFICATIVA (NOVO) ---
    with tab2:
        st.header("Definição dos Usos da Água")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Usos Pretendidos")
            uso1 = st.text_input("Uso 1", "Consumo Humano")
            uso2 = st.text_input("Uso 2", "Dessedentação Animal")
            uso3 = st.text_input("Uso 3", "")
            uso4 = st.text_input("Uso 4", "")
        
        with c2:
            st.subheader("Justificativa da Demanda")
            st.info("💡 Assistente de Redação Técnica")
            
            # Opções de templates para ajudar na escrita
            tipo_demanda = st.selectbox("Selecione o tipo de justificativa para gerar um rascunho:", 
                                      ["Personalizado (Escrever do zero)", 
                                       "Consumo Humano (Padrão)", 
                                       "Irrigação (Padrão)", 
                                       "Indústria/Processos", 
                                       "Misto (Humano + Animal)"])
            
            texto_sugerido = ""
            if tipo_demanda == "Consumo Humano (Padrão)":
                texto_sugerido = "A demanda justifica-se pela necessidade de abastecimento contínuo e potável para atendimento das necessidades básicas dos residentes/usuários do local, visando a segurança hídrica e sanitária."
            elif tipo_demanda == "Irrigação (Padrão)":
                texto_sugerido = "A demanda justifica-se pela necessidade de irrigação complementar para incremento da produtividade agrícola, garantindo o desenvolvimento das culturas mesmo em períodos de estiagem."
            elif tipo_demanda == "Indústria/Processos":
                texto_sugerido = "A demanda justifica-se para uso em processos industriais e higienização de instalações, sendo insumo fundamental para a manutenção das atividades produtivas da empresa."
            elif tipo_demanda == "Misto (Humano + Animal)":
                texto_sugerido = "A demanda justifica-se para o abastecimento da sede da propriedade e dessedentação animal, garantindo o bem-estar animal e as condições sanitárias adequadas."
            
            # Caixa de texto editável (se o usuário selecionou um template, ele já preenche)
            justificativa_final = st.text_area("Edite o texto final abaixo:", value=texto_sugerido, height=150)
            
            if justificativa_final:
                st.success("Texto pronto para o relatório!")

    # --- ABA 3: DOWNLOADS ---
    with tab3:
        st.header("Gerar Documentos")
        
        # Prepara imagens
        img_reb = io.BytesIO()
        fig1.savefig(img_reb, format='png', dpi=150)
        img_reb.seek(0)
        
        img_rec_word = "Gráfico ausente"
        if not df_rec.empty and not df_rec.isnull().values.all():
             # Recriar fig2 para garantir que existe
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.scatter(df_rec['ratio'], df_rec['res'], color='green', s=20)
            if a_rec is not None:
                 x_fit2 = np.logspace(np.log10(min(df_rec['ratio'])), np.log10(max(df_rec['ratio'])), 100)
                 y_fit2 = modelo_logaritmico(x_fit2, a_rec, b_rec)
                 ax2.plot(x_fit2, y_fit2, 'r--')
            ax2.set_xscale('log')
            ax2.invert_yaxis()
            
            img_rec = io.BytesIO()
            fig2.savefig(img_rec, format='png', dpi=150)
            img_rec.seek(0)
            img_rec_word = InlineImage(doc, img_rec, width=Mm(150)) if 'doc' in locals() else img_rec # Ajuste técnico

        # Contexto Base
        contexto_base = {
            'cliente': cliente,
            'municipio': municipio,
            'ne': format_padrao(ne),
            'nd': format_padrao(nd_final),
            'q': format_padrao(q),
            's_total': format_padrao(s_total),
            'transmissividade': format_transmissividade(T_reb_h),
        }

        # BOTÃO MEMORIAL
        if st.button("📄 Baixar Memorial (.docx)"):
            try:
                doc = DocxTemplate("template_memorial.docx")
                ctx = contexto_base.copy()
                
                # Tratamento imagem recuperação para o memorial
                if not isinstance(img_rec_word, str):
                     img_rec_word_mem = InlineImage(doc, img_rec, width=Mm(150))
                else:
                     img_rec_word_mem = "N/A"

                ctx.update({
                    'ds_linha': format_padrao(ds_reb),
                    't_reb_s': format_transmissividade(T_reb_s),
                    'ce_reb': format_cap_especifica(cap_esp_reb),
                    'vazao_otima': format_padrao(vazao_otima),
                    'grafico_rebaixamento': InlineImage(doc, img_reb, width=Mm(150)),
                    'ds_rec': format_padrao(ds_rec),
                    't_rec_h': format_transmissividade(T_rec_h),
                    't_rec_s': format_transmissividade(T_rec_s),
                    'ce_rec': format_cap_especifica(cap_esp_rec),
                    'grafico_recuperacao': img_rec_word_mem
                })
                doc.render(ctx)
                bio = io.BytesIO()
                doc.save(bio)
                st.download_button("⬇️ Download Memorial", bio.getvalue(), f"Memorial_{cliente}.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            except Exception as e:
                st.error(f"Erro Memorial: {e}")

        st.divider()

        # BOTÃO PROJETO
        if st.button("📄 Baixar Projeto Operacional (.docx)"):
            try:
                doc_proj = DocxTemplate("template_projeto.docx")
                ctx_proj = contexto_base.copy()
                
                # Atualizando com os novos dados
                ctx_proj.update({
                    'modelo_bomba': modelo_bomba,
                    'potencia': potencia,
                    'estagios': num_estagios,
                    'diametro_edutor': diametro_edutor,
                    'prof_bomba': format_padrao(prof_bomba),
                    'submergencia': format_padrao(submergencia),
                    
                    # NOVOS DADOS
                    'tempo': f"{tempo_op}",          # Inteiro ou float
                    'q_dia': format_padrao(vazao_diaria),
                    'uso1': uso1 if uso1 else "",
                    'uso2': uso2 if uso2 else "",
                    'uso3': uso3 if uso3 else "",
                    'uso4': uso4 if uso4 else "",
                    'justificativa': justificativa_final
                })
                
                doc_proj.render(ctx_proj)
                bio_proj = io.BytesIO()
                doc_proj.save(bio_proj)
                st.download_button("⬇️ Download Projeto", bio_proj.getvalue(), f"Projeto_{cliente}.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            except Exception as e:
                st.error(f"Erro Projeto: {e}")
