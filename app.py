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
    
    # MUDANÇA 1: Caixa de digitação para o tempo (permite decimais, ex: 0.5)
    tempo_op = st.sidebar.number_input("Tempo de Operação (horas/dia)", min_value=0.01, max_value=24.0, value=20.0, step=0.1, format="%.2f")
    
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

    # --- INTERFACE ---
    tab1, tab2, tab3 = st.tabs(["📉 Gráficos e Dados", "📝 Usos e Demandas", "📥 Downloads"])
    
    with tab1:
        # SEÇÃO 1: GRÁFICOS
        col1, col2 = st.columns([2, 1])
        with col1:
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.scatter(df_reb['t'], df_reb['nd'], color='navy', s=20)
            if a_reb is not None:
                x_fit = np.logspace(np.log10(min(df_reb['t'])), np.log10(max(df_reb['t'])), 100)
                y_fit = modelo_logaritmico(x_fit, a_reb, b_reb)
                ax1.plot(x_fit, y_fit, 'r--')
            ax1.set_xscale('log')
            ax1.set_xlabel('Tempo (min)')
            ax1.set_ylabel('Nível (m)')
            ax1.invert_yaxis()
            ax1.grid(True, ls="--", alpha=0.4)
            st.pyplot(fig1)
        with col2:
            st.metric("T (m²/h)", f"{T_reb_h:.9f}")
            st.metric("Submergência", f"{submergencia:.2f} m", delta_color="normal" if submergencia > 2 else "inverse")
        
        # MUDANÇA 2: TABELAS DE DADOS
        st.divider()
        st.subheader("📋 Dados Brutos da Planilha (Linhas 4 a 58)")
        
        c_tab1, c_tab2 = st.columns(2)
        
        with c_tab1:
            st.markdown("**Bombeamento (Colunas A, B, C, D)**")
            # iloc usa índices 0,1,2,3 para colunas A,B,C,D
            df_show_bomb = df_full.iloc[3:58, 0:4].reset_index(drop=True)
            # Tentativa de renomear para ficar legível (opcional)
            df_show_bomb.columns = ["t (min)", "N.D (m)", "s (m)", "Recup (m)"] 
            st.dataframe(df_show_bomb, use_container_width=True, height=300)
            
        with c_tab2:
            st.markdown("**Recuperação (Colunas G a M)**")
            # iloc usa índices 6 a 12 para colunas G,H,I,J,K,L,M
            df_show_rec = df_full.iloc[3:58, 6:13].reset_index(drop=True)
            df_show_rec.columns = ["G", "H", "I", "J (N.A)", "K", "L", "M (t/t')"]
            st.dataframe(df_show_rec, use_container_width=True, height=300)

    # --- ABA 2: LOGICA DE USOS NOVA ---
    with tab2:
        st.header("Definição dos Usos e Porcentagens")
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Finalidades")
            uc1, pc1 = st.columns([3, 1])
            uso1 = uc1.text_input("Uso 1", "Consumo Humano")
            porc1 = pc1.number_input("% Uso 1", 0, 100, 100, key="p1")
            
            uc2, pc2 = st.columns([3, 1])
            uso2 = uc2.text_input("Uso 2", "Limpeza Geral")
            porc2 = pc2.number_input("% Uso 2", 0, 100, 0, key="p2")
            
            uc3, pc3 = st.columns([3, 1])
            uso3 = uc3.text_input("Uso 3", "Combate a Incêndios")
            porc3 = pc3.number_input("% Uso 3", 0, 100, 0, key="p3")
            
            uc4, pc4 = st.columns([3, 1])
            uso4 = uc4.text_input("Uso 4", "")
            porc4 = pc4.number_input("% Uso 4", 0, 100, 0, key="p4")

            total_porc = porc1 + porc2 + porc3 + porc4
            if total_porc != 100 and total_porc != 0:
                st.warning(f"⚠️ Soma: {total_porc}%. Ideal: 100%.")

        with c2:
            st.subheader("Assistente de Justificativa")
            tipo = st.selectbox("Tipo de Demanda:", 
                              ["Personalizado", "Consumo Humano", "Abastecimento Público", 
                               "Limpeza Geral", "Combate a Incêndios", "Irrigação", "Dessedentação Animal"])
            
            sugestao = ""
            if tipo == "Consumo Humano":
                n_pessoas = st.number_input("Nº Pessoas", min_value=1, value=4)
                vol_pessoas = n_pessoas * 0.18
                sugestao = (f"A demanda justifica-se pela necessidade de abastecimento contínuo e potável para atendimento "
                            f"das necessidades básicas de {n_pessoas} pessoas, totalizando um consumo estimado de "
                            f"{format_padrao(vol_pessoas)} m³/dia (considerando 0,18 m³/hab/dia), visando a segurança hídrica e sanitária.")
            elif tipo == "Abastecimento Público":
                n_pessoas = st.number_input("População Atendida", min_value=1, value=50)
                vol_pessoas = n_pessoas * 0.18
                sugestao = (f"O poço destina-se ao abastecimento público, viabilizado pela administração municipal para "
                            f"atendimento da localidade. Estima-se o atendimento de {n_pessoas} habitantes, "
                            f"gerando uma demanda de {format_padrao(vol_pessoas)} m³/dia (base 0,18 m³/hab/dia).")
            elif tipo == "Limpeza Geral":
                sugestao = ("A demanda justifica-se pela necessidade de manutenção e operacionalização básica do empreendimento, "
                            "incluindo a limpeza geral das instalações, banheiros e pátios, garantindo as condições de higiene.")
            elif tipo == "Combate a Incêndios":
                sugestao = ("A demanda justifica-se pela necessidade de abastecimento e manutenção da reserva técnica de incêndio (RTI), "
                            "visando a adequação do estabelecimento às normas de segurança e prevenção (PPCI), garantindo a proteção da edificação e dos usuários.")
            elif tipo == "Irrigação":
                sugestao = ("A demanda justifica-se pela necessidade de irrigação complementar para incremento da produtividade "
                            "agrícola, garantindo o desenvolvimento das culturas mesmo em períodos de estiagem.")
            elif tipo == "Dessedentação Animal":
                sugestao = ("A demanda justifica-se para a dessedentação animal, garantindo o bem-estar e o desenvolvimento "
                            "do rebanho, bem como as condições sanitárias das instalações.")
            
            justificativa = st.text_area("Texto Final:", value=sugestao, height=200)

    with tab3:
        st.header("Gerar Documentos")
        
        buffer_reb = io.BytesIO()
        fig1.savefig(buffer_reb, format='png', dpi=150)
        buffer_reb.seek(0)
        
        buffer_rec = None
        if not df_rec.empty and not df_rec.isnull().values.all():
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.scatter(df_rec['ratio'], df_rec['res'], color='green', s=20)
            if a_rec is not None:
                 x_fit2 = np.logspace(np.log10(min(df_rec['ratio'])), np.log10(max(df_rec['ratio'])), 100)
                 y_fit2 = modelo_logaritmico(x_fit2, a_rec, b_rec)
                 ax2.plot(x_fit2, y_fit2, 'r--')
            ax2.set_xscale('log')
            ax2.invert_yaxis()
            ax2.grid(True, ls="--", alpha=0.4)
            
            buffer_rec = io.BytesIO()
            fig2.savefig(buffer_rec, format='png', dpi=150)
            buffer_rec.seek(0)

        ctx_base = {
            'cliente': cliente, 'municipio': municipio,
            'ne': format_padrao(ne), 'nd': format_padrao(nd_final),
            'q': format_padrao(q), 's_total': format_padrao(s_total),
            'transmissividade': format_transmissividade(T_reb_h),
        }

        if st.button("📄 Baixar Memorial (.docx)"):
            try:
                doc = DocxTemplate("template_memorial.docx")
                ctx = ctx_base.copy()
                img_reb_obj = InlineImage(doc, buffer_reb, width=Mm(150))
                if buffer_rec:
                    img_rec_obj = InlineImage(doc, buffer_rec, width=Mm(150))
                else:
                    img_rec_obj = "Gráfico de Recuperação não gerado"

                ctx.update({
                    'ds_linha': format_padrao(ds_reb),
                    't_reb_s': format_transmissividade(T_reb_s),
                    'ce_reb': format_cap_especifica(cap_esp_reb),
                    'vazao_otima': format_padrao(vazao_otima),
                    'grafico_rebaixamento': img_reb_obj,
                    'ds_rec': format_padrao(ds_rec),
                    't_rec_h': format_transmissividade(T_rec_h),
                    't_rec_s': format_transmissividade(T_rec_s),
                    'ce_rec': format_cap_especifica(cap_esp_rec),
                    'grafico_recuperacao': img_rec_obj
                })
                doc.render(ctx)
                bio = io.BytesIO()
                doc.save(bio)
                st.download_button("⬇️ Download Memorial", bio.getvalue(), f"Memorial_{cliente}.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            except Exception as e:
                st.error(f"Erro no Memorial: {e}")

        st.divider()

        if st.button("📄 Baixar Projeto (.docx)"):
            try:
                doc_proj = DocxTemplate("template_projeto.docx")
                ctx_proj = ctx_base.copy()
                ctx_proj.update({
                    'modelo_bomba': modelo_bomba,
                    'potencia': potencia, 'estagios': num_estagios,
                    'diametro_edutor': diametro_edutor,
                    'prof_bomba': format_padrao(prof_bomba),
                    'submergencia': format_padrao(submergencia),
                    'tempo': f"{tempo_op}",
                    'q_dia': format_padrao(vazao_diaria),
                    'uso1': uso1, 'porc1': str(porc1),
                    'uso2': uso2, 'porc2': str(porc2),
                    'uso3': uso3, 'porc3': str(porc3),
                    'uso4': uso4, 'porc4': str(porc4),
                    'justificativa': justificativa
                })
                doc_proj.render(ctx_proj)
                bio_proj = io.BytesIO()
                doc_proj.save(bio_proj)
                st.download_button("⬇️ Download Projeto", bio_proj.getvalue(), f"Projeto_{cliente}.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            except Exception as e:
                st.error(f"Erro no Projeto: {e}")
