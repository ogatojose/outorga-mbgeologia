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

def gerar_imagem_tabela(df, titulo):
    """Converte DataFrame em imagem PNG"""
    # Altura dinâmica baseada no número de linhas
    altura = max(2, len(df) * 0.25 + 1.5)
    fig, ax = plt.subplots(figsize=(8, altura))
    ax.axis('off')
    ax.set_title(titulo, fontweight="bold", fontsize=12, pad=10)
    
    # Criar tabela
    tabela = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(9)
    tabela.scale(1.2, 1.3) # Ajuste de espaçamento
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    return buf

def gerar_imagem_cabecalho(dados):
    """Gera uma imagem com os dados do cabeçalho"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # Borda decorativa simples
    rect = plt.Rectangle((0.01, 0.01), 0.98, 0.98, fill=False, color="black", lw=1.5)
    ax.add_patch(rect)
    
    # Título
    ax.text(0.5, 0.9, "RESUMO TÉCNICO - TESTE DE BOMBEAMENTO", 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Linha separadora
    ax.plot([0.05, 0.95], [0.82, 0.82], color='black', lw=0.5)

    # Coluna 1 (X=0.05)
    y_start = 0.75
    espaco = 0.12
    
    ax.text(0.05, y_start, f"Cliente: {dados['cliente']}", fontsize=11, fontweight='bold')
    ax.text(0.05, y_start - espaco, f"Município: {dados['municipio']}", fontsize=11)
    ax.text(0.05, y_start - 2*espaco, f"Aquífero: {dados['aquifero']}", fontsize=11)
    ax.text(0.05, y_start - 3*espaco, f"Execução: {dados['execucao']}", fontsize=11)
    
    # Coluna 2 (X=0.50)
    ax.text(0.50, y_start, f"Data Início: {dados['data_ini']}", fontsize=11)
    ax.text(0.50, y_start - espaco, f"Data Fim: {dados['data_fim']}", fontsize=11)
    ax.text(0.50, y_start - 2*espaco, f"Profundidade: {dados['prof']} m", fontsize=11)
    ax.text(0.50, y_start - 3*espaco, f"Crivo: {dados['crivo']} m", fontsize=11)
    
    # Coluna 3 (X=0.75)
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

# --- SIDEBAR (INPUTS) ---
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
    
    # --- DADOS PARA O PROJETO ---
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

    # ================= PREPARAÇÃO DAS TABELAS (2 CASAS DECIMAIS) =================
    # Bombeamento
    df_bomb_clean = df_full.iloc[3:58, 0:4].copy()
    df_bomb_clean.columns = ["t (min)", "N.D (m)", "s (m)", "r (m)"]
    df_bomb_clean = df_bomb_clean.apply(pd.to_numeric, errors='coerce')
    # Formatação visual para 2 casas
    df_bomb_fmt = df_bomb_clean.map('{:.2f}'.format)
    # Substituir nan por traço
    df_bomb_fmt = df_bomb_fmt.replace('nan', '-')

    # Recuperação
    df_rec_clean = df_full.iloc[3:58, 6:13].copy()
    df_rec_clean.columns = ["t'", "t", "ND", "NA", "r'", "s'", "t/t'"]
    df_rec_clean = df_rec_clean.apply(pd.to_numeric, errors='coerce')
    df_rec_fmt = df_rec_clean.map('{:.2f}'.format)
    df_rec_fmt = df_rec_fmt.replace('nan', '-')

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Dados do Teste", "📝 Usos e Demandas", "📥 Downloads"])
    
    with tab1:
        st.markdown("### 📋 Resumo do Teste de Bombeamento")
        
        # Dados para o cabeçalho
        dados_cabecalho = {
            'cliente': cliente, 'municipio': municipio, 'aquifero': aquifero, 'execucao': execucao,
            'data_ini': data_inicio.strftime('%d/%m/%Y'), 'data_fim': data_fim.strftime('%d/%m/%Y'),
            'prof': profundidade_poco, 'crivo': crivo_bomba,
            'ne': f"{ne:.2f}", 'nd': f"{nd_final:.2f}", 'q': f"{q:.2f}", 'tempo': tempo_bombeamento
        }

        # Geração da Imagem do Cabeçalho
        img_cabecalho_buf = gerar_imagem_cabecalho(dados_cabecalho)
        
        # Mostra o cabeçalho visualmente
        st.image(img_cabecalho_buf, use_container_width=True)
        
        col_down_head, _ = st.columns([1, 3])
        with col_down_head:
            st.download_button("⬇️ Baixar Imagem Cabeçalho", img_cabecalho_buf, f"cabecalho_{cliente}.png", "image/png")

        st.divider()

        # GRÁFICOS
        c_graf, c_res = st.columns([2, 1])
        with c_graf:
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
        
        with c_res:
            st.metric("Transmissividade (T)", f"{T_reb_h:.9f} m²/h")
            st.metric("Cap. Específica", f"{cap_esp_reb:.6f} m³/h/m")

        st.divider()

        # TABELAS
        st.subheader("📷 Tabelas (Formatadas 2 casas)")
        c_tab1, c_tab2 = st.columns(2)
        
        with c_tab1:
            st.markdown("**Bombeamento**")
            st.dataframe(df_bomb_fmt, height=300, use_container_width=True)
            img_bomb = gerar_imagem_tabela(df_bomb_fmt, f"Bombeamento - {cliente}")
            st.download_button("⬇️ Baixar Tabela Bombeamento", img_bomb, f"tabela_bombeamento_{cliente}.png", "image/png")

        with c_tab2:
            st.markdown("**Recuperação**")
            st.dataframe(df_rec_fmt, height=300, use_container_width=True)
            img_rec = gerar_imagem_tabela(df_rec_fmt, f"Recuperação - {cliente}")
            st.download_button("⬇️ Baixar Tabela Recuperação", img_rec, f"tabela_recuperacao_{cliente}.png", "image/png")

    # --- ABA 2: USOS (Mantida igual) ---
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

    # --- ABA 3: DOWNLOADS ---
    with tab3:
        st.header("Gerar Documentos")
        
        # PREPARAÇÃO DAS IMAGENS PARA O WORD
        # 1. Gráfico Rebaixamento
        buffer_reb = io.BytesIO()
        fig1.savefig(buffer_reb, format='png', dpi=150)
        buffer_reb.seek(0)
        
        # 2. Gráfico Recuperação
        buffer_rec = None
        if not df_rec.empty:
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

        # 3. Cabeçalho e Tabelas (Já gerados anteriormente nas variáveis: img_cabecalho_buf, img_bomb, img_rec)
        # Precisamos garantir que o ponteiro do buffer esteja no zero para re-leitura
        img_cabecalho_buf.seek(0)
        img_bomb.seek(0)
        img_rec.seek(0)

        ctx_base = {
            'cliente': cliente, 'municipio': municipio,
            'ne': format_padrao(ne), 'nd': format_padrao(nd_final),
            'q': format_padrao(q), 's_total': format_padrao(s_total),
            'transmissividade': format_transmissividade(T_reb_h),
        }

        # --- MEMORIAL ---
        if st.button("📄 Baixar Memorial (.docx)"):
            try:
                doc = DocxTemplate("template_memorial.docx")
                ctx = ctx_base.copy()
                img_reb_obj = InlineImage(doc, buffer_reb, width=Mm(150))
                img_rec_obj = InlineImage(doc, buffer_rec, width=Mm(150)) if buffer_rec else "N/A"

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

        # --- PROJETO OPERACIONAL ---
        if st.button("📄 Baixar Projeto (.docx)"):
            try:
                doc_proj = DocxTemplate("template_projeto.docx")
                ctx_proj = ctx_base.copy()
                
                # INSERÇÃO DAS NOVAS IMAGENS (CABEÇALHO E TABELAS)
                img_cabecalho_word = InlineImage(doc_proj, img_cabecalho_buf, width=Mm(160))
                img_bomb_word = InlineImage(doc_proj, img_bomb, width=Mm(160))
                img_rec_word = InlineImage(doc_proj, img_rec, width=Mm(160))

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
                    'justificativa': justificativa,
                    
                    # NOVAS TAGS PARA O TEMPLATE
                    'img_cabecalho': img_cabecalho_word,
                    'img_tabela_bomb': img_bomb_word,
                    'img_tabela_rec': img_rec_word
                })
                doc_proj.render(ctx_proj)
                bio_proj = io.BytesIO()
                doc_proj.save(bio_proj)
                st.download_button("⬇️ Download Projeto", bio_proj.getvalue(), f"Projeto_{cliente}.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            except Exception as e:
                st.error(f"Erro no Projeto: {e}")
