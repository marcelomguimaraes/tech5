#Importação das bibliotecas
import streamlit as st 
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from joblib import load
import numpy as np
# np==1.26.4

#FUNÇÃO BOTÃO
def acao_botao(estudante):
    print(estudante)
    modelo = joblib.load('model/xgb.joblib')
    final_pred = modelo.predict(estudante_predito)
    risco_estudante = modelo.predict_proba(estudante_predict_df.drop(['risco'], axis=1))
    
    # print(risco_estudante)
    st.markdown(
        '''
        <h2>📊 Resultado da avaliação</h2>
        ''',unsafe_allow_html=True
    )
    
    if final_pred[-1] == 1:
        st.error('🚨 Ahh, o estudante apresenta um RISCO de defasagem.') 
        st.error(f'Risco de Defasagem: {risco_estudante[0][1]:.2%}')
    else:
        st.success('Ebba, o estudante apresenta BAIXO RISCO de defasagem.')
        st.success(f'Risco de Defasagem: {risco_estudante[0][1]:.2%}')
        st.balloons()
    
def probabilidade_risco(modelo, X_novo):
    """
    Retorna a probabilidade do evento ocorrer (classe 1)
    
    Parâmetros:
    modelo  -> modelo treinado (RandomForest)
    X_novo  -> dados para previsão (DataFrame ou array)
    
    Retorno:
    probabilidade da classe positiva (evento = 1)
    """
    
    # predict_proba retorna [prob_classe_0, prob_classe_1]
    prob = modelo.predict_proba(X_novo)
    print(prob[:, 1])
    
#IMPORTAR BASE LIMPA
df = pd.read_csv(r'https://raw.githubusercontent.com/marcelomguimaraes/tech5/refs/heads/main/datasets/clean.csv', sep=";")
df = df[['genero', 'inde_22', 'inde_23', 'inde_24', 'ida', 'ieg', 'iaa', 'ips', 'ipp', 'ipv', 'risco']]

#DESENHO DA PÁGINA NO STREAMLIT
st.set_page_config(page_title="Risco de Defasagem", page_icon="📔")

st.sidebar.info('Desenvolvido por Marcelo Luiz Mendes Guimarães no Tech Challenge da Fase 5 do curso de Data Analytics da FIAP. 🏆')

st.title('🔍 Defasagem Educacional')

st.markdown(
    '''
    <p style="font-size: 16px; line-height: 1.6; text-align: justify"><strong>Que tal preencher o formulário abaixo para descobrir o risco de defasagem do seu estudante?</strong> Basta informar alguns dados sobre as dimensões acadêmicas, psicossociais e psicopedagógicas do aluno. Na sequência, o resultado estará na palma da sua mão e você poderá pensar em estratégicas pedagógicas para potencializar o aprendizado do estudante.</p> 
    <p style="font-size: 16px; font-weight: 600">Vamos começar? 🚀✨</p>
    ''', unsafe_allow_html=True)

st.markdown(
    '''
    <div style="padding: 20px 0;">
        <h3>1º Passo | Dados Pessoais </h3>
        <p style="text-align: justify; line-height: 1.6"><strong>Vamos começar preenchendo os dados pessoais?</strong> Essas informações são importantes para que possamos ter uma visão geral e oferecer uma avaliação mais precisa.</p>
    </div>
    ''', unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    input_idade = int(st.slider('📅 Idade: ', 1, 30))

with col2:
    input_sexo = st.selectbox('♂️ Sexo: ', ['Feminino', 'Masculino'])
    input_sexo = 0 if input_sexo == 'Feminino' else 1
    
st.markdown(
    '''
    <div style="padding: 20px 0;">
        <h3>2º Passo | Desempenho acadêmico </h3>
        <p style="text-align: justify; line-height: 1.6"><strong>Agora, chegou a vez de preencher os dados sobre a dimensão acadêmica.</strong> Essas informações são cruciais para que a previsão seja altamente assertiva.</p>
    </div>
    ''', unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    ida = float(st.number_input(label='🧮 Desempenho Acadêmico (IDA): ', min_value = 0.0, max_value = 10.0, value=0.0, step = 0.01, format='%.1f'))

with col2:
    ieg = float(st.number_input(label='🎉 Engajamento (IEG): ', min_value = 0.0, max_value = 10.0, value=0.0, step = 0.01, format='%.1f'))

st.markdown(
    '''
    <div style="padding: 20px 0;">
        <h3>3º Passo | Psicossocial </h3>
        <p style="text-align: justify; line-height: 1.6"><strong>Tão importante quanto o desempenho acadêmico, o aspecto psicossocial ajuda a compreender o panorama do estudante.</strong> Não deixe de preenchê-las com assertividade, combinado?</p>
    </div>
    ''', unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    iaa = float(st.number_input(label='🤔 Autoavaliação (IAA): ', min_value = 0.0, max_value = 10.0, value=0.0, step = 0.01, format='%.1f'))

with col2:
    ips = float(st.number_input(label='📄 Psicossocial (IPS): ', min_value = 0.0, max_value = 10.0, value=0.0, step = 0.01, format='%.1f'))


st.markdown(
    '''
    <div style="padding: 20px 0;">
        <h3>4º Passo | Psicopedagógica </h3>
        <p style="text-align: justify; line-height: 1.6"><strong>Chegou o momento de preencher as informações da dimensão psicopedagógica.</strong> Estas informações contribuem significativamente para a análise do estudante.</p>
    </div>
    ''', unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    ipp = float(st.number_input(label='⚖️ Psicopedagógico (IPP): ', min_value = 0.0, max_value = 10.0, value=0.0, step = 0.01, format='%.1f'))

with col2:
    ipv = float(st.number_input(label='➡️ Ponto de Virada (IPV): ', min_value = 0.0, max_value = 10.0, value=0.0, step = 0.01, format='%.1f'))

st.markdown(
    '''
    <div style="padding: 20px 0;">
        <h3>5º Passo | Índice de Desenvolvimento Educacional </h3>
        <p style="text-align: justify; line-height: 1.6"><strong>Para finalizar, nada melhor que informar o INDE do estudante.</strong> Caso você não tenha as informações dos anos anteriores, sempre problemas, só deixar o campo em branco, ok?</p>
    </div>
    ''', unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    inde_atual = st.number_input(label='⚖️ INDE (atual): ', min_value = 0.0, max_value = 10.0, step = 0.01, value=None, format='%.1f')
    inde_atual = np.nan if inde_atual == None else float(inde_atual)

with col2:
    inde_anterior = st.number_input(label='➡️ INDE (ano anterior): ', min_value = 0.0, max_value = 10.0, step = 0.01, value=None, format='%.1f')
    inde_anterior = np.nan if inde_anterior == None else float(inde_anterior)

with col3:
    inde_dois_anos = st.number_input(label='➡️ INDE (há dois anos): ', min_value = 0.0, max_value = 10.0, step = 0.01, value=None, format='%.1f')
    inde_dois_anos = np.nan if inde_dois_anos == None else float(inde_dois_anos)


estudante = [input_sexo, inde_dois_anos, inde_anterior, inde_atual, ida, ieg, iaa, ips, ipp, ipv, 0]

def data_split(df, test_size):
    SEED = 42
    treino_df, teste_df = train_test_split(df, test_size=test_size, random_state=SEED)
    return treino_df.reset_index(drop=True), teste_df.reset_index(drop=True)

treino_df, teste_df = data_split(df, 0.3)

#Criando novo paciente
estudante_predict_df = pd.DataFrame([estudante], columns=teste_df.columns)
print(estudante_predict_df['risco'])

#Concatenando novo estudante ao dataframe dos dados de teste
teste_novo_estudante  = pd.concat([teste_df, estudante_predict_df], ignore_index=True)

estudante_predito = teste_novo_estudante.drop(['risco'], axis=1)

if st.button(label='Exibir Resultado de Risco de Desafagem', icon="🔥", type='primary', width="stretch"):
    acao_botao(estudante)
