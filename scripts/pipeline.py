#libs
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier

#modelos
from sklearn.linear_model import LogisticRegression
import joblib

#VARIÁVEIS GLOBAIS
SEED = 42


#CRIAÇÃO DE DATAFRAME
def df_importacao_dados():
    df_2022 = pd.read_excel(r'datasets\raw.xlsx', sheet_name='PEDE2022')
    df_2023 = pd.read_excel(r'datasets\raw.xlsx', sheet_name='PEDE2023')
    df_2024 = pd.read_excel(r'datasets\raw.xlsx', sheet_name='PEDE2024')

    return (df_2022, df_2023, df_2024)

def df_selecionar_colunas(dfs):
    df_2022, df_2023, df_2024 = dfs
    df_2022 = df_2022[['ano', 'Idade', 'Gênero', 'INDE 22', 'IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN', 'Fase ideal', 'Defas']]
    df_2023 = df_2023[['ano', 'Idade', 'Gênero', 'INDE 22', 'INDE 23', 'IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV', 'IAN', 'Fase Ideal', 'Defasagem']]
    df_2024 = df_2024[['ano', 'Idade', 'Gênero', 'INDE 22', 'INDE 23', 'INDE 2024', 'IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV', 'IAN', 'Fase Ideal', 'Defasagem']]
    return (df_2022, df_2023, df_2024)

def df_criar_colunas(dfs):
    df_2022, df_2023, df_2024 = dfs
    df_2022['ano'] = 2022 
    df_2023['ano'] = 2023 
    df_2024['ano'] = 2024 
    return (df_2022, df_2023, df_2024)

def df_padronizar_nome_coluna(dfs):
    df_2022, df_2023, df_2024 = dfs
    df_2022.columns = ['ano', 'idade', 'genero', 'inde_22', 'iaa', 'ieg', 'ips', 'ida', 'ipv', 'ian', 'fase_ideal', 'defasagem']
    df_2023.columns= ['ano', 'idade', 'genero', 'inde_22', 'inde_23', 'iaa', 'ieg', 'ips', 'ipp', 'ida', 'ipv', 'ian', 'fase_ideal', 'defasagem']
    df_2024.columns = ['ano', 'idade', 'genero', 'inde_22', 'inde_23', 'inde_24', 'iaa', 'ieg', 'ips', 'ipp', 'ida', 'ipv', 'ian', 'fase_ideal', 'defasagem']
    return (df_2022, df_2023, df_2024)

def df_concatenar_dfs(dfs):
    df_2022, df_2023, df_2024 = dfs
    df = pd.concat([df_2022, df_2023, df_2024])
    return df

def padronizar_genero(genero):
    if genero in ('Menina', 'Feminino'):
        return 0
    elif genero in ('Menino', 'Masculino'):
        return 1
    
def padronizar_fase_ideal(fase_ideal):
    dict_fase_ideal = {
        'ALFA  (2º e 3º ano)' : 0,
        'ALFA (1° e 2° ano)' : 0,
        'Fase 1 (3° e 4° ano)' : 1,
        'Fase 1 (4º ano)' : 1,
        'Fase 2 (5° e 6° ano)' : 2,
        'Fase 2 (5º e 6º ano)' : 2,
        'Fase 3 (7° e 8° ano)' : 3,
        'Fase 3 (7º e 8º ano)' : 3,
        'Fase 4 (9° ano)' : 4,
        'Fase 4 (9º ano)' : 4,
        'Fase 5 (1° EM)' : 5,
        'Fase 5 (1º EM)' : 5,
        'Fase 6 (2° EM)' : 6,
        'Fase 6 (2º EM)' : 6,
        'Fase 7 (3° EM)' : 7,
        'Fase 7 (3º EM)' : 7,
        'Fase 8 (Universitários)':8
    }
    
    return dict_fase_ideal[fase_ideal]
    
def padronizar_dados_coluna(df):
    df['genero'] = df['genero'].map(lambda x : padronizar_genero(x))
    df['fase_ideal'] = df['fase_ideal'].map(lambda x : padronizar_fase_ideal(x))
    df['inde_24'] = df['inde_24'].replace("INCLUIR", np.nan)
    df['risco'] = df['ian'].map(lambda x : 1 if x <= 5 else 0)
    return df

def padronizar_casas_decimais(df):
    df[['inde_22', 'iaa', 'ieg', 'ips', 'ida', 'ipv', 'inde_23', 'ipp', 'inde_24']] = df[['inde_22', 'iaa', 'ieg', 'ips', 'ida', 'ipv', 'inde_23', 'ipp', 'inde_24']].round(2)
    return df

def ordenar_colunas(df):
    df = df[['ano', 'idade', 'genero', 'inde_22', 'inde_23', 'inde_24', 'iaa','ieg', 'ips', 'ida', 'ipv', 'ian', 'ipp', 'fase_ideal', 'defasagem', 'risco']]
    return df

def exportar_df(df):
    df.to_excel(r'datasets\clean.xlsx', index=False)

def df_criar_dataframe():
    df = df_importacao_dados()
    df = df_criar_colunas(df)
    df = df_selecionar_colunas(df)
    df = df_padronizar_nome_coluna(df)
    df = df_concatenar_dfs(df)
    df = padronizar_dados_coluna(df)
    df = padronizar_casas_decimais(df)
    df = ordenar_colunas(df)
    exportar_df(df)
    return df



#-----------------------------------RESPOSTAS PARA O QUESTIONÁRIO-------------------------------------------------#


def questao_01(df):

    # 1. Adequação do nível (IAN): Qual é o perfil geral de defasagem dos alunos (IAN) e como ele evolui ao longo do ano?
    ian_resultado_medio = df.groupby('ano')['ian'].mean()
    ian_resultado_medio_genero = df.groupby(['ano', 'genero'])['ian'].mean()

    print(df['ian'].describe())
    print(ian_resultado_medio)
    print(ian_resultado_medio_genero)

def questao_02(df):
    # 2. Desempenho acadêmico (IDA): O desempenho acadêmico médio (IDA) está melhorando, estagnado ou caindo ao longo das fases e anos?
    #Feito no Power BI
    pass

def questao_03(df):
    # 3. Engajamento nas atividades (IEG): O grau de engajamento dos alunos (IEG) tem relação direta com seus indicadores de desempenho (IDA) e do ponto de virada (IPV)?
    correlacao = df[['ieg','ida','ipv']].corr()
    print("Correlação entre IEG, IDA e IPV:")
    print(correlacao)

    df['ieg_q'] = pd.qcut(df['ieg'], 4, labels=['1Q','2Q', '3Q', '4Q'])
    media_por_faixa_ida = df.groupby('ieg_q')['ida'].mean()
    media_por_faixa_ipv = df.groupby('ieg_q')['ipv'].mean()
    
    print(media_por_faixa_ida)
    print(media_por_faixa_ipv)
    
    #Plotar os dois resultados em gráfico
    df = pd.concat([media_por_faixa_ida, media_por_faixa_ipv], axis=1)
    df = df.reset_index().rename(columns={'index': 'ieg_q'})
    df_melt = df.melt(id_vars='ieg_q', 
                  value_vars=['ida', 'ipv'], 
                  var_name='categoria', 
                  value_name='valor')

    ax = sns.barplot(data=df_melt, x='ieg_q', y='valor', hue='categoria', palette='flare')
    for container in ax.containers:
        ax.bar_label(container, label_type='edge', padding=3)

    plt.show()

def questao_04(df):
    # 4. Autoavaliação (IAA): As percepções dos alunos sobre si mesmos (IAA) são coerentes com seu desempenho real (IDA) e engajamento (IEG)?
    print(df[['iaa','ida','ieg']].corr())
    df['iaa_q'] = pd.qcut(df['iaa'], 4, labels=['1Q','2Q', '3Q', '4Q'])

    ida_iaa_avg = df.groupby('iaa_q')['ida'].mean()
    ieg_iaa_avg = df.groupby('iaa_q')['ieg'].mean()

    #Plotar os dois resultados em gráfico
    df = pd.concat([ida_iaa_avg, ieg_iaa_avg], axis=1)
    df = df.reset_index().rename(columns={'index': 'iaa_q'})
    df_melt = df.melt(id_vars='iaa_q', 
                  value_vars=['ida', 'ieg'], 
                  var_name='categoria', 
                  value_name='valor')

    ax = sns.barplot(data=df_melt, x='iaa_q', y='valor', hue='categoria', palette='magma')
    for container in ax.containers:
        ax.bar_label(container, label_type='edge', padding=3)

    plt.show()

    print(ieg_iaa_avg)
    print(ida_iaa_avg)

def questao_05(df):
    # 5. Aspectos psicossociais (IPS): Há padrões psicossociais (IPS) que antecedem quedas de desempenho acadêmico ou de engajamento?
    df['ips_q'] = pd.qcut(df['ips'], 4, labels=['1Q','2Q', '3Q', '4Q'])

    ida_ips_avg = df.groupby('ips_q')['ida'].mean()
    ieg_ips_avg = df.groupby('ips_q')['ieg'].mean()

    print(df[['ips','ida','ieg', 'ian', 'risco']].corr())
    print(ieg_ips_avg)
    print(ida_ips_avg)
    
    #Plotar os dois resultados em gráfico
    df = pd.concat([ida_ips_avg, ieg_ips_avg], axis=1)
    df = df.reset_index().rename(columns={'index': 'ips_q'})
    df_melt = df.melt(id_vars='ips_q', 
                  value_vars=['ida', 'ieg'], 
                  var_name='categoria', 
                  value_name='valor')

    ax = sns.barplot(data=df_melt, x='ips_q', y='valor', hue='categoria', palette='crest')
    for container in ax.containers:
        ax.bar_label(container, label_type='edge', padding=3)

    plt.show()



def questao_06(df):
    # 6. Aspectos psicopedagógicos (IPP): As avaliações psicopedagógicas (IPP) confirmam ou contradizem a defasagem identificada pelo IAN?
    df['ipp_q'] = pd.qcut(df['ipp'], 4, labels=['1Q','2Q', '3Q', '4Q'])

    ipp_ian_avg = df.groupby('ian')['ipp'].mean()
    ipp_defasagem_avg = df.groupby('defasagem')['ipp'].mean()
    ipp_analise_risco = df.groupby('risco')['ipp'].mean()
    ax = sns.barplot(data=ipp_defasagem_avg, palette='crest')
    for container in ax.containers:
         ax.bar_label(container, label_type='edge', padding=3)
    plt.show()

    print(df[['ipp','ian']].corr())
    print(ipp_ian_avg)
    print(ipp_defasagem_avg)
    print(ipp_analise_risco)   

def questao_07(df):
    # 7. Ponto de virada (IPV): Quais comportamentos - acadêmicos, emocionais ou de engajamento - mais influenciam o IPV ao longo do tempo?
    corr = df[['ipv','ida','ieg','iaa','ips','ipp']].corr()
    print(corr)
    print(corr['ipv'].sort_values(ascending=False))    
    
    ipv_correlacao = corr['ipv'].sort_values(ascending=False)
    
    ax = sns.barplot(data=ipv_correlacao, palette='crest')
    for container in ax.containers:
         ax.bar_label(container, label_type='edge', padding=3)
    
    plt.show()

def questao_08(df):
    corr = df[['inde_24', 'ipv','ida','ieg','iaa','ips','ipp']].corr()
    corr_inde = corr['inde_24'].sort_values(ascending=False)
    ax = sns.barplot(data=corr_inde, palette='flare')
    for container in ax.containers:
         ax.bar_label(container, label_type='edge', padding=3)
    plt.show()
    
def questao_09():
    #MODELO
    pass

def questao_10(df):
    indicadores = ['ida','ieg','ips','ipp','ipv','ian']
    evolucao_ano = df.groupby('ano')[indicadores].mean()
    evolucao_ano = evolucao_ano.T
    print(evolucao_ano)
    
#--------------------------------------------------------------------------------------------------------#

#FUNÇÕES PARA CRIAÇÃO DO MODELO

# BASE DE TESTE E TREINO
def criar_base_teste_treino(df):
    df = df[['genero', 'inde_22', 'inde_23', 'inde_24', 'ida', 'ieg', 'iaa', 'ips', 'ipp', 'ipv', 'risco']]
    x = df.drop('risco', axis=1)
    y = df['risco']
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
    return (x_treino, x_teste, y_treino, y_teste)

def criar_modelo():
    modelo = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', RandomForestClassifier(n_estimators=300, random_state=42))
    ])
    return modelo

def analisar_modelo_ml(modelo, x_treino, y_treino, x_teste, y_teste):
    modelo.fit(x_treino, y_treino)
    y_pred = modelo.predict(x_teste)
    y_proba = modelo.predict_proba(x_teste)[:, 1]

    # ------------- Resultado das Métricas -------------
    acuracia = accuracy_score(y_teste, y_pred)
    precisao = precision_score(y_teste, y_pred)
    recall = recall_score(y_teste, y_pred)
    score_f1 = f1_score(y_teste, y_pred)
    score_auc = roc_auc_score(y_teste, y_proba)

    # Apresentar graficamente a matriz de confusão
    matriz = confusion_matrix(y_teste, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz)
    disp.plot(cmap=plt.cm.Blues)
    plt.xlabel('Label Predita', fontsize=18)
    plt.ylabel('Label Verdadeira', fontsize=18)
    plt.title("Matriz de Confusão")
    plt.show()

    # Apresentar os resultados da Classification Report
    predicao = modelo.predict(x_teste)
    print(f"\n|------------- CLASSIFICATION REPORT -------------|")
    print(classification_report(y_teste, predicao))
    
    # Apresentar graficamente a Curva ROC
    RocCurveDisplay.from_predictions(y_teste, y_proba, name=f"Resultado AUC = {score_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="red")
    plt.tight_layout()
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.show()

    # Apresentação dos Resultados do modelo
    print(f"\n|------------- RESULTADOS DO MODELO -------------|")
    print(f"Acurácia do modelo : {acuracia:.2f}")
    print(f"Precisão do modelo : {precisao:.2f}")
    print(f"Recall   : {recall:.2f}")
    print(f"F1-score : {score_f1:.2f}")
    print(f"AUC-ROC  : {score_auc:.2f}")
    
    # Apresentar a importância das variáveis
    fi = modelo.named_steps['model'].feature_importances_
    ordenamento = pd.DataFrame(fi, index=['genero', 'inde_22', 'inde_23', 'inde_24', 'ida', 'ieg', 'iaa', 'ips', 'ipp', 'ipv'], columns=['resultado'])
    ordenamento = ordenamento.sort_values(ascending=False, by='resultado')
    plt.bar(ordenamento.index, ordenamento['resultado'])
    for i, v in enumerate(ordenamento['resultado']):
        plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    plt.show()

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

def main():
    df = df_criar_dataframe()
    x_treino, x_teste, y_treino, y_teste = criar_base_teste_treino(df)
    modelo = criar_modelo()
    analisar_modelo_ml(modelo, x_treino, y_treino, x_teste, y_teste)
    probabilidade_risco(modelo, x_teste)
    joblib.dump(modelo, r'model\xgb.joblib')
    print(df)
    
main()
