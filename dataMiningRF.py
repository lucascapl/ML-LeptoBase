# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, MinMaxScaler
import numpy as np

def tratar_planilha(df):
    df.replace(' ', "", inplace=True) # Remove os espaços em branco
    df = df.apply(pd.to_numeric, errors='coerce') # Força todas as células a serem numéricas
    return df

def remover_coluna_sem_relacao(df):
    df.drop(df.columns[[3, 4, 5]], axis=1, inplace=True) # Remoção das colunas de lixo entulho, sinais roedores, contato água/lama
    return df

def remover_outliers(df):
    # Remoção de outliers das 3 primeiras colunas através do IQR
    for column in df.columns[:3]:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2.5 * IQR
        upper_bound = Q3 + 2.5 * IQR
        df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), np.nan, df[column])
    return df

def preenche_missing_value(df):
    # Preenchimento de missing values
    for column in df.columns[:3]: # Apenas as 3 primeiras colunas
        media = df[column].median()
        df[column] = df[column].fillna(media) # Preencher os missing values com a mediana de cada coluna
    for column in df.columns[:-1]:
        if column not in df.columns[:3]: # Preenchendo com zero a coluna dos sintomas onde estiver com missing value
            df[column] = df[column].fillna(0)
    return df

def normalizar_dados(df):
    scaler = MinMaxScaler()
    df[df.columns[:3]] = scaler.fit_transform(df[df.columns[:3]])  # Normalizando apenas as 3 primeiras colunas contínuas
    return df

def discretizar_colunas(df):
    est = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='quantile')
    df[df.columns[:3]] = est.fit_transform(df[df.columns[:3]])
    return df

def gera_nova_planilha(df):
    # Gera nova planilha para ver o resultado da base que será processada
    new_file_path = 'lepto_base_limpa.xlsx'
    df.to_excel(new_file_path, index=False)

file_path = 'lepto_base.xlsx' # Carrega a base original
df = pd.read_excel(file_path, sheet_name='base2') # Folha contendo os dados

df = tratar_planilha(df) # Função para tratar a planilha inicialmente
df = remover_coluna_sem_relacao(df) # Função para remover as colunas que não possuem relação de causa e efeito com o que queremos prever
df = remover_outliers(df) # Função para remover outliers
df = preenche_missing_value(df) # Função para preencher missing values
df = normalizar_dados(df) # Função para normalizar os dados contínuos
df = discretizar_colunas(df) # Função para discretizar colunas

gera_nova_planilha(df) # Função para gerar nova planilha

feature_names = df.columns[:-1] # Coluna de atributos
target_name = df.columns[-1] # Coluna de target (quem queremos prever)

X = df[feature_names]
Y = df[target_name]

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.3)

# Aplicação do SMOTE apenas no conjunto de treino
smote = SMOTE(random_state=0, sampling_strategy="not majority")
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

# Definir os parâmetros para a busca em grade para RandomForest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': [None, 'balanced']
}

# Configurando a busca em grade para RandomForest
grid_search = GridSearchCV(RandomForestClassifier(random_state=0), param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)

# Treinar o modelo usando a busca em grade
grid_search.fit(X_train_resampled, Y_train_resampled)

# Obter o melhor modelo
best_clf = grid_search.best_estimator_

# Prever nos dados de teste
Y_pred = best_clf.predict(X_test)

# Calcular a acurácia
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Acurácia: {accuracy:.2f}')

# Calcular o F1-score
f1 = f1_score(Y_test, Y_pred, average='weighted')
print(f'F1-score: {f1:.2f}')

# Imprimir relatório de classificação
report = classification_report(Y_test, Y_pred, target_names=['Morreu', 'Viveu'])
print('Relatório de Classificação:\n', report)

# Imprimir os melhores hiperparâmetros
print('Melhores hiperparâmetros:', grid_search.best_params_)

# Calcular a matriz de confusão
conf_matrix = confusion_matrix(Y_test, Y_pred)

# Exibir a matriz de confusão
print('Matriz de Confusão:')
print(conf_matrix)
