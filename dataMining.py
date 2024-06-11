# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import numpy as np

def tratar_planilha(df):
    df.replace(' ', "", inplace=True) #remove os espacos em branco
    df = df.apply(pd.to_numeric, errors='coerce') #forca todas as celulas a serem numericas

    return df

def remover_coluna_sem_relacao(df):
    df.drop(df.columns[[3, 4, 5]], axis=1, inplace=True) #remocao das colunas de lixo entulho, sinais roedores, contato agua/lama
    return df


def remover_outliers(df): #remocao de outliers das 3 primeiras colunas atraves de z-score
    for column in df.columns[:3]:
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        df[column] = np.where(np.abs(z_scores) > 3, np.nan, df[column])
    return df

def preenche_missing_value(df): #preenchimento de missing
    for column in df.columns[:3]: # Apenas as 3 primeiras colunas
        median = df[column].median()
        df[column] = df[column].fillna(median) # Preencher os missing values com a mediana de cada coluna
    for column in df.columns[:-1]:
        if column not in df.columns[:3]: # Preenchendo com zero a coluna dos sintomas onde estiver com missing value
            df[column] = df[column].fillna(0)
    return df

def gera_nova_planilha(df): #gera nova planilha pra ver o resultado da base que sera processada
    new_file_path = 'lepto_base_limpa.xlsx'
    df.to_excel(new_file_path, index=False)



file_path = 'lepto_base.xlsx' #carrega a base original
df = pd.read_excel(file_path, sheet_name='base2') #folha contendo os dados

df = tratar_planilha(df) #funcao para tratar a planilha inicialmente
df = remover_coluna_sem_relacao(df) #funcao para remover as colunas que nao possuem relacao de causa e efeito com o que queremos prever
df = remover_outliers(df) #funcao para remover outliers
df = preenche_missing_value(df) #funcao para preencher missing

gera_nova_planilha(df) #funcao para gerar nova planilha

feature_names = df.columns[:-1] #coluna de atributos
target_name = df.columns[-1] #coluna de target (quem queremos prever)

X = df[feature_names]
Y = df[target_name]

# Aplicação do SMOTE para rebalancear as classes de target
smote = SMOTE(random_state=0)
X_resampled, Y_resampled = smote.fit_resample(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, random_state=0, test_size=0.4)

# Definir os parâmetros para a busca em grade
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10, 20],
    'max_features': [None, 'sqrt', 'log2', 4]
}

# Configurando a busca em grade
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=0), param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Treinar o modelo usando a busca em grade
grid_search.fit(X_train, Y_train)

# Obter o melhor modelo
best_clf = grid_search.best_estimator_

# Prever nos dados de teste
Y_pred = best_clf.predict(X_test)

# Calcular a acurácia
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Acurácia: {accuracy:.2f}')

# Imprimir relatório de classificação
report = classification_report(Y_test, Y_pred, target_names=['Viveu', 'Morreu'])
print('Relatório de Classificação:\n', report)

# Imprimir os melhores hiperparâmetros
print('Melhores hiperparâmetros:', grid_search.best_params_)

#matriz de confusao gerada pelo chat gpt

# Prever nos dados de teste
Y_pred = best_clf.predict(X_test)

# Calcular a matriz de confusão
conf_matrix = confusion_matrix(Y_test, Y_pred)

# Exibir a matriz de confusão
print('Matriz de Confusão:')
print(conf_matrix)