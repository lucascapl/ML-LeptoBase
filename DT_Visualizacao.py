# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
from imblearn.over_sampling import SMOTE

file_path = 'lepto_base_original_sem_outlier_mediana_sem_lixo_entulho.xlsx'  # base de dados
df = pd.read_excel(file_path, sheet_name='base2')

# Limpeza de dados
# Remover linhas com valores ausentes
df.dropna(inplace=True)

# Tentar converter todas as colunas para numérico, ignorando erros
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Remover novamente linhas que possam ter se tornado NaN após a conversão
df.dropna(inplace=True)

feature_names = df.columns[:-1]
target_name = df.columns[-1]

X = df[feature_names]
Y = df[target_name]

# Aplicar SMOTE para rebalancear as classes
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

# Configurar a busca em grade
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

# Plotar a árvore de decisão
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 20), dpi=600)
tree.plot_tree(
    best_clf,
    feature_names=feature_names,
    class_names=best_clf.classes_.astype(str),
    filled=True
)
plt.show()

# Imprimir os melhores hiperparâmetros
print('Melhores hiperparâmetros:', grid_search.best_params_)
