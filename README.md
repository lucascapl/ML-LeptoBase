# DataMining-LeptospiroseBase

## PT-BR

Trabalho de avaliação da disciplina de data mining do curso de Ciência da Computação na UERJ com a professora Karla Figueiredo

**Objetivo:** Criar um modelo de classificação de pacientes acometidos por leptospirose a fim de tentar prever o destino de sua cura.

### Técnicas Utilizadas:
- **Descrição geral dos atributos:**
  - Cardinalidade
  - Escala
  - Numérico ou categórico
- **Descrição detalhada:**
  - Frequência
  - Valores mínimos e máximos
  - Desvio padrão
  - Porcentagem de dados faltantes
  - Média
  - Mediana
  - Proporção de positivos e negativos (para dados binários)
- **Limpeza de dados (outliers e missing):**
  -  Inter Quartile Range para remoção de outliers de dados contínuos
  - Preenchimento de dados faltantes (mediana para dados contínuos e moda para dados binários)
  - Redução de atributos sem relação de causa e efeito com a classificação do paciente
- **SMOTE para balanceamento da saída após o pré-processamento**
- **GridSearchCV para validação cruzada**

### Resultados:
- Acurácia: 69%
- Confira os resultados no arquivo "resultado.png"

---

## EN-US

Data mining evaluation project for the Computer Science course at UERJ.

**Objective:** Create a classification model for patients affected by leptospirosis to predict their recovery outcome.

### Techniques Used:
- **General Description of Attributes:**
  - Cardinality
  - Scale
  - Numeric or categorical
- **Detailed Description:**
  - Frequency
  - Minimum and maximum values
  - Standard deviation
  - Percentage of missing data
  - Mean
  - Median
  - Proportion of positives and negatives (for binary data)
- **Data Cleaning (outliers and missing):**
  - Z-Score for outlier removal in continuous data
  - Imputation of missing data (median for continuous data and mode for binary data)
  - Reduction of attributes unrelated to the classification of the patient
- **SMOTE for output balancing after preprocessing**
- **GridSearchCV for cross validation**

### Results:
- **Accuracy:** 69%
- Check the visual results in the file "resultado.png"
