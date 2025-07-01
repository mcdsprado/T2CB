# 🧠 Predição de Risco com Redes Neurais MLP

Este projeto aplica redes neurais do tipo **Perceptron Multicamadas (MLP)** para classificar o risco de reincidência de câncer com base em variáveis clínicas. O objetivo é comparar o desempenho de diferentes configurações da rede usando uma abordagem de **multi-saída (multi-output)** com validação cruzada e otimização de hiperparâmetros.

---

## 📊 Descrição do Problema

O modelo prevê o risco (BAIXO ou ALTO) de três métricas clínicas:
- **OED Risco**
- **EAR Risco**
- **V70% Risco**

Com base em cinco variáveis de entrada:
- **Órgão**
- **Técnica**
- **OED**
- **EAR**
- **V70%**

---

## ⚙️ Pipeline de Implementação

1. **Pré-processamento:**
   - One-Hot Encoding das variáveis categóricas (`Órgão`, `Técnica`)
   - Padronização (z-score) das variáveis numéricas (`OED`, `EAR`, `V70%`)
   - Codificação binária das saídas (0 = BAIXO, 1 = ALTO)

2. **Modelagem:**
   - Uso do `MLPClassifier` com `MultiOutputClassifier`
   - Arquitetura com uma única camada oculta

3. **Otimização:**
   - `GridSearchCV` com validação cruzada (3-fold)
   - Parâmetros ajustados:
     - Tamanho da camada oculta: `(5,)`, `(10,)`, `(20,)`, `(50,)`
     - Taxa de aprendizado: `0.001`, `0.01`, `0.1`
     - Funções de ativação: `relu`, `tanh`, `logistic`

4. **Avaliação:**
   - Comparação entre o **melhor** e o **pior** modelo encontrados
   - Geração de matrizes de confusão
   - Curvas de treinamento (loss × épocas)
   - Métricas: precisão, recall, F1-score, acurácia

---

## 📁 Arquivos

- `main.py`: Código principal contendo pipeline, otimização e avaliação
- `dadosatt1.xlsx`: Conjunto de dados com 160 amostras (não incluso por privacidade)
- `mlp_grid_results.csv`: Resultados da validação cruzada
- `mlp_best_worst.csv`: Hiperparâmetros dos melhores e piores modelos
- Imagens: Gráficos gerados automaticamente (matriz de confusão e curvas de loss)

---

## ✅ Resultados

- O **melhor modelo** obteve **89.8%** de acurácia na validação cruzada.
- O **pior modelo**, com mesma arquitetura, mas hiperparâmetros inadequados, teve apenas **57.1%**.
- As métricas mostraram ganho significativo principalmente na sensibilidade (recall) da variável **V70% Risco**.

---

## 🚧 Limitações

- Conjunto de dados reduzido (160 amostras)
- Modelos simples, com apenas uma camada oculta
- Resultados preliminares, mas promissores

---

## 📌 Requisitos

- Python 3.8+
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- openpyxl (para ler arquivos .xlsx)

Instale as dependências com:

```bash
pip install -r requirements.txt
