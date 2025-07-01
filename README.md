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

## 📁 Arquivos

- `main.py`: Código principal contendo pipeline, otimização e avaliação
- `dadosatt1.xlsx`: Conjunto de dados com 160 amostras (não incluso por privacidade)

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
