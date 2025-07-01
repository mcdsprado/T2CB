#Trabalho para a disciplina de computacao bioinspirada :)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

#Carregar os dados:
df = pd.read_excel('dadosatt1.xlsx')

#Definicao das saidas
X = df[['Orgão', 'Técnica', 'OED', 'EAR', 'V70%']]
risk_cols = ['OED Risco', 'EAR Risco', 'V70% Risco']
y_df = df[risk_cols].copy()

#Codificacao binaria
y_enc = np.zeros((len(df), len(risk_cols)), dtype=int)
for i, col in enumerate(risk_cols):
    le = LabelEncoder()
    y_enc[:, i] = le.fit_transform(y_df[col])

#treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42
)

#preprocessamento
cat_cols = ['Orgão', 'Técnica']
num_cols = ['OED', 'EAR', 'V70%']
preproc = ColumnTransformer([
    ('cat', OneHotEncoder(sparse_output=False, drop='if_binary'), cat_cols),
    ('num', StandardScaler(), num_cols),
])

#Pipeline
base_mlp = MLPClassifier(max_iter=500, random_state=42)
multi_mlp = MultiOutputClassifier(base_mlp)
pipe = Pipeline([
    ('pre', preproc),
    ('clf', multi_mlp),
])

#fazer um gridsearch com 3 fold CV
param_grid = {
    'clf__estimator__hidden_layer_sizes': [(n,) for n in [5, 10, 20, 50]],
    'clf__estimator__learning_rate_init': [0.001, 0.01, 0.1],
    'clf__estimator__activation': ['relu', 'tanh', 'logistic'],
}

cv = KFold(n_splits=3, shuffle=True, random_state=42)
gs = GridSearchCV(
    pipe,
    param_grid,
    cv=cv,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy',  # subset accuracy para multi-saída
    refit=True
)
gs.fit(X_train, y_train)

#extrair resultados e salvar:
results_df = pd.DataFrame(gs.cv_results_)[[
    'param_clf__estimator__hidden_layer_sizes',
    'param_clf__estimator__learning_rate_init',
    'param_clf__estimator__activation',
    'mean_test_score',
    'std_test_score'
]]
best_idx  = results_df['mean_test_score'].idxmax()
worst_idx = results_df['mean_test_score'].idxmin()
results_df['is_best']  = False
results_df['is_worst'] = False
results_df.loc[best_idx,  'is_best']  = True
results_df.loc[worst_idx, 'is_worst'] = True
results_df.to_csv('mlp_grid_results.csv', index=False)

#selecionar best and worst apenas:

best_params  = results_df.loc[best_idx]
worst_params = results_df.loc[worst_idx]
pd.DataFrame([best_params, worst_params]) \
  .to_csv('mlp_best_worst.csv', index=False)

#treinar para o pior caso
worst_mlp = MLPClassifier(
    hidden_layer_sizes=tuple(worst_params['param_clf__estimator__hidden_layer_sizes']),
    learning_rate_init=float(worst_params['param_clf__estimator__learning_rate_init']),
    activation=worst_params['param_clf__estimator__activation'],
    max_iter=500,
    random_state=42
)
worst_model = Pipeline([
    ('pre', preproc),
    ('clf', MultiOutputClassifier(worst_mlp)),
])
worst_model.fit(X_train, y_train)

#matriz confusao e predicao:
best_model  = gs.best_estimator_
y_pred_best  = best_model.predict(X_test)
y_pred_worst = worst_model.predict(X_test)

for title, y_pred in [('Melhor Modelo', y_pred_best), ('Pior Modelo', y_pred_worst)]:
    for i, col_name in enumerate(risk_cols):
        cm = confusion_matrix(y_test[:, i], y_pred[:, i])
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, interpolation='nearest')
        for (row, col), val in np.ndenumerate(cm):
            plt.text(col, row, val, ha='center', va='center', fontsize=16, fontweight='bold')
        plt.title(f'{title} — {col_name}')
        plt.xlabel('Previsto', fontsize=19)
        plt.ylabel('Verdadeiro', fontsize=19)
        plt.xticks([0, 1], ['BAIXO', 'ALTO'], fontsize=18)
        plt.yticks([0, 1], ['BAIXO', 'ALTO'], fontsize=18)
        plt.tight_layout()
        plt.show()

#plot loss x epoca
for title, model in [('Melhor Modelo', best_model), ('Pior Modelo', worst_model)]:
    clf = model.named_steps['clf']
    plt.figure(figsize=(8, 5))
    for i, est in enumerate(clf.estimators_):
        if hasattr(est, 'loss_curve_'):
            plt.plot(est.loss_curve_, label=risk_cols[i])
    plt.title(f'Loss x Épocas ({title})', fontsize=16)
    plt.xlabel('Época', fontsize=19)
    plt.ylabel('Training Loss', fontsize=19)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.show()


#Print das principais métricas best and worst:

for title, y_pred in [('Melhor Modelo', y_pred_best), ('Pior Modelo', y_pred_worst)]:
    print(f"\n===== {title} =====")
    for i, col_name in enumerate(risk_cols):
        print(f"\n--- {col_name} ---")
        print(classification_report(
            y_test[:, i],
            y_pred[:, i],
            target_names=['BAIXO', 'ALTO'],
            zero_division=0
        ))