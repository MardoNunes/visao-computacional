
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")              # só salva as figuras (troque por "TkAgg" se quiser janelas)
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ARQUIVO_IRIS = os.path.join(os.path.dirname(__file__), "irisDataSet", "iris.data")
# Para usar a versão corrigida de Bezdek, troque para "bezdekIris.data"

SAIDA = "saida_iris"
os.makedirs(SAIDA, exist_ok=True)
SEED = 42                           # fixa o sorteio -> resultados reproduzíveis (como nos slides)

# --------------------------------------------------------------------------------------
# 0. CARREGAR O DATASET
# --------------------------------------------------------------------------------------
# O dataset local irisDataSet/iris.data é o mesmo Iris (Fisher) do repositório UCI,
# no formato CSV sem cabeçalho: 4 colunas numéricas + "Iris-setosa|versicolor|virginica".
# (Equivalente ao antigo load_iris() do scikit-learn, sem precisar baixar nada.)
COLUNAS = ["comp_sepala", "larg_sepala", "comp_petala", "larg_petala"]
NOMES = COLUNAS + ["especie"]
df = pd.read_csv(ARQUIVO_IRIS, header=None, names=NOMES)
df = df.dropna(how="any")  # iris.data termina com linha em branco -> remove a linha NaN
# Normaliza "Iris-setosa" -> "setosa" (mesmo padrão do load_iris)
df["especie"] = df["especie"].str.replace("Iris-", "", case=False).str.lower()
CLASSES = ["setosa", "versicolor", "virginica"]  # ordem fixa = mesma do load_iris
MAPA_CLASSES = {nome: i for i, nome in enumerate(CLASSES)}
CORES = {"setosa": "tab:blue", "versicolor": "tab:orange", "virginica": "tab:green"}

print("=" * 70)
print("Dataset:", df.shape[0], "amostras,", len(COLUNAS), "variáveis,", len(CLASSES), "classes")
print("=" * 70)

# --------------------------------------------------------------------------------------
# 1. ESTATÍSTICAS DESCRITIVAS
# --------------------------------------------------------------------------------------
def resumo(tabela):
    """Devolve média, desvio e moda de cada coluna numérica."""
    return pd.DataFrame({
        "media": tabela[COLUNAS].mean(),
        "desvio": tabela[COLUNAS].std(),
        "moda": tabela[COLUNAS].mode().iloc[0],   # mode() pode devolver vários valores; pegamos o 1º
    })

print("\n### 1. Estatísticas GERAIS")
print(resumo(df).round(3))

for classe in CLASSES:
    print(f"\n### Estatísticas da classe: {classe}")
    print(resumo(df[df["especie"] == classe]).round(3))

print("\n### Frequência de cada categoria")
print(df["especie"].value_counts())

# --------------------------------------------------------------------------------------
# 2. RELAÇÃO COMPRIMENTO x LARGURA (sépala e pétala), POR CATEGORIA
# --------------------------------------------------------------------------------------
print("\n### 2. Correlação (Pearson) comprimento x largura, por categoria")
fig, eixos = plt.subplots(1, 2, figsize=(12, 5))
for eixo, parte in zip(eixos, ["sepala", "petala"]):
    col_c, col_l = f"comp_{parte}", f"larg_{parte}"
    for classe in CLASSES:
        sub = df[df["especie"] == classe]
        r = sub[col_c].corr(sub[col_l])                      # coeficiente de correlação
        print(f"  {parte:7s} | {classe:11s} | r = {r:+.3f}")
        eixo.scatter(sub[col_c], sub[col_l], label=f"{classe} (r={r:.2f})",
                     color=CORES[classe], alpha=0.7)
        # reta de regressão linear simples (np.polyfit) para ver a tendência
        a, b = np.polyfit(sub[col_c], sub[col_l], 1)
        xs = np.array([sub[col_c].min(), sub[col_c].max()])
        eixo.plot(xs, a * xs + b, color=CORES[classe], linewidth=1)
    eixo.set_xlabel(f"Comprimento da {parte} (cm)")
    eixo.set_ylabel(f"Largura da {parte} (cm)")
    eixo.set_title(f"{parte.capitalize()}: comprimento x largura")
    eixo.legend()
plt.tight_layout()
plt.savefig(f"{SAIDA}/02_comprimento_x_largura.png", dpi=120)
plt.close()

# --------------------------------------------------------------------------------------
# 3. DISTRIBUIÇÃO DAS CATEGORIAS NAS 4 DIMENSÕES
# --------------------------------------------------------------------------------------
# (a) Histogramas por variável, uma cor por classe
fig, eixos = plt.subplots(2, 2, figsize=(11, 8))
for eixo, col in zip(eixos.ravel(), COLUNAS):
    for classe in CLASSES:
        eixo.hist(df.loc[df["especie"] == classe, col], bins=12, alpha=0.6,
                  label=classe, color=CORES[classe])
    eixo.set_title(col)
    eixo.legend()
plt.tight_layout()
plt.savefig(f"{SAIDA}/03a_histogramas.png", dpi=120)
plt.close()

# (b) Matriz de dispersão: todos os pares de variáveis (mostra quais separam melhor as classes)
fig, eixos = plt.subplots(4, 4, figsize=(12, 12))
for i, ci in enumerate(COLUNAS):
    for j, cj in enumerate(COLUNAS):
        eixo = eixos[i, j]
        for classe in CLASSES:
            sub = df[df["especie"] == classe]
            if i == j:
                eixo.hist(sub[ci], bins=10, alpha=0.6, color=CORES[classe])
            else:
                eixo.scatter(sub[cj], sub[ci], s=8, color=CORES[classe], alpha=0.7)
        if i == 3: eixo.set_xlabel(cj)
        if j == 0: eixo.set_ylabel(ci)
plt.tight_layout()
plt.savefig(f"{SAIDA}/03b_matriz_dispersao.png", dpi=100)
plt.close()

# --------------------------------------------------------------------------------------
# 4. PCA (com OpenCV)
# --------------------------------------------------------------------------------------
# Padronizamos antes (média 0, desvio 1) para que nenhuma variável domine - ver slide "Scaler".
X_all = StandardScaler().fit_transform(df[COLUNAS].values).astype(np.float32)

# PCACompute2 devolve: média, autovetores (componentes) e autovalores (variância de cada componente)
media, autovetores, autovalores = cv2.PCACompute2(X_all, mean=None)
variancia_exp = autovalores.ravel() / autovalores.sum()

print("\n### 4. PCA")
for i, v in enumerate(variancia_exp, 1):
    print(f"  PC{i}: {v * 100:5.1f}% da variância (acumulado: {variancia_exp[:i].sum() * 100:5.1f}%)")
print("  Pesos (autovetores) - quanto cada variável contribui para cada PC:")
print(pd.DataFrame(autovetores, index=[f"PC{i}" for i in range(1, 5)], columns=COLUNAS).round(3))

# Projeção dos dados nos componentes principais: (X - média) * autovetores^T
X_pca = (X_all - media) @ autovetores.T

plt.figure(figsize=(7, 6))
for classe in CLASSES:
    m = (df["especie"] == classe).values
    plt.scatter(X_pca[m, 0], X_pca[m, 1], label=classe, color=CORES[classe], alpha=0.7)
plt.xlabel(f"PC1 ({variancia_exp[0] * 100:.1f}%)")
plt.ylabel(f"PC2 ({variancia_exp[1] * 100:.1f}%)")
plt.title("Iris projetado nos 2 primeiros componentes principais")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f"{SAIDA}/04_pca.png", dpi=120)
plt.close()

# --------------------------------------------------------------------------------------
# 5. DIVISÃO TREINO (80%) / TESTE (20%)
# --------------------------------------------------------------------------------------
X = df[COLUNAS].values.astype(np.float32)
y = df["especie"].map(MAPA_CLASSES).values.astype(np.int32)

# stratify=y mantém a mesma proporção de cada classe no treino e no teste
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

# Escalonamento: o scaler é ajustado SÓ no treino (fit_transform) e apenas aplicado no teste
# (transform). Assim nenhuma informação do teste "vaza" para o treino.
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
X_te_s = scaler.transform(X_te).astype(np.float32)

print(f"\n### 5. Divisão: {len(X_tr)} treino / {len(X_te)} teste")

# --------------------------------------------------------------------------------------
# 6. K-NN (OpenCV)
# --------------------------------------------------------------------------------------
def knn_prever(X_treino, y_treino, X_teste, k):
    """Treina (apenas memoriza os dados) e prediz com o K-NN do OpenCV (distância L2/Euclidiana)."""
    if hasattr(cv2, "ml"):
        knn = cv2.ml.KNearest_create()
        knn.train(X_treino, cv2.ml.ROW_SAMPLE, y_treino)   # ROW_SAMPLE: cada linha é uma amostra
        _, resultado, _, _ = knn.findNearest(X_teste, k)   # voto majoritário entre os k vizinhos
        return resultado.ravel().astype(int)
    # Plano B: se o seu OpenCV estiver sem o módulo "ml" (instalação quebrada/conflitante),
    # usa o K-NN do scikit-learn, que faz exatamente a mesma coisa (distância Euclidiana).
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=k).fit(X_treino, y_treino).predict(X_teste)

# 6a. Escolha do k por validação cruzada (5-fold) SOMENTE no conjunto de treino.
#     O teste fica guardado para a avaliação final (ideia #4 dos slides).
print("\n### 6a. Validação cruzada 5-fold para escolher k (apenas no treino)")
valores_k = [1, 3, 5, 7, 9, 11, 13, 15]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
medias, desvios = [], []
for k in valores_k:
    acuracias = []
    for idx_tr, idx_val in skf.split(X_tr_s, y_tr):
        # cada "fold" faz papel de validação uma vez
        pred = knn_prever(X_tr_s[idx_tr], y_tr[idx_tr], X_tr_s[idx_val], k)
        acuracias.append(accuracy_score(y_tr[idx_val], pred))
    medias.append(np.mean(acuracias))
    desvios.append(np.std(acuracias))
    print(f"  k={k:2d} -> acurácia média = {medias[-1]:.4f} (± {desvios[-1]:.4f})")

melhor_k = valores_k[int(np.argmax(medias))]      # em empate, argmax pega o menor k
print(f"  Melhor k = {melhor_k}")

plt.figure(figsize=(6, 4))
plt.errorbar(valores_k, medias, yerr=desvios, marker="o", capsize=3)
plt.axvline(melhor_k, color="red", linestyle="--", label=f"melhor k = {melhor_k}")
plt.xlabel("k")
plt.ylabel("Acurácia (validação cruzada)")
plt.title("Escolha de k - 5-fold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SAIDA}/06_escolha_k.png", dpi=120)
plt.close()

# 6b. Avaliação final no conjunto de TESTE com o melhor k
print(f"\n### 6b. Resultado no conjunto de teste (k={melhor_k}, 4 variáveis)")
y_pred = knn_prever(X_tr_s, y_tr, X_te_s, melhor_k)
print("Acurácia:", round(accuracy_score(y_te, y_pred), 4))
print("\nRelatório de classificação:")
print(classification_report(y_te, y_pred, target_names=CLASSES, digits=3))
print("Matriz de confusão (linhas = real, colunas = predito):")
print(pd.DataFrame(confusion_matrix(y_te, y_pred), index=CLASSES, columns=CLASSES))

# 6c. Extra: o K-NN funciona só com os 2 primeiros componentes do PCA?
#     (o PCA é ajustado no treino e aplicado ao teste)
media_tr, comp_tr, _ = cv2.PCACompute2(X_tr_s, mean=None, maxComponents=2)
X_tr_p = ((X_tr_s - media_tr) @ comp_tr.T).astype(np.float32)
X_te_p = ((X_te_s - media_tr) @ comp_tr.T).astype(np.float32)
y_pred_pca = knn_prever(X_tr_p, y_tr, X_te_p, melhor_k)
print(f"\n### 6c. K-NN com apenas 2 componentes principais (k={melhor_k})")
print("Acurácia:", round(accuracy_score(y_te, y_pred_pca), 4))

# 6d. Extra: fronteiras de decisão do K-NN no plano PCA (como na demo de Stanford dos slides)
xx, yy = np.meshgrid(
    np.linspace(X_tr_p[:, 0].min() - 1, X_tr_p[:, 0].max() + 1, 300),
    np.linspace(X_tr_p[:, 1].min() - 1, X_tr_p[:, 1].max() + 1, 300))
grade = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
zz = knn_prever(X_tr_p, y_tr, grade, melhor_k).reshape(xx.shape)

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5, 2.5], colors=["#cce5ff", "#ffe5cc", "#d5f5d5"])
for i, classe in enumerate(CLASSES):
    plt.scatter(X_tr_p[y_tr == i, 0], X_tr_p[y_tr == i, 1], color=CORES[classe],
                edgecolor="k", label=f"{classe} (treino)")
    plt.scatter(X_te_p[y_te == i, 0], X_te_p[y_te == i, 1], color=CORES[classe],
                marker="*", s=150, edgecolor="k")
plt.title(f"K-NN (k={melhor_k}) no espaço PCA  (estrelas = teste)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.savefig(f"{SAIDA}/06_fronteiras_knn_pca.png", dpi=120)
plt.close()

print(f"\nPronto! Gráficos salvos em '{SAIDA}/'.")