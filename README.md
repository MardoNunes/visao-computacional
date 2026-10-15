# Classificador K-NN — Dataset Iris

Classificação das 3 espécies da flor Iris (*setosa*, *versicolor*, *virginica*) com **K-NN do OpenCV** (`cv2.ml.KNearest`), incluindo análise exploratória, PCA e escolha de `k` por validação cruzada. Todo o experimento está em `iris_knn.py` e os gráficos saem em `saida_iris/`.

## Estrutura

| Arquivo / pasta | O que é |
|---|---|
| `iris_knn.py` | Script principal: estatísticas, gráficos, PCA, treino e avaliação do K-NN |
| `main.tex` | Relatório em LaTeX (mesma estrutura do template SBC da atividade anterior) |
| `saida_iris/` | Gráficos gerados pelo script (6 imagens `.png`) |
| `irisDataSet/` | Cópia dos arquivos originais da base UCI (referência; o script usa `load_iris` do scikit-learn, sem download) |

## Requisitos

- Python 3.10+
- `pip install numpy pandas matplotlib opencv-python scikit-learn`

## Como executar

```bash
python3 iris_knn.py
```

Saída esperada no terminal: estatísticas por classe, correlações, variância do PCA, acurácia por `k` no 5-fold, relatório de classificação e matriz de confusão. Os gráficos são salvos em `saida_iris/`.

Reprodutibilidade: semente fixa `SEED = 42` e divisão estratificada 80% treino (120) / 20% teste (30, sendo 10 de cada classe).

## Como funciona (resumo)

1. **Carga:** `iris.data` dataset abaixado de: https://archive.ics.uci.edu/dataset/53/iris 
2. **Padronização:** `StandardScaler` ajustado só no treino, aplicado no teste.
3. **PCA:** `cv2.PCACompute2` nos dados padronizados (PC1 73,0% + PC2 22,9% ≈ 95,8% da variância).
4. **K-NN:** distância Euclidiana + voto majoritário entre os `k` vizinhos.
5. **Escolha de `k`:** 5-fold só no treino, testando `k ∈ {1, 3, 5, 7, 9, 11, 13, 15}` → melhor `k = 3` (96,7% média no CV).
6. **Avaliação final:** teste guardado até o fim → **acurácia 93,3%** (28/30). Erros: 2 *virginica* preditas como *versicolor*. Repetindo com só 2 PCs, mesma acurácia (93,3%).

## Gráficos em `saida_iris/` (leitura direta)

| Imagem | O que é | Como ler |
|---|---|---|
| `02_comprimento_x_largura.png` | Dispersão comprimento × largura da sépala (esq.) e da pétala (dir.), por classe, com `r` e reta de tendência | Na pétala, a *setosa* (azul) está isolada embaixo à esquerda; *versicolor* e *virginica* se encostam no centro. Na sépala as 3 se misturam |
| `03a_histogramas.png` | 4 histogramas (um por medida), 3 cores sobrepostas | Pétala: azul isolado à esquerda (*setosa* sempre acerta); laranja e verde se sobrepõem no centro. Sépala: sobreposição maior |
| `03b_matriz_dispersao.png` | Grade 4×4 com todos os pares de variáveis; diagonal = histograma, resto = dispersão | Qualquer par com medida da pétala isola a nuvem azul; pares só de sépala misturam tudo |
| `04_pca.png` | Projeção das 150 flores em PC1 × PC2 | *Setosa* isolada à esquerda; *versicolor* e *virginica* vizinhas à direita com pequena mistura na fronteira |
| `06_escolha_k.png` | Acurácia média do 5-fold por `k`, com barra de desvio entre folds | Pico em `k = 3` (linha tracejada); `k` grande demais derruba a acurácia |
| `06_fronteiras_knn_pca.png` | Fronteiras de decisão do K-NN (`k = 3`) no plano PCA | Fundo colorido = classe prevista; círculos = treino, estrelas = teste. Estrela fora da cor da sua classe = erro (2 casos na fronteira laranja/verde) |

