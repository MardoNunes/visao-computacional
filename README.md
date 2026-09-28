# Calibração de Câmera (Zhang / OpenCV)

Trabalho de Visão Computacional (UFPR) que estima os parâmetros **intrínsecos**
(foco, ponto principal) e de **distorção** de uma câmera a partir de fotos de um
tabuleiro de xadrez, usando o método clássico de Zhang (2000) e a OpenCV.

**Resultado:** erro médio de reprojeção de **1,66 px** com 4 imagens (196
correspondências).


Lê `img/*.jpg` e grava as imagens de resultado em `calibracao_resultados/`.

## Pipeline

1. **Padrão** — pontos 3D dos cantos internos em grade 7×7, quadrado de 40 mm, plano `Z=0`.
2. **Detecção** — `findChessboardCornersSB` (fallback: `findChessboardCorners`) + `cornerSubPix`.
3. **Calibração** — `calibrateCamera` minimiza o erro de reprojeção (Levenberg–Marquardt).
4. **Verificação** — `undistort` remove a distorção e a pose de uma imagem é usada para reprojetar os cantos 3D.

## Resultados gerados

| Arquivo | Conteúdo |
|---|---|
| `cantos_*.jpg` | Cantos detectados, coloridos por coluna (checagem da ordem da grade) |
| `comparacao_distorcao.jpg` | Imagem original (esquerda) vs. corrigida (direita) |
| `imagem_corrigida.jpg` | Imagem sem distorção |
| `experimento_projecao.jpg` | Pontos observados (verde) vs. projetados (vermelho) |

## Parâmetros estimados

```
K = [[3379.70,    0.00,  883.74],      k = (k1, k2, p1, p2, k3)
     [   0.00, 3178.99, 1573.11],           = (-0.3166, 2.9547, -0.0474, -0.0103, -7.3389)
     [   0.00,    0.00,    1.00]]
```

RMS global: 1,92 px — por imagem: `1.jpg` 0,94 · `5.jpg` 1,68 · `6.jpg` 1,61 · `8.jpg` 2,41.


## Estrutura

```
src/main.py             pipeline completo
img/                    fotos do tabuleiro (entradas)
calibracao_resultados/  saídas (geradas pelo script)

```
## Para Testar
```bash

#1
python3 -m venv venv

#2
source venv/bin/activate

#3 
pip install opencv-python

#4

python3 src/main.py
```
