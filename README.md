#### Transformação de Perspectiva - TA03

Modelo interativo de correção de perspectiva: o usuário clica em 4 pontos na imagem e o programa calcula a homografia `3x3` (`cv.getPerspectiveTransform`) e aplica `cv.warpPerspective` para gerar uma vista frontal.

##### Para instalar dependência
- crie um ambiente vitual e ative-o
```bash
python3 -m venv venv
source venv/bin/activate
```
- Com o ambiente ativado instale as dependências
```bash
pip install opencv-python
pip install matplotlib
pip install numpy
```

##### Como executar
- com o ambiente virutal ativado e dependências instaladas, rode:
```bash
python3 main.py <caminho da imagem>
python3 main.py entrada/sudoku.jpg --largura 300 --altura 300
python3 main.py entrada/sudoku.jpg --sem-mostrar   # não exibe o matplotlib
```

Ao executar, uma janela `Selecione os 4 pontos` abre:
- clique nos 4 cantos do objeto (qualquer ordem);
- `r` reinicia a seleção, `q` ou ESC cancela;
- após o 4º clique o programa fecha a janela, calcula e exibe `Input x Output`.

No diretório `/entrada` existe uma imagem de exemplo. Todas as imagens reconstruidas com outra perspectivas são salvas do diretório `/saida`.

##### Estrutura de pastas
```
visao-computacional/
├── main.py                 # pipeline completo (seleção + homografia + warp)
├── entrada/
│   └── sudoku.jpg          # imagem de entrada de exemplo
├── saida/
│   ├── sudoku_perspectiva.jpg
│   └── sudoku_perspectiva_1..4.jpg  # saídas retificadas (300x300 por padrão)
├── comparativo.png         # screenshot Input x Output (matplotlib)
├── selecionaPontos.png     # screenshot da janela de seleção dos 4 pontos
├── relatorio/
│   └── relatorio.pdf       # artigo compilado
└── README.md
```

##### Como funciona o algoritmo

1. **Leitura** — `cv.imread(caminho)` carrega a imagem em BGR.
2. **Seleção interativa (`selecionar_pontos`)** — janela OpenCV com `setMouseCallback`: cada `EVENT_LBUTTONDOWN` salva `(x, y)`, desenha círculo vermelho + número verde. Retorna `np.float32 (4,2)` ou `None` se cancelado.
3. **Ordenação (`ordenar_pontos`)** — garante ordem `[topo-esq, topo-dir, base-esq, base-dir]` independente da ordem do clique, usando `soma = x+y` (menor = topo-esq, maior = base-dir) e `diff = y-x` (menor = topo-dir, maior = base-esq).
4. **Homografia** — destino canônico `pts2 = [(0,0), (w,0), (0,h), (w,h)]`, depois `M = cv.getPerspectiveTransform(pts1, pts2)`. `M` é a matriz `3x3` que resolve o sistema DLT das 4 correspondências (8 graus de liberdade):
   ```
   [x' y' w']^T = M · [x y 1]^T,  X' = x'/w', Y' = y'/w'
   ```
5. **Deformação (`warpPerspective`)** — `dst = cv.warpPerspective(img, M, (w, h))` aplica `M⁻¹` com interpolação bilinear para cada pixel de destino.
6. **Exibição e salvamento** — `matplotlib` mostra `Input x Output` lado a lado; `proximo_destino()` gera `saida/<stem>_perspectiva_<n>.jpg` sem sobrescrever e `cv.imwrite` salva.

