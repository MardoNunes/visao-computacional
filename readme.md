# Pipeline de Classificação e Segmentação de Texturas em Imagens

Este repositório contém a documentação e implementação do script `classificador.py`. O sistema tem como objetivo extrair características estruturais e de textura de um lote de imagens.

## Pré-requisitos e Dependências

Para executar o pipeline, o ambiente Python deve conter as seguintes bibliotecas:
*   `opencv-python` (cv2): Manipulação de matrizes de imagem e convoluções.
*   `numpy`: Operações matemáticas e manipulação de vetores de alta performance.
*   `scikit-learn`: Execução do algoritmo de agrupamento K-Means e normalização estatística dos dados.
* Criando um ambiente python:
    ```bash
    python3 -m venv venv
    ```
* Acessando ao ambiente:
  ```bash
    source venv/bin/activate
  ```
* Instalando as dependências:
    ```bash
    pip install opencv-python && scikit-learn
    ```
Agora basta executar o classificador.py com as imagens presentes em `/img`.
* Para sair do ambiente virtual
    ```bash
    deactivate
    ```
## Estrutura de Diretórios

O script foi arquitetado para rodar em lote e exige uma hierarquia de pastas específica, baseada no caminho de execução do arquivo `classificador.py`:
*   `../img/`: Diretório de origem. Aqui você deve depositar as imagens a serem analisadas nesta pasta antes da execução.
*   `../img_proc/`: Diretório gerado automaticamente. Armazena as cópias das imagens padronizadas (escala de cinza e 512x512 pixels) utilizadas pelo pipeline.
*   `../img_resultados/`: Diretório gerado automaticamente. Recebe o output final do sistema: as imagens originais sobrepostas pela malha de cores que representam os clusters segmentados.

## Detalhes Técnicos da Implementação

O fluxo de dados no `classificador.py` é dividido em cinco etapas arquiteturais estritas:

### 1. Pré-processamento
O carregamento das imagens descarta componentes de cor, forçando a leitura nativa em escala de cinza (`cv2.IMREAD_GRAYSCALE`). Para garantir consistência na extração de janelas, todas as matrizes de entrada são redimensionadas para a dimensão padrão de 512x512 pixels e indexadas em memória.

### 2. Definição de Kernels Direcionais
A detecção de textura primária é baseada em filtros espaciais. O sistema inicializa matrizes NumPy (`CV_64F`) para varrer cinco padrões de borda usando `cv2.filter2D`:
*   Filtros ortogonais: Horizontal (`kernel_h`) e Vertical (`kernel_v`).
*   Filtros diagonais: 45° (`kernel_45`) e 135° (`kernel_135`).
*   Filtro isotrópico/Laplaciano: Circular (`kernel_c`), que detecta pontos e quebras abruptas de frequência em todas as direções.

### 3. Extração de Características (Pirâmide Multi-Escala)
O algoritmo recorta a imagem original em uma grade de janelas estáticas de 32x32 pixels, iterando ao longo dos eixos X e Y. Para capturar o contexto visual, uma abordagem de pirâmide de imagens é empregada:
*   **Escala 1 (512x512):** Processa o detalhe fino usando janelas de 32x32 pixels.
*   **Escala 2 (256x256):** A imagem original sofre um borramento (`cv2.GaussianBlur`) e é reduzida à metade. A janela analisada cai para 16x16 pixels.
*   **Escala 3 (128x128):** Novo borramento e redução, processando a macro-textura em janelas de 8x8 pixels.

Para cada uma das três janelas concêntricas, o sistema extrai oito métricas: Média de intensidade, Desvio Padrão, Amplitude de contraste (Máximo - Mínimo) e a média absoluta de resposta dos cinco kernels. A concatenação das três escalas resulta em um vetor rígido de 24 dimensões que descreve matematicamente aquela região de 32x32 da imagem.

### 4. Normalização e Aprendizado Não Supervisionado (Global)
A validação de um lote de imagens exige que a classificação seja consistente entre diferentes fotos (ou seja, a cor "Vermelha" precisa apontar para a mesma textura na Foto 1 e na Foto 10). 
*   **Aglutinação:** Todos os vetores extraídos de todas as imagens em memória são reunidos em uma única matriz NumPy global.
*   **Padronização (`StandardScaler`):** Etapa crítica para o funcionamento do K-Means. A matriz é normalizada pelo scikit-learn para que todas as 24 variáveis possuam média 0 e variância 1. Isso equaliza pesos estatísticos e impede que variações severas de iluminação ofusquem a detecção direcional dos kernels.
*   **K-Means:** O agrupamento processa a matriz global utilizando a Distância Euclidiana, dividindo o espaço em 5 clusters definidos (parâmetro expansível). A semente de aleatoriedade (`random_state=42`) é travada para garantir reprodutibilidade matemática entre execuções.

### 5. Visualização e Transparência
As predições lineares devolvidas pelo `kmeans.labels_` são remapeadas espacialmente de volta para a grade bidimensional das imagens originais. 
*   Uma paleta interna suporta mapeamento em RGB para até 16 grupos distintos.
*   O algoritmo gera uma camada temporária de pintura (`camada_pintura`) onde preenche retângulos sólidos nas posições mapeadas.
*   Por fim, emprega a função `cv2.addWeighted` com pesos de 0.4 para a máscara de cor e 0.6 para a imagem original em escala de cinza. Isso permite que a fotografia permaneça visível através das demarcações de textura geradas pela máquina antes de ser salva em disco.
