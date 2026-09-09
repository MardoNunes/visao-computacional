import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# PRE PROCESSAMENTO 

diretorio_script = os.path.dirname(os.path.abspath(__file__))           #caminho atual
pasta_origem = os.path.join(diretorio_script, '../img')                 #pasta com as imagens originais
pasta_destino = os.path.join(diretorio_script, '../img_proc')           #pasta com as imagens normalizadas, escala de cinza e 512x512
pasta_resultados = os.path.join(diretorio_script, '../img_resultados')  # Nova pasta para saída

#se nao existir cria as pastas
if not os.path.exists(pasta_destino): 
    os.makedirs(pasta_destino)
if not os.path.exists(pasta_resultados):
    os.makedirs(pasta_resultados)

#listas dos arquivos mantidos na memoria e dos nomes
imagens_prontas = []
nomes_arquivos = [] # Precisamos guardar a ordem dos nomes para salvar no final

#percorre os arquivos da entrada
for nome_arquivo in os.listdir(pasta_origem):
    caminho_completo = os.path.join(pasta_origem, nome_arquivo)
    imagem = cv2.imread(caminho_completo, cv2.IMREAD_GRAYSCALE) #abre eles em escala de cinza
    
    if imagem is not None: #verifica se a imagem foi carregada
        imagem_padronizada = cv2.resize(imagem, (512, 512))         #redimensiona
        caminho_saida = os.path.join(pasta_destino, nome_arquivo)   #caminho da saida
        cv2.imwrite(caminho_saida, imagem_padronizada)              #escreve a imagem
        
        imagens_prontas.append(imagem_padronizada)                  #append na lista mantida na memoria
        nomes_arquivos.append(nome_arquivo)                         #append do nome associado a imagem no indice

print(f"Pré-processamento: {len(imagens_prontas)} imagens carregadas na memória.")


if len(imagens_prontas) == 0:
    print("ERRO: Nenhuma imagem processada.")
    exit()


# DEFINE DOS KERNELS
kernel_h = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]])
kernel_v = np.array([[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]])
kernel_45 = np.array([[ 0,  1,  2], [-1,  0,  1], [-2, -1,  0]])
kernel_135 = np.array([[-2, -1,  0], [-1,  0,  1], [ 0,  1,  2]])
kernel_c = np.array([[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]])

def aplicar_filtros(imagem):
    return {
        'h': np.abs(cv2.filter2D(imagem, cv2.CV_64F, kernel_h)),
        'v': np.abs(cv2.filter2D(imagem, cv2.CV_64F, kernel_v)),
        '45': np.abs(cv2.filter2D(imagem, cv2.CV_64F, kernel_45)),
        '135': np.abs(cv2.filter2D(imagem, cv2.CV_64F, kernel_135)),
        'c': np.abs(cv2.filter2D(imagem, cv2.CV_64F, kernel_c))
    }

# EXTRACAO DE CARACTERISTICAS 

#vetores de caracteristicas das imagens
lista_com_todos_os_vetores = []

print("Iniciando extracao de caracteristicas...")
for idx, escala_1 in enumerate(imagens_prontas):
    
    #Passa a gaussiana nas imagens e cria as escalas reduzidas 
    #Ja temos em 512x512, reduz pela metade cada dimensao e cria em 256x256 e 128x128
    borrada_1 = cv2.GaussianBlur(escala_1, (5, 5), 0) 
    escala_2 = cv2.resize(borrada_1, (256, 256)) 
    borrada_2 = cv2.GaussianBlur(escala_2, (5, 5), 0)
    escala_3 = cv2.resize(borrada_2, (128, 128))
    
    # Aplica filtros nas 3 escalas
    filtros_e1 = aplicar_filtros(escala_1)
    filtros_e2 = aplicar_filtros(escala_2)
    filtros_e3 = aplicar_filtros(escala_3)
    
    # Laço de recortes 
    for y in range(0, 512, 32):
        for x in range(0, 512, 32):

            # PARA CADA ESCALA TEM UMA JANELA DIFERENTE
            # Aplica os kernels e as outras metricas na janela atual da imagem
            # np.mean(j1) tira a media da janela
            # np.std(j1) tira o desvio padrao
            # np.max(j1) - np.min(j1) tira a amplitude dos valores
            # os demais sao os kernels, horizontal, vertical, circular, 45 e 135graus
            
            # Escala 1 (32x32)
            l1=32
            j1 = escala_1[y:y+l1, x:x+l1]
            v1 = [np.mean(j1), np.std(j1), np.max(j1) - np.min(j1),
                  np.mean(filtros_e1['h'][y:y+l1, x:x+l1]), np.mean(filtros_e1['v'][y:y+l1, x:x+l1]),
                  np.mean(filtros_e1['45'][y:y+l1, x:x+l1]), np.mean(filtros_e1['135'][y:y+l1, x:x+l1]),
                  np.mean(filtros_e1['c'][y:y+l1, x:x+l1])]
            
            # Escala 2 (16x16)
            l2=16
            y2, x2 = y // 2, x // 2
            j2 = escala_2[y2:y2+l2, x2:x2+l2]
            v2 = [np.mean(j2), np.std(j2), np.max(j2) - np.min(j2),
                  np.mean(filtros_e2['h'][y2:y2+l2, x2:x2+l2]), np.mean(filtros_e2['v'][y2:y2+l2, x2:x2+l2]),
                  np.mean(filtros_e2['45'][y2:y2+l2, x2:x2+l2]), np.mean(filtros_e2['135'][y2:y2+l2, x2:x2+l2]),
                  np.mean(filtros_e2['c'][y2:y2+l2, x2:x2+l2])]
            
            # Escala 3 (8x8)
            l3=8
            y3, x3 = y // 4, x // 4
            j3 = escala_3[y3:y3+l3, x3:x3+l3]
            v3 = [np.mean(j3), np.std(j3), np.max(j3) - np.min(j3),
                  np.mean(filtros_e3['h'][y3:y3+l3, x3:x3+l3]), np.mean(filtros_e3['v'][y3:y3+l3, x3:x3+l3]),
                  np.mean(filtros_e3['45'][y3:y3+l3, x3:x3+l3]), np.mean(filtros_e3['135'][y3:y3+l3, x3:x3+l3]),
                  np.mean(filtros_e3['c'][y3:y3+l3, x3:x3+l3])]

            # Agrupa os valores no vetor
            lista_com_todos_os_vetores.append(v1 + v2 + v3)

# AGRUPAMENTO K-MEANS 
print(f"Treinando K-Means com {len(lista_com_todos_os_vetores)} vetores...")
matriz_vetores = np.array(lista_com_todos_os_vetores) #Como o kmeans recebe uma matriz, colocamos os vetores numa matriz

# Aplica normalização global (média 0, desvio padrão 1)
padronizador = StandardScaler()
matriz_normalizada = padronizador.fit_transform(matriz_vetores)

#Parametros do algoritmo de clustering, que vai fazer o agrupamento dos vetores de caracteristicas
kmeans = KMeans(n_clusters=6, random_state=42) 
kmeans.fit(matriz_normalizada)
grupos_globais = kmeans.labels_


# Grupo de cores com varias cores para testar varios numeros de K do kmeans
cores_dos_grupos = [
    (0, 0, 255),      # 0: Vermelho
    (0, 255, 0),      # 1: Verde
    (255, 0, 0),      # 2: Azul
    (0, 255, 255),    # 3: Amarelo
    (255, 0, 255),    # 4: Magenta
    (255, 255, 0),    # 5: Ciano
    (0, 165, 255),    # 6: Laranja
    (255, 255, 255),  # 7: Branco
    (128, 0, 128),    # 8: Púrpura Escuro
    (0, 100, 0),      # 9: Verde Escuro
    (147, 20, 255),   # 10: Rosa Forte (Deep Pink)
    (0, 215, 255),    # 11: Dourado (Gold)
    (30, 105, 210),   # 12: Marrom (Chocolate)
    (211, 0, 148),    # 13: Violeta (Dark Violet)
    (113, 204, 46),   # 14: Verde Folha (Sea Green)
    (130, 0, 75)      # 15: Índigo Escuro
]

indice_global = 0
print("Pintando imagens...")

# Itera novamente sobre as imagens originais para desenhar
for idx, escala_1 in enumerate(imagens_prontas):
    imagem_colorida = cv2.cvtColor(escala_1, cv2.COLOR_GRAY2BGR)
    camada_pintura = imagem_colorida.copy()
    
    for y in range(0, 512, 32):
        for x in range(0, 512, 32):
            cor = cores_dos_grupos[grupos_globais[indice_global]]
            cv2.rectangle(camada_pintura, (x, y), (x+32, y+32), cor, -1)
            indice_global += 1
            
    # Aplica transparencia
    resultado_final = cv2.addWeighted(camada_pintura, 0.4, imagem_colorida, 0.6, 0)
    
    # Salva no disco
    nome_arquivo_saida = nomes_arquivos[idx]
    caminho_final = os.path.join(pasta_resultados, f"kmeans_{nome_arquivo_saida}")
    cv2.imwrite(caminho_final, resultado_final)

print("Processamento em lote concluído com sucesso.")