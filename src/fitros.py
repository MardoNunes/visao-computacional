import cv2
import os

# Descobre o caminho exato de onde este script (fitros.py) está salvo
diretorio_script = os.path.dirname(os.path.abspath(__file__))

# A pasta de origem 'img' está um nível atrás do script (..)
pasta_origem = os.path.join(diretorio_script, '../img')

# A pasta de destino 'img_proc' está junto com o script
pasta_destino = os.path.join(diretorio_script, 'img_proc')

# Cria a pasta de destino se ela não existir
if not os.path.exists(pasta_destino):
    os.makedirs(pasta_destino)

# Lista para guardar na memória, caso a gente continue o código aqui mesmo
imagens_prontas = []

# Lista todos os arquivos que estão lá na pasta 'img'
for nome_arquivo in os.listdir(pasta_origem):
    
    caminho_completo = os.path.join(pasta_origem, nome_arquivo)
    
    # Lê a imagem forçando escala de cinza
    imagem = cv2.imread(caminho_completo, cv2.IMREAD_GRAYSCALE)
    
    # Se o arquivo for mesmo uma imagem, ele processa
    if imagem is not None:
        # Redimensiona para 512x512
        imagem_padronizada = cv2.resize(imagem, (512, 512))
        
        # Salva a imagem processada dentro de src/img_proc/
        caminho_saida = os.path.join(pasta_destino, nome_arquivo)
        cv2.imwrite(caminho_saida, imagem_padronizada)
        
        imagens_prontas.append(imagem_padronizada)
        print(f"Sucesso! {nome_arquivo} processada e salva em: src/img_proc/")

print(f"\nConcluído! {len(imagens_prontas)} imagem(ns) prontas.")