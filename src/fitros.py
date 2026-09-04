import cv2
import os

# Define as pastas de entrada e saída
pasta_origem = './img'
pasta_destino = './img_proc'

# Cria a pasta de destino se ela não existir
if not os.path.exists(pasta_destino):
    os.makedirs(pasta_destino)

# Lista para guardar na memória, caso queira continuar o código na mesma execução
imagens_prontas = []

for nome_arquivo in os.listdir(pasta_origem):
    
    caminho_completo = os.path.join(pasta_origem, nome_arquivo)
    imagem = cv2.imread(caminho_completo, cv2.IMREAD_GRAYSCALE)
    
    if imagem is not None:
        # Redimensiona para 512x512
        imagem_padronizada = cv2.resize(imagem, (512, 512))
        
        # Cria o caminho onde a nova imagem será salva
        caminho_saida = os.path.join(pasta_destino, nome_arquivo)
        
        # Salva a imagem processada na nova pasta
        cv2.imwrite(caminho_saida, imagem_padronizada)
        
        # Adiciona na lista da memória
        imagens_prontas.append(imagem_padronizada)
        
        print(f"Imagem {nome_arquivo} padronizada e salva em {caminho_saida}!")

print(f"\nConcluído! {len(imagens_prontas)} imagens prontas na pasta {pasta_destino}.")