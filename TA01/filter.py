import cv2
import numpy as np

# Lê uma imagem do disco (formato padrão BGR)
imagem = cv2.imread("ufpr.jpg")

# Converte a imagem para escala de cinza
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('Teste', cv2.WINDOW_NORMAL) # cria uma janela, sem ela nao tem como mudar o size da janela
cv2.resizeWindow('Teste', 800, 600) # redimensiona o tamanho da janela


# convertendo imagem para RGB
imagemRgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

# convertendo para cinza
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
tamanho = (512, 512)
imagemCinza = cv2.resize(imagemCinza, tamanho, interpolation=cv2.INTER_LINEAR) # redimensiona o tamanho da imagem

# kernel horizontal?
kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

#aplicando filtro
response = cv2.filter2D(
    imagemCinza,
    cv2.CV_32F,
    kernel
)

# isso daqui instancia a imagem
cv2.imshow('Teste', response)
cv2.waitKey(0)  # sem isso, ela abre e fecha rapidamente
cv2.destroyAllWindows()
