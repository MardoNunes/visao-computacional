import cv2
import numpy as np

# Lê uma imagem do disco (formato padrão BGR)
imagem = cv2.imread("ufpr.jpg")

# Converte a imagem para escala de cinza
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# cv2.namedWindow('TesteR', cv2.WINDOW_NORMAL) # cria uma janela, sem ela nao tem como mudar o size da janela
# cv2.resizeWindow('TesteR', 800, 600) # redimensiona o tamanho da janela


# convertendo imagem para RGB
imagemRgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

# convertendo para cinza
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
tamanho = (512, 512)
imagemCinza = cv2.resize(imagemCinza, tamanho, interpolation=cv2.INTER_LINEAR) # redimensiona o tamanho da imagem

# kernel horizontal
kernelH = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
], dtype=np.float32)


# kernel Vertical
kernelV = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# kernel diagonal 45
kernelD45 = np.array([
    [-2, -1, 0],
    [-1, 0, 1],
    [0, 1, 2]
], dtype=np.float32)

#kernel diagonal 135
kernelD135 = np.array([
    [-2, -1, 0],
    [-1, 0, 1],
    [0, 1, 2]
], dtype=np.float32)

#aplicando filtro
responseV = cv2.filter2D(
    imagemCinza,
    cv2.CV_32F,
    kernelV
)

# circular
kernelC = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))

responseH = cv2.filter2D(
    imagemCinza,
    cv2.CV_32F,
    kernelH
)


responseD45 = cv2.filter2D(
    imagemCinza,
    cv2.CV_32F,
    kernelD45
)

responseD135 = cv2.filter2D(
    imagemCinza,
    cv2.CV_32F,
    kernelD135
)

responseC = cv2.filter2D(
    imagemCinza,
    cv2.CV_32F,
    kernelC
)


# isso daqui instancia a imagem
cv2.imshow('Teste Vertical', responseV)
cv2.imshow('Teste Horizontal', responseH)
cv2.imshow('Teste diagonal 45 graus', responseD45)
# cv2.imshow('Teste diagonal 135 graus', responseD135)
cv2.imshow('Teste circular', responseC)
cv2.waitKey(0)  # sem isso, ela abre e fecha rapidamente
cv2.destroyAllWindows()
