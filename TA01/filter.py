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

kernelH5 = np.array([
    [1, 2, 1, 2, 1],
    [0, 0, 0, 0, 0],
    [-1, -2, -1, -2, -1]
], dtype=np.float32)

kernelH11 = np.array([
    [1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0],
    [-1, -2, -2, -1, -1, -1, -1, -1, -2, -2, -1]
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

kernelC = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, -1, 0]
], dtype=np.float32)

#aplicando filtro
responseV = cv2.filter2D(
    imagemCinza,
    cv2.CV_32F,
    kernelV
)

# circular

responseH = cv2.filter2D(
    imagemCinza,
    cv2.CV_32F,
    kernelH
)


responseH5 = cv2.filter2D(
    imagemCinza,
    cv2.CV_32F,
    kernelH5
)

responseH11 = cv2.filter2D(
    imagemCinza,
    cv2.CV_32F,
    kernelH11
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

#normalização das imagens
# pode printar as imagens direto com os reponses acima, porém fica feio de ver
# aqui normalizamos 
magnitudeV = np.abs(responseV)
visV = cv2.normalize(
    magnitudeV,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

magnitudeH = np.abs(responseH)
visH = cv2.normalize(
    magnitudeV,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

magnitudeD45 = np.abs(responseD45)
visD45 = cv2.normalize(
    magnitudeV,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

magnitudeD135 = np.abs(responseD135)
visD135 = cv2.normalize(
    magnitudeV,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

magnitudeC = np.abs(responseC)
visC = cv2.normalize(
    magnitudeV,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)


# printar imagens normalizadas, sem valores negativos de pixel
# o filtrosdo cv2 pode retorna vlores do tipo: -500 à 500
# normalizar deixa os pixel no range: 0 à 255 apenas
# cv2.imshow("Teste Vertical2", visV)
# cv2.imshow("Teste Horizontal", visH)
# cv2.imshow("Teste Diagonal 45", visD45)
# cv2.imshow("Teste Diagonal 135", visD135)
# cv2.imshow("Teste Circular", visC)


# printar imagens sem normalização
cv2.imshow('Teste Vertical', responseH)
# cv2.imshow('Teste Vertical escala 5x5', responseH5)
# cv2.imshow('Teste Vertical escala 11x11', responseH11)
cv2.imshow('Teste Horizontal', responseV)
cv2.imshow('Teste diagonal 45 graus', responseD45)
cv2.imshow('Teste diagonal 135 graus', responseD135)
cv2.imshow('Teste circular', responseC)
cv2.waitKey(0)  # sem isso, ela abre e fecha rapidamente
cv2.destroyAllWindows()
