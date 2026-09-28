"""
Calibração de Câmera - Método clássico (Zhang / OpenCV)
Disciplina: Visão Computacional e Percepção - UFPR (Prof. Eduardo Todt)
By: Davi Lazzarin e Mardoqueu Nunes

Fluxo:
  1. Detecta os cantos do tabuleiro em várias fotos
  2. Calibra a câmera (matriz intrínseca K, coeficientes de distorção)
  3. Remove a distorção de uma imagem de teste
  4. Faz o experimento de projeção 3D -> 2D e calcula o erro de reprojeção

"""

import cv2
import numpy as np
import glob
import os
import json

# ============================================================
# CONFIGURAÇÃO — AJUSTE AQUI
# ============================================================

# Pasta com as fotos do tabuleiro (aceita jpg, jpeg, png)
IMAGES_PATH = "img/*.jpg"

# Número de CANTOS INTERNOS do tabuleiro (não é o número de quadrados!)
# Tabuleiro de xadrez padrão: 8x8 quadrados (a-h, 1-8) -> 7x7 cantos internos.
# A moldura de madeira NÃO conta; só a área quadriculada.
CHESSBOARD_SIZE = (7, 7)   # (colunas, linhas) de cantos internos

# Tamanho de cada quadrado do tabuleiro, em milímetros
SQUARE_SIZE_MM = 40.0      # 4cm = 40mm

# Pasta de saída para os resultados
OUTPUT_DIR = "calibracao_resultados"

# ============================================================
# 1. PREPARAR PONTOS 3D DO PADRÃO (mundo real, Z=0, plano do tabuleiro)
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pontos do tipo (0,0,0), (1,0,0), (2,0,0) ... (cols-1, rows-1, 0), escalados
# pelo tamanho real do quadrado.
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

objpoints = []  # pontos 3D no mundo real, para cada imagem
imgpoints = []  # pontos 2D correspondentes na imagem, para cada imagem
nomes_usados = []  # nomes das imagens em que os cantos foram encontrados

# Critério de refinamento de sub-pixel
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def desenhar_padrao_estilo_tutorial(img, corners, chessboard_size):
    """
    Recria a visualização do tutorial oficial do OpenCV (findChessboardCorners):
    cada COLUNA do tabuleiro recebe uma cor do arco-íris, os cantos dessa coluna
    são ligados por uma linha, e cada canto vira um círculo colorido.

    Serve como checagem visual: se as linhas seguirem a grade de forma organizada,
    a ordem/orientação dos cantos foi detectada corretamente. Desenhamos manualmente
    (em vez de usar cv2.drawChessboardCorners) porque em fotos de alta resolução os
    marcadores padrão do OpenCV ficam pequenos demais para enxergar.
    """
    n_cols, n_rows = chessboard_size
    pts = np.asarray(corners, dtype=np.float64).reshape(n_rows, n_cols, 2)

    vis = img.copy()
    espessura_linha = max(2, img.shape[1] // 400)
    raio_ponto = max(5, img.shape[1] // 250)

    for col in range(n_cols):
        # Matiz (hue) do OpenCV vai de 0 a 179 -> cor diferente para cada coluna
        matiz = int(179 * col / max(1, n_cols - 1))
        cor_bgr = cv2.cvtColor(np.uint8([[[matiz, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()

        for row in range(n_rows - 1):
            p1 = tuple(pts[row, col].astype(int))
            p2 = tuple(pts[row + 1, col].astype(int))
            cv2.line(vis, p1, p2, cor_bgr, espessura_linha)

        for row in range(n_rows):
            p = tuple(pts[row, col].astype(int))
            cv2.circle(vis, p, raio_ponto, cor_bgr, -1)
            cv2.circle(vis, p, raio_ponto, (0, 0, 0), 2)  # contorno preto para contraste

    return vis

# ============================================================
# 2. DETECTAR OS CANTOS EM CADA IMAGEM
# ============================================================

imagens = sorted(glob.glob(IMAGES_PATH))
if len(imagens) == 0:
    raise SystemExit(f"Nenhuma imagem encontrada em '{IMAGES_PATH}'. Ajuste IMAGES_PATH.")

print(f"Encontradas {len(imagens)} imagens. Processando...")

image_size = None

for fname in imagens:
    img = cv2.imread(fname)
    if img is None:
        print(f"  [aviso] não consegui abrir {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = gray.shape[::-1]  # (largura, altura)

    # Tenta achar os cantos com o detector mais moderno e robusto (SB = "Sector Based"),
    # que lida melhor com tabuleiros de baixo contraste / madeira / iluminação irregular.
    found, corners = cv2.findChessboardCornersSB(
        gray, CHESSBOARD_SIZE,
        flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
    )

    # Se não achou, tenta o detector clássico como fallback
    if not found:
        found, corners = cv2.findChessboardCorners(
            gray, CHESSBOARD_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                  + cv2.CALIB_CB_FAST_CHECK
        )

    if found:
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        nomes_usados.append(fname)

        # Salva uma visualização dos cantos detectados (útil para o relatório),
        # no estilo colorido do tutorial oficial do OpenCV
        vis = desenhar_padrao_estilo_tutorial(img, corners_refined, CHESSBOARD_SIZE)
        out_name = os.path.join(OUTPUT_DIR, f"cantos_{os.path.basename(fname)}")
        cv2.imwrite(out_name, vis)

        print(f"  [ok] {fname} -> cantos detectados (visualização salva em {out_name})")
    else:
        print(f"  [ ! ] {fname} -> cantos NÃO detectados (verifique CHESSBOARD_SIZE ou a foto)")

print(f"\nTotal de imagens usadas na calibração: {len(objpoints)} de {len(imagens)}")

if len(objpoints) < 5:
    raise SystemExit(
        "Poucas imagens com cantos detectados (mínimo recomendado: ~10-15). "
        "Verifique CHESSBOARD_SIZE, iluminação e foco das fotos."
    )

# ============================================================
# 3. CALIBRAR A CÂMERA
# ============================================================

print("\nCalibrando câmera...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)

print("\n=== RESULTADO DA CALIBRAÇÃO ===")
print(f"Erro de reprojeção (RMS): {ret:.4f} pixels")
print("\nMatriz intrínseca K:")
print(mtx)
print("\nCoeficientes de distorção [k1, k2, p1, p2, k3]:")
print(dist.ravel())

# ============================================================
# 4. CALCULAR ERRO DE REPROJEÇÃO POR IMAGEM (para o relatório)
# ============================================================

print("\nErro de reprojeção por imagem:")
erros_por_imagem = []
for i in range(len(objpoints)):
    imgpoints_proj, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist
    )
    observado = np.asarray(imgpoints[i], dtype=np.float64).reshape(-1, 2)
    previsto = np.asarray(imgpoints_proj, dtype=np.float64).reshape(-1, 2)
    erro = float(np.linalg.norm(observado - previsto, axis=1).mean())
    erros_por_imagem.append(erro)
    print(f"  {os.path.basename(nomes_usados[i])}: {erro:.4f} px")

erro_medio = float(np.mean(erros_por_imagem))
print(f"\nErro médio de reprojeção: {erro_medio:.4f} px")

# ============================================================
# 5. SALVAR RESULTADOS (matrizes + coeficientes) EM ARQUIVO
# ============================================================

np.savez(
    os.path.join(OUTPUT_DIR, "calibracao.npz"),
    mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs,
    image_size=image_size, erro_rms=ret
)

resultado_json = {
    "erro_rms_reprojeção": ret,
    "erro_medio_por_imagem": erro_medio,
    "image_size": image_size,
    "matriz_intrinseca_K": mtx.tolist(),
    "coeficientes_distorcao": dist.ravel().tolist(),
    "numero_imagens_usadas": len(objpoints),
    "chessboard_size": CHESSBOARD_SIZE,
    "square_size_mm": SQUARE_SIZE_MM,
}
with open(os.path.join(OUTPUT_DIR, "calibracao.json"), "w") as f:
    json.dump(resultado_json, f, indent=2, ensure_ascii=False)

print(f"\nResultados salvos em '{OUTPUT_DIR}/calibracao.npz' e 'calibracao.json'")

# ============================================================
# 6. REMOVER A DISTORÇÃO DE UMA IMAGEM DE TESTE
# ============================================================

img_teste = cv2.imread(nomes_usados[0])
h, w = img_teste.shape[:2]

new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha=1, newImgSize=(w, h))
img_undist = cv2.undistort(img_teste, mtx, dist, None, new_mtx)

# Monta uma imagem lado a lado: original (distorcida) vs corrigida
comparacao = np.hstack((img_teste, img_undist))
cv2.imwrite(os.path.join(OUTPUT_DIR, "comparacao_distorcao.jpg"), comparacao)
cv2.imwrite(os.path.join(OUTPUT_DIR, "imagem_corrigida.jpg"), img_undist)
print(f"\nComparação distorcida/corrigida salva em '{OUTPUT_DIR}/comparacao_distorcao.jpg'")

# ============================================================
# 7. EXPERIMENTO: DADO UM PONTO 3D, PREVER ONDE ELE APARECE NA IMAGEM
# ============================================================
#
# Ideia: usar a pose (rvec, tvec) já calculada para uma das fotos do
# tabuleiro (ela já corresponde a uma posição real conhecida da câmera
# em relação ao tabuleiro). A partir disso, escolhemos pontos 3D no
# referencial do tabuleiro (em mm) e projetamos na imagem com
# cv2.projectPoints. Depois comparamos com a posição observada.
#
# Você pode:
#   (a) escolher pontos que SÃO cantos do tabuleiro (já sabemos a posição
#       observada, pois foi detectada no passo 2) -> comparação automática
#   (b) escolher um ponto 3D arbitrário fora do tabuleiro, mas medido com
#       trena/régua, e clicar manualmente na imagem para comparar

print("\n=== EXPERIMENTO: PROJEÇÃO DE PONTOS 3D -> 2D ===")

# Escolhe a primeira imagem usada como referência do experimento
idx_ref = 0
img_ref_nome = nomes_usados[idx_ref]
rvec_ref = rvecs[idx_ref]
tvec_ref = tvecs[idx_ref]

print(f"Imagem de referência: {img_ref_nome}")

# Exemplo (a): reprojetar os próprios cantos do tabuleiro dessa imagem
# e comparar com os pontos detectados originalmente.
pontos_3d_exemplo = objp  # todos os cantos, em mm, no plano do tabuleiro (Z=0)
pontos_2d_previstos, _ = cv2.projectPoints(
    pontos_3d_exemplo, rvec_ref, tvec_ref, mtx, dist
)
pontos_2d_observados = imgpoints[idx_ref]

observado_exp = np.asarray(pontos_2d_observados, dtype=np.float64).reshape(-1, 2)
previsto_exp = np.asarray(pontos_2d_previstos, dtype=np.float64).reshape(-1, 2)
erro_experimento = float(np.linalg.norm(observado_exp - previsto_exp, axis=1).mean())
print(f"Erro médio (previsto vs observado) para {len(pontos_3d_exemplo)} pontos: {erro_experimento:.4f} px")

# Salva uma imagem mostrando os pontos observados (verde) vs previstos (vermelho)
img_exp = cv2.imread(img_ref_nome).copy()
for obs, prev in zip(pontos_2d_observados.reshape(-1, 2), pontos_2d_previstos.reshape(-1, 2)):
    cv2.circle(img_exp, tuple(obs.astype(int)), 6, (0, 255, 0), 2)   # observado = verde
    cv2.circle(img_exp, tuple(prev.astype(int)), 3, (0, 0, 255), -1)  # previsto = vermelho

cv2.imwrite(os.path.join(OUTPUT_DIR, "experimento_projecao.jpg"), img_exp)
print(f"Imagem do experimento salva em '{OUTPUT_DIR}/experimento_projecao.jpg'")
print("(círculo verde = ponto observado, ponto vermelho = ponto previsto pela projeção)")

# --------------------------------------------------------------
# Exemplo (b): projetar um ponto 3D arbitrário definido por você.
# Descomente e edite as coordenadas (em mm, no referencial do
# tabuleiro: X para a direita, Y para baixo, Z para fora do tabuleiro)
# --------------------------------------------------------------
#
# ponto_3d_customizado = np.array([[[120.0, 80.0, -50.0]]], dtype=np.float32)
# ponto_2d_previsto, _ = cv2.projectPoints(
#     ponto_3d_customizado, rvec_ref, tvec_ref, mtx, dist
# )
# print("Ponto 3D customizado projetado em:", ponto_2d_previsto.ravel())

print("\nConcluído. Confira a pasta:", OUTPUT_DIR)