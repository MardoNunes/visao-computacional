import cv2 as cv
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import argparse

def proximo_destino(pasta, stem, sufixo):
    contador = 1
    while True:
        destino = pasta / f"{stem}_perspectiva_{contador}{sufixo}"
        if not destino.exists():
            return destino
        contador += 1

def selecionar_pontos(img, n_pontos=4, janela="Selecione os 4 pontos"):
    """Abre a imagem e pega 4 pontos clicados pelo usuário.

    Teclas: 'r' reinicia a seleção, 'q' ou ESC cancela.
    Retorna um array float32 (n_pontos, 2) ou None se cancelado.
    """
    pontos = []
    exibicao = img.copy()

    def on_click(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN and len(pontos) < n_pontos:
            pontos.append([x, y])
            cv.circle(exibicao, (x, y), 5, (0, 0, 255), -1)
            cv.putText(exibicao, str(len(pontos)), (x + 8, y - 8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.imshow(janela, exibicao)

    cv.namedWindow(janela)
    cv.setMouseCallback(janela, on_click)
    cv.imshow(janela, exibicao)

    while True:
        tecla = cv.waitKey(20) & 0xFF
        if len(pontos) == n_pontos:
            cv.waitKey(300)  # pequena pausa para ver o último ponto
            break
        if tecla == ord('r'):
            pontos.clear()
            exibicao = img.copy()
            cv.imshow(janela, exibicao)
        elif tecla in (ord('q'), 27):
            cv.destroyWindow(janela)
            return None
        # fecha se o usuário clicar no X da janela
        if cv.getWindowProperty(janela, cv.WND_PROP_VISIBLE) < 1:
            return None

    cv.destroyWindow(janela)
    return np.float32(pontos)


def ordenar_pontos(pts):
    """Ordena em: topo-esq, topo-dir, base-esq, base-dir,
    para o usuário poder clicar em qualquer ordem."""
    soma = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()  # y - x
    tl = pts[np.argmin(soma)]
    br = pts[np.argmax(soma)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.float32([tl, tr, bl, br])

def transformar_perspectiva(caminho, saida=(300, 300), mostrar=True):
    """
        Código semelhante ao que foi apresentado na bliblioteca do openCV
        Captura 4 pontos, nesse caso, pede ao usupario selecionar
        cv.getPerspectiveTransform(pts1, pts2) - função opnecv que calcula a matrix 3x3
        cv.warpPerspective(img, M, (w, h)) - aplica a matriz de transformação na imagem inteira
    """

    img = cv.imread(caminho)
    assert img is not None, f"Não foi possível ler '{caminho}'"

    pts1 = selecionar_pontos(img)
    if pts1 is None:
        print("Seleção cancelada.")
        return None

    pts1 = ordenar_pontos(pts1)
    w, h = saida
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img, M, (w, h))

    if mostrar:
        plt.subplot(121), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title('Input')
        plt.subplot(122), plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB)), plt.title('Output')
        plt.show()

    pasta = pathlib.Path("saida")
    pasta.mkdir(exist_ok=True)
    origem = pathlib.Path(caminho)
    destino = pasta / f"{origem.stem}_perspectiva{origem.suffix}"
    destino = proximo_destino(pasta, origem.stem, origem.suffix)
    if not cv.imwrite(str(destino), dst):
        raise IOError(f"Não foi possível salvar em '{destino}'")
    print(f"Resultado salvo em: {destino.resolve()}")

    return dst, M, pts1


def main():
    parser = argparse.ArgumentParser(description="Transformação de perspectiva")
    parser.add_argument("imagem", help="caminho (relativo ou absoluto) da imagem de entrada")
    parser.add_argument("--largura", type=int, default=300)
    parser.add_argument("--altura", type=int, default=300)
    parser.add_argument("--sem-mostrar", action="store_true", help="não exibe o resultado no matplotlib")
    args = parser.parse_args()

    transformar_perspectiva(args.imagem, saida=(args.largura, args.altura), mostrar=not args.sem_mostrar)

if __name__ == "__main__":
    main()