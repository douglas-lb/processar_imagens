"""
Reconhece o dígito de uma ou mais imagens usando o modelo treinado.

Uso:

    python prever.py foto.jpg
    python prever.py foto1.jpg dataset/7/7_00001.png
    python prever.py foto.jpg --tflite            # usa modelos/digitos.tflite
    python prever.py foto.jpg --salvar-preprocessada /tmp/entrada.png

Aceita tanto fotografias (número escuro em papel claro) quanto
imagens já no padrão MNIST (28x28, fundo preto e número branco).

Para usar em outro script (ex.: câmera):

    from prever import ModeloDigitos, preparar_imagem

    modelo = ModeloDigitos()
    imagem_28x28 = preparar_imagem(quadro_bgr)
    digito, confianca, probabilidades = modelo.prever(imagem_28x28)
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse

from pathlib import Path

import cv2
import numpy as np

from preprocess import (
    TAMANHO_FINAL,
    carregar_imagem,
    normalizar_binaria,
    preprocessar_digito,
)
from treinar_modelo import CAMINHO_KERAS, CAMINHO_TFLITE, binarizar


# Fotografias maiores que isso recebem desfoque antes da binarização.
TAMANHO_MINIMO_DESFOQUE = TAMANHO_FINAL * 2


def fundo_escuro(imagem_cinza):
    """
    Indica se a imagem tem fundo escuro (número claro), olhando a borda.
    """

    borda = np.concatenate([
        imagem_cinza[0, :],
        imagem_cinza[-1, :],
        imagem_cinza[:, 0],
        imagem_cinza[:, -1],
    ])

    return np.median(borda) < 128


def preparar_imagem(entrada):
    """
    Converte uma imagem (caminho ou np.ndarray BGR/cinza) no padrão
    usado pelo modelo: 28x28 uint8, fundo preto, número branco (0/255).

    Retorna None se não for possível ler ou encontrar o número.
    """

    imagem_cinza = carregar_imagem(entrada)

    if imagem_cinza is None:
        return None

    if not fundo_escuro(imagem_cinza):
        # Fotografia comum: número escuro sobre papel claro.
        return preprocessar_digito(imagem_cinza)

    # Já está no padrão MNIST (ou é número claro sobre fundo escuro):
    # binariza sem inverter e recentraliza do mesmo jeito.
    if max(imagem_cinza.shape) >= TAMANHO_MINIMO_DESFOQUE:
        imagem_cinza = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)

    _, imagem_binaria = cv2.threshold(
        imagem_cinza,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return normalizar_binaria(imagem_binaria)


class ModeloDigitos:
    """
    Carrega o modelo Keras (.keras) ou TFLite (.tflite) e faz previsões.
    """

    def __init__(self, caminho=None, usar_tflite=False):

        if caminho is None:
            caminho = CAMINHO_TFLITE if usar_tflite else CAMINHO_KERAS

            # Em um clone novo o .keras pode não existir (fica no .gitignore).
            if not caminho.exists() and CAMINHO_TFLITE.exists():
                caminho = CAMINHO_TFLITE

        caminho = Path(caminho)

        if not caminho.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {caminho}. Rode antes: python treinar_modelo.py"
            )

        self.caminho = caminho
        self.tflite = caminho.suffix == ".tflite"

        import tensorflow as tf

        if self.tflite:
            self.interpretador = tf.lite.Interpreter(model_path=str(caminho))
            self.interpretador.allocate_tensors()
            self.entrada = self.interpretador.get_input_details()[0]
            self.saida = self.interpretador.get_output_details()[0]
        else:
            import keras
            self.modelo = keras.models.load_model(caminho)

    def prever_lote(self, imagens):
        """
        imagens: uint8 (N, 28, 28), fundo preto e número branco.
        Retorna as probabilidades (N, 10).
        """

        x = binarizar(np.asarray(imagens)).astype(np.float32)
        x = x.reshape(-1, TAMANHO_FINAL, TAMANHO_FINAL, 1)

        if not self.tflite:
            return self.modelo.predict(x, verbose=0)

        probabilidades = []

        for imagem in x:
            self.interpretador.set_tensor(self.entrada["index"], imagem[np.newaxis])
            self.interpretador.invoke()
            probabilidades.append(self.interpretador.get_tensor(self.saida["index"])[0])

        return np.array(probabilidades)

    def prever(self, imagem_28x28):
        """
        Retorna (digito, confianca, probabilidades) para uma imagem 28x28.
        """

        probabilidades = self.prever_lote(imagem_28x28[np.newaxis])[0]
        digito = int(np.argmax(probabilidades))

        return digito, float(probabilidades[digito]), probabilidades


def mostrar(imagem_original, imagem_28x28, titulo):

    ampliada = cv2.resize(imagem_28x28, (280, 280), interpolation=cv2.INTER_NEAREST)

    cv2.imshow(f"{titulo} - original", imagem_original)
    cv2.imshow(f"{titulo} - entrada do modelo", ampliada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def executar(argumentos):

    modelo = ModeloDigitos(argumentos.modelo, usar_tflite=argumentos.tflite)

    print(f"Modelo: {modelo.caminho.name}")

    for caminho in argumentos.imagens:

        imagem_28x28 = preparar_imagem(caminho)

        if imagem_28x28 is None:
            print(f"{caminho}: não foi possível ler a imagem ou encontrar um número.")
            continue

        digito, confianca, probabilidades = modelo.prever(imagem_28x28)

        top3 = np.argsort(probabilidades)[::-1][:3]
        alternativas = ", ".join(f"{d}: {probabilidades[d] * 100:.1f}%" for d in top3)

        print(f"{caminho}: dígito {digito} (confiança {confianca * 100:.1f}%) | top 3 -> {alternativas}")

        if argumentos.salvar_preprocessada:
            destino = Path(argumentos.salvar_preprocessada)

            if len(argumentos.imagens) > 1:
                destino = destino.with_name(f"{destino.stem}_{Path(caminho).stem}{destino.suffix}")

            cv2.imwrite(str(destino), imagem_28x28)

        if argumentos.mostrar:
            mostrar(cv2.imread(str(caminho)), imagem_28x28, Path(caminho).name)


def ler_argumentos():

    parser = argparse.ArgumentParser(
        description="Reconhece o dígito (0-9) presente em uma imagem."
    )

    parser.add_argument("imagens", nargs="+", help="caminho(s) da(s) imagem(ns)")

    parser.add_argument(
        "--modelo",
        type=Path,
        help=f"arquivo .keras ou .tflite (padrão: {CAMINHO_KERAS.relative_to(CAMINHO_KERAS.parents[1])})",
    )

    parser.add_argument(
        "--tflite",
        action="store_true",
        help="usa modelos/digitos.tflite em vez do modelo Keras",
    )

    parser.add_argument(
        "--salvar-preprocessada",
        help="salva a imagem 28x28 que entrou no modelo (útil para conferir o pré-processamento)",
    )

    parser.add_argument(
        "--mostrar",
        action="store_true",
        help="mostra a imagem original e a imagem 28x28 em janelas",
    )

    return parser.parse_args()


if __name__ == "__main__":
    executar(ler_argumentos())
