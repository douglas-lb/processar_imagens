"""
Avalia o modelo treinado no conjunto de teste.

Uso:

    python avaliar.py                        # teste do MNIST
    python avaliar.py --dataset dataset/     # teste do dataset próprio (splits.json)
    python avaliar.py --tflite               # avalia o modelo TFLite

Mostra a acurácia, o relatório por classe e a matriz de confusão,
que também é salva em modelos/matriz_confusao.png.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from prever import ModeloDigitos
from treinar_modelo import (
    NUM_CLASSES,
    PASTA_MODELOS,
    PASTA_PROJETO,
    carregar_dataset_proprio,
    carregar_mnist,
)


CAMINHO_MATRIZ = PASTA_MODELOS / "matriz_confusao.png"


def salvar_matriz(matriz, titulo, caminho=CAMINHO_MATRIZ):

    figura, eixo = plt.subplots(figsize=(8, 7))

    ConfusionMatrixDisplay(matriz, display_labels=list(range(NUM_CLASSES))).plot(
        ax=eixo,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )

    eixo.set_title(titulo)
    eixo.set_xlabel("Previsto")
    eixo.set_ylabel("Verdadeiro")

    figura.tight_layout()
    figura.savefig(caminho, dpi=120)
    plt.close(figura)


def executar(argumentos):

    if argumentos.dataset:
        dados, estrategia = carregar_dataset_proprio(argumentos.dataset)
        nome_dataset = str(argumentos.dataset)
    else:
        dados = carregar_mnist()
        estrategia = "MNIST oficial"
        nome_dataset = "MNIST"

    x_teste, y_teste = dados["teste"]

    if len(x_teste) == 0:
        print("Conjunto de teste vazio. Gere mais fotografias / o splits.json.")
        return

    modelo = ModeloDigitos(argumentos.modelo, usar_tflite=argumentos.tflite)

    probabilidades = modelo.prever_lote(x_teste)
    previsoes = np.argmax(probabilidades, axis=1)

    acuracia = accuracy_score(y_teste, previsoes)
    matriz = confusion_matrix(y_teste, previsoes, labels=list(range(NUM_CLASSES)))

    print("=" * 50)
    print("AVALIAÇÃO DO MODELO")
    print("=" * 50)
    print(f"Modelo: {modelo.caminho.name}")
    print(f"Dataset: {nome_dataset} ({estrategia})")
    print(f"Imagens de teste: {len(y_teste)}")
    print(f"\nAcurácia: {acuracia * 100:.2f}%\n")

    print(classification_report(
        y_teste,
        previsoes,
        labels=list(range(NUM_CLASSES)),
        digits=4,
        zero_division=0,
    ))

    print("Matriz de confusão (linhas = verdadeiro, colunas = previsto):")
    print("     " + " ".join(f"{d:>5}" for d in range(NUM_CLASSES)))

    for numero, linha in enumerate(matriz):
        print(f"{numero:>3}  " + " ".join(f"{v:>5}" for v in linha))

    destino = Path(argumentos.saida) if argumentos.saida else CAMINHO_MATRIZ
    destino.parent.mkdir(parents=True, exist_ok=True)

    salvar_matriz(
        matriz,
        f"{nome_dataset} - acurácia {acuracia * 100:.2f}%",
        destino,
    )

    try:
        exibido = destino.resolve().relative_to(PASTA_PROJETO)
    except ValueError:
        exibido = destino

    print(f"\nMatriz de confusão salva em {exibido}")


def ler_argumentos():

    parser = argparse.ArgumentParser(
        description="Avalia o modelo de dígitos no conjunto de teste."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        help="pasta do dataset próprio (ex.: dataset/). Sem essa opção, usa o teste do MNIST.",
    )

    parser.add_argument("--modelo", type=Path, help="arquivo .keras ou .tflite")

    parser.add_argument(
        "--tflite",
        action="store_true",
        help="avalia modelos/digitos.tflite em vez do modelo Keras",
    )

    parser.add_argument(
        "--saida",
        help=f"onde salvar a imagem da matriz (padrão: {CAMINHO_MATRIZ.relative_to(PASTA_PROJETO)})",
    )

    return parser.parse_args()


if __name__ == "__main__":
    executar(ler_argumentos())
