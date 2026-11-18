"""
Treinamento da rede neural convolucional (CNN) que reconhece dígitos de 0 a 9.

Stack: TensorFlow/Keras, com exportação também para TFLite.
O TensorFlow ainda não tem versão para Python 3.14: use Python 3.10–3.13.
Exemplo (Linux, com uv):

    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements.txt

Uso:

    python treinar_modelo.py                              # treina com o MNIST
    python treinar_modelo.py --dataset dataset/           # treina com o dataset próprio
    python treinar_modelo.py --dataset dataset/ --com-mnist   # dataset próprio + MNIST no treino

Arquivos gerados:

    modelos/digitos.keras      modelo Keras
    modelos/digitos.tflite     modelo TFLite (para celular / Raspberry etc.)
    modelos/digitos_info.json  informações do treinamento

Formato de entrada do modelo:
    float32 com forma (N, 28, 28, 1), valores de 0 a 255,
    fundo preto e número branco (o mesmo padrão do preprocess.py).
    A divisão por 255 é feita dentro do próprio modelo.

Saída: probabilidades com forma (N, 10), uma por dígito.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import json
import random

from datetime import datetime
from pathlib import Path

import cv2
import keras
import numpy as np
import tensorflow as tf

from keras import layers

from preprocess import LIMIAR_BINARIZACAO, SEMENTE, TAMANHO_FINAL


PASTA_PROJETO = Path(__file__).resolve().parent

PASTA_MODELOS = PASTA_PROJETO / "modelos"
CAMINHO_KERAS = PASTA_MODELOS / "digitos.keras"
CAMINHO_TFLITE = PASTA_MODELOS / "digitos.tflite"
CAMINHO_INFO = PASTA_MODELOS / "digitos_info.json"

NUM_CLASSES = 10

EXTENSOES = {".png", ".jpg", ".jpeg", ".bmp"}

# Divisão usada quando o dataset próprio não tem splits.json.
PROPORCAO_VALIDACAO = 0.1
PROPORCAO_TESTE = 0.1

# Sem metadata.csv não sabemos de qual fotografia cada imagem veio.
# O processar.py salva a original seguida de 5 aumentos com números
# consecutivos, então blocos de 6 arquivos formam um mesmo grupo.
TAMANHO_GRUPO_SEM_METADATA = 6

# Quantidade de imagens do MNIST reservadas para validação.
TAMANHO_VALIDACAO_MNIST = 6000

EPOCAS_MNIST = 5
EPOCAS_DATASET = 30
TAMANHO_LOTE = 128


def fixar_semente():
    """
    Deixa o treinamento reprodutível.
    """

    random.seed(SEMENTE)
    np.random.seed(SEMENTE)
    keras.utils.set_random_seed(SEMENTE)
    tf.config.experimental.enable_op_determinism()


def binarizar(imagens):
    """
    Converte para 0/255, igual ao preprocess.py.

    O MNIST tem bordas em tons de cinza, mas as imagens da câmera e do
    dataset próprio são binárias; o treino precisa ver o mesmo padrão.
    """

    return np.where(imagens >= LIMIAR_BINARIZACAO, 255, 0).astype(np.uint8)


def carregar_mnist():
    """
    Baixa (na primeira vez) e devolve o MNIST dividido em
    treino / validação / teste, já binarizado.
    """

    (x_treino, y_treino), (x_teste, y_teste) = keras.datasets.mnist.load_data()

    indices = np.random.default_rng(SEMENTE).permutation(len(x_treino))
    x_treino, y_treino = x_treino[indices], y_treino[indices]

    corte = len(x_treino) - TAMANHO_VALIDACAO_MNIST

    return {
        "treino": (binarizar(x_treino[:corte]), y_treino[:corte]),
        "validacao": (binarizar(x_treino[corte:]), y_treino[corte:]),
        "teste": (binarizar(x_teste), y_teste),
    }


def listar_dataset(pasta):
    """
    Lista as imagens de pasta/0 ... pasta/9 como caminhos
    relativos (ex.: "7/7_00001.png").
    """

    arquivos = []

    for numero in range(NUM_CLASSES):

        pasta_classe = pasta / str(numero)

        if not pasta_classe.is_dir():
            continue

        arquivos.extend(
            f"{numero}/{caminho.name}"
            for caminho in sorted(pasta_classe.iterdir())
            if caminho.is_file() and caminho.suffix.lower() in EXTENSOES
        )

    return arquivos


def classe_do_arquivo(arquivo):
    """
    O rótulo é o nome da pasta: "7/7_00001.png" -> 7.
    """

    return int(Path(arquivo).parent.name)


def dividir_grupos(grupos_por_classe):
    """
    Divide grupos de imagens em treino / validação / teste,
    separadamente em cada classe.

    grupos_por_classe: {classe: {id_do_grupo: [arquivos]}}

    Imagens do mesmo grupo (mesma fotografia) nunca ficam em
    conjuntos diferentes.
    """

    gerador = random.Random(SEMENTE)

    divisao = {"treino": [], "validacao": [], "teste": []}

    for classe in sorted(grupos_por_classe):

        grupos = grupos_por_classe[classe]
        ids = sorted(grupos)
        gerador.shuffle(ids)

        quantidade = len(ids)
        quantidade_teste = int(quantidade * PROPORCAO_TESTE + 0.5)
        quantidade_validacao = int(quantidade * PROPORCAO_VALIDACAO + 0.5)

        # Com poucas fotografias, garante pelo menos uma em cada conjunto.
        if quantidade >= 3:
            quantidade_teste = max(1, quantidade_teste)
            quantidade_validacao = max(1, quantidade_validacao)

        partes = {
            "teste": ids[:quantidade_teste],
            "validacao": ids[quantidade_teste:quantidade_teste + quantidade_validacao],
            "treino": ids[quantidade_teste + quantidade_validacao:],
        }

        for nome, ids_parte in partes.items():
            for id_grupo in ids_parte:
                divisao[nome].extend(grupos[id_grupo])

    return divisao


def dividir_dataset_proprio(pasta):
    """
    Decide quais imagens vão para treino, validação e teste.

    Ordem de preferência:
        1. dataset/splits.json (gerado pelo processar.py);
        2. dataset/metadata.csv, agrupando por source_id;
        3. blocos de imagens consecutivas de cada classe.

    Retorna (divisao, estrategia).
    """

    existentes = set(listar_dataset(pasta))

    arquivo_splits = pasta / "splits.json"

    if arquivo_splits.exists():

        with open(arquivo_splits, encoding="utf-8") as arquivo:
            splits = json.load(arquivo)

        divisao = {
            nome: [a for a in splits.get(nome, []) if a in existentes]
            for nome in ("treino", "validacao", "teste")
        }

        return divisao, "splits.json (por source_id)"

    arquivo_metadata = pasta / "metadata.csv"

    if arquivo_metadata.exists():

        with open(arquivo_metadata, newline="", encoding="utf-8") as arquivo:
            linhas = list(csv.DictReader(arquivo))

        grupos_por_classe = {}

        for linha in linhas:

            if linha["arquivo"] not in existentes:
                continue

            grupos_por_classe.setdefault(int(linha["classe"]), {}).setdefault(
                linha["source_id"], []
            ).append(linha["arquivo"])

        return dividir_grupos(grupos_por_classe), "metadata.csv (por source_id)"

    print(
        "Aviso: dataset sem splits.json e sem metadata.csv. "
        f"Agrupando blocos de {TAMANHO_GRUPO_SEM_METADATA} imagens "
        "consecutivas (original + aumentos) por classe."
    )

    grupos_por_classe = {}

    for arquivo in sorted(existentes):

        classe = classe_do_arquivo(arquivo)
        grupos = grupos_por_classe.setdefault(classe, {})
        posicao = sum(len(g) for g in grupos.values())

        grupos.setdefault(posicao // TAMANHO_GRUPO_SEM_METADATA, []).append(arquivo)

    return dividir_grupos(grupos_por_classe), "blocos consecutivos (sem metadata)"


def ler_imagem_28x28(caminho):
    """
    Lê uma imagem do dataset em escala de cinza, 28x28.
    """

    imagem = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)

    if imagem is None:
        return None

    if imagem.shape != (TAMANHO_FINAL, TAMANHO_FINAL):
        imagem = cv2.resize(
            imagem,
            (TAMANHO_FINAL, TAMANHO_FINAL),
            interpolation=cv2.INTER_AREA
        )

    return imagem


def carregar_arquivos(pasta, arquivos):
    """
    Carrega uma lista de arquivos e devolve (x, y).
    """

    imagens = []
    rotulos = []

    for arquivo in arquivos:

        imagem = ler_imagem_28x28(pasta / arquivo)

        if imagem is None:
            print(f"Aviso: não foi possível ler {arquivo}")
            continue

        imagens.append(imagem)
        rotulos.append(classe_do_arquivo(arquivo))

    x = np.array(imagens, dtype=np.uint8).reshape(-1, TAMANHO_FINAL, TAMANHO_FINAL)
    y = np.array(rotulos, dtype=np.int64)

    return binarizar(x), y


def carregar_dataset_proprio(pasta):
    """
    Devolve ({"treino": (x, y), "validacao": ..., "teste": ...}, estrategia).
    """

    pasta = Path(pasta)

    if not pasta.is_absolute():
        pasta = (Path.cwd() / pasta).resolve()

    divisao, estrategia = dividir_dataset_proprio(pasta)

    dados = {
        nome: carregar_arquivos(pasta, arquivos)
        for nome, arquivos in divisao.items()
    }

    return dados, estrategia


def criar_modelo():
    """
    CNN pequena para imagens 28x28 em escala de cinza.
    """

    return keras.Sequential(
        [
            keras.Input(shape=(TAMANHO_FINAL, TAMANHO_FINAL, 1)),
            layers.Rescaling(1.0 / 255),

            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),

            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="cnn_digitos",
    )


def criar_aumento():
    """
    Pequenas rotações, deslocamentos e zoom aplicados só no treino,
    para o modelo tolerar fotografias imperfeitas.
    """

    return keras.Sequential(
        [
            # 0.035 de uma volta completa ≈ 12,6 graus.
            layers.RandomRotation(0.035, fill_mode="constant", seed=SEMENTE),
            layers.RandomTranslation(0.08, 0.08, fill_mode="constant", seed=SEMENTE),
            layers.RandomZoom(0.1, fill_mode="constant", seed=SEMENTE),
        ],
        name="aumento",
    )


def criar_tf_dataset(x, y, treino=False, tamanho_lote=TAMANHO_LOTE):
    """
    Monta o tf.data.Dataset; no treino embaralha e aplica aumento de dados.
    """

    x = x.astype(np.float32)[..., np.newaxis]

    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    if treino:

        aumento = criar_aumento()
        limiar = float(LIMIAR_BINARIZACAO)

        def aumentar(imagens, rotulos):
            imagens = aumento(imagens, training=True)
            # A interpolação cria tons de cinza; volta para 0/255.
            imagens = tf.where(imagens >= limiar, 255.0, 0.0)
            return imagens, rotulos

        dataset = (
            dataset
            .shuffle(len(x), seed=SEMENTE, reshuffle_each_iteration=True)
            .batch(tamanho_lote)
            .map(aumentar, num_parallel_calls=tf.data.AUTOTUNE)
        )

    else:
        dataset = dataset.batch(tamanho_lote)

    return dataset.prefetch(tf.data.AUTOTUNE)


def exportar_tflite(modelo, caminho=CAMINHO_TFLITE):
    """
    Converte o modelo Keras para TFLite. Retorna True se deu certo.
    """

    try:
        conversor = tf.lite.TFLiteConverter.from_keras_model(modelo)
        conteudo = conversor.convert()
    except Exception as erro:
        print(f"Aviso: não foi possível exportar para TFLite ({erro}).")
        return False

    caminho.write_bytes(conteudo)

    return True


def contar_por_classe(y):
    return [int(np.sum(y == numero)) for numero in range(NUM_CLASSES)]


def treinar(argumentos):

    fixar_semente()

    PASTA_MODELOS.mkdir(exist_ok=True)

    if argumentos.dataset:

        dados, estrategia = carregar_dataset_proprio(argumentos.dataset)
        nome_dataset = str(argumentos.dataset)

        if argumentos.com_mnist:

            mnist = carregar_mnist()
            x_treino, y_treino = dados["treino"]

            dados["treino"] = (
                np.concatenate([x_treino, mnist["treino"][0]]),
                np.concatenate([y_treino, mnist["treino"][1]]),
            )

            nome_dataset += " + MNIST (treino)"

        epocas = argumentos.epocas or EPOCAS_DATASET

    else:

        dados = carregar_mnist()
        estrategia = "MNIST oficial (validação = 6000 imagens do treino)"
        nome_dataset = "MNIST"
        epocas = argumentos.epocas or EPOCAS_MNIST

    x_treino, y_treino = dados["treino"]
    x_validacao, y_validacao = dados["validacao"]
    x_teste, y_teste = dados["teste"]

    print("=" * 50)
    print("TREINAMENTO DA CNN DE DÍGITOS")
    print("=" * 50)
    print(f"Dataset: {nome_dataset}")
    print(f"Divisão: {estrategia}")
    print(f"Treino: {len(x_treino)} | Validação: {len(x_validacao)} | Teste: {len(x_teste)}")
    print(f"Imagens de treino por classe: {contar_por_classe(y_treino)}")
    print(f"Épocas: {epocas}")

    if len(x_treino) == 0:
        print("Nenhuma imagem de treino encontrada. Rode o processar.py primeiro.")
        return

    modelo = criar_modelo()
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    modelo.summary()

    treino = criar_tf_dataset(x_treino, y_treino, treino=True, tamanho_lote=argumentos.lote)
    validacao = None
    callbacks = []

    if len(x_validacao) > 0:

        validacao = criar_tf_dataset(x_validacao, y_validacao, tamanho_lote=argumentos.lote)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=5,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
            ),
        ]

    modelo.fit(
        treino,
        validation_data=validacao,
        epochs=epocas,
        callbacks=callbacks,
        verbose=2,
    )

    acuracia_teste = None

    if len(x_teste) > 0:
        _, acuracia_teste = modelo.evaluate(
            criar_tf_dataset(x_teste, y_teste, tamanho_lote=argumentos.lote),
            verbose=0,
        )
        acuracia_teste = float(acuracia_teste)
    else:
        print("Aviso: conjunto de teste vazio; acurácia de teste não calculada.")

    modelo.save(CAMINHO_KERAS)
    tflite_ok = exportar_tflite(modelo, CAMINHO_TFLITE)

    info = {
        "data": datetime.now().isoformat(timespec="seconds"),
        "dataset": nome_dataset,
        "divisao": estrategia,
        "epocas_solicitadas": epocas,
        "semente": SEMENTE,
        "quantidades": {
            "treino": int(len(x_treino)),
            "validacao": int(len(x_validacao)),
            "teste": int(len(x_teste)),
        },
        "acuracia_teste": acuracia_teste,
        "entrada": {
            "forma": [1, TAMANHO_FINAL, TAMANHO_FINAL, 1],
            "tipo": "float32",
            "valores": "0 a 255 (binário 0/255), fundo preto e número branco",
            "preprocessamento": "preprocess.preprocessar_digito",
        },
        "saida": "probabilidades softmax para as classes 0..9",
        "arquivos": {
            "keras": CAMINHO_KERAS.name,
            "tflite": CAMINHO_TFLITE.name if tflite_ok else None,
        },
        "versoes": {
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
        },
    }

    with open(CAMINHO_INFO, "w", encoding="utf-8") as arquivo:
        json.dump(info, arquivo, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("TREINAMENTO CONCLUÍDO")
    print("=" * 50)

    if acuracia_teste is not None:
        print(f"Acurácia no teste: {acuracia_teste * 100:.2f}%")

    print(f"Modelo Keras: {CAMINHO_KERAS.relative_to(PASTA_PROJETO)}")

    if tflite_ok:
        print(f"Modelo TFLite: {CAMINHO_TFLITE.relative_to(PASTA_PROJETO)}")

    print(f"Informações: {CAMINHO_INFO.relative_to(PASTA_PROJETO)}")


def ler_argumentos():

    parser = argparse.ArgumentParser(
        description="Treina a CNN de reconhecimento de dígitos (0-9)."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        help="pasta com subpastas 0..9 (ex.: dataset/). Sem essa opção, usa o MNIST.",
    )

    parser.add_argument(
        "--com-mnist",
        action="store_true",
        help="junta o MNIST ao treino do dataset próprio (validação e teste continuam só no dataset próprio)",
    )

    parser.add_argument(
        "--epocas",
        type=int,
        help=f"quantidade de épocas (padrão: {EPOCAS_MNIST} no MNIST, {EPOCAS_DATASET} no dataset próprio)",
    )

    parser.add_argument(
        "--lote",
        type=int,
        default=TAMANHO_LOTE,
        help=f"tamanho do lote (padrão: {TAMANHO_LOTE})",
    )

    return parser.parse_args()


if __name__ == "__main__":
    treinar(ler_argumentos())
