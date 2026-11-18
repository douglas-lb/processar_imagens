"""
Pré-processamento compartilhado de dígitos manuscritos.

Este módulo concentra as etapas que transformam uma fotografia (ou um
quadro da câmera) em uma imagem no padrão MNIST:

    - 28x28 pixels;
    - fundo preto (0);
    - número branco (255);
    - imagem estritamente binária (somente 0 e 255).

Ele é usado pelo processar.py (criação do dataset) e pode ser importado
pela câmera / inferência para garantir que o modelo receba imagens
processadas exatamente da mesma forma que no treinamento:

    from preprocess import preprocessar_digito

    imagem_28x28 = preprocessar_digito(quadro_bgr)
"""

import random

from pathlib import Path

import cv2
import numpy as np


# Tamanho final das imagens.
TAMANHO_FINAL = 28

# Tamanho aproximado que o número ocupará dentro da imagem.
TAMANHO_NUMERO = 20

# Semente para tornar os resultados reproduzíveis.
SEMENTE = 42

# Limiar usado para voltar a imagem para 0/255 depois de
# redimensionamentos e transformações geométricas.
# É baixo de propósito: traços finos ficam acinzentados após
# a redução para 20x20 e não podem desaparecer.
LIMIAR_BINARIZACAO = 64

# Borda preta adicionada antes da rotação para que os traços
# do número não sejam cortados pelos limites da imagem.
MARGEM_AUMENTO = 8

# Limites do aumento de dados.
ANGULO_MAXIMO = 12
DESLOCAMENTO_MAXIMO = 2


def carregar_imagem(entrada):
    """
    Aceita um caminho (str ou Path) ou uma imagem já carregada
    (np.ndarray BGR ou em escala de cinza).

    Retorna a imagem em escala de cinza ou None se não for possível ler.
    """

    if isinstance(entrada, (str, Path)):
        imagem = cv2.imread(str(entrada))
    else:
        imagem = entrada

    if imagem is None or not isinstance(imagem, np.ndarray) or imagem.size == 0:
        return None

    if imagem.ndim == 2:
        return imagem

    if imagem.shape[2] == 4:
        return cv2.cvtColor(imagem, cv2.COLOR_BGRA2GRAY)

    return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


def rebinarizar(imagem, limiar=LIMIAR_BINARIZACAO):
    """
    Força a imagem a conter apenas os valores 0 e 255.
    """

    _, binaria = cv2.threshold(
        imagem,
        limiar,
        255,
        cv2.THRESH_BINARY
    )

    return binaria


def binarizar_fotografia(imagem_cinza):
    """
    Separa o número do fundo de uma fotografia em escala de cinza.

    Supõe número escuro sobre papel claro e devolve número branco
    sobre fundo preto.
    """

    # Aplica um desfoque leve para reduzir pequenas
    # imperfeições e ruídos da fotografia.
    imagem_suave = cv2.GaussianBlur(
        imagem_cinza,
        (5, 5),
        0
    )

    # binarização: deixa o número branco e o fundo preto.
    _, imagem_binaria = cv2.threshold(
        imagem_suave,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return imagem_binaria


def detectar_regiao_papel(imagem_cinza, percentil=60.0):
    """
    Encontra a maior região clara (folha de papel) na imagem.

    Retorna (x, y, w, h) do retângulo do papel ou None.
    """

    if imagem_cinza is None or imagem_cinza.size == 0:
        return None

    # Limita o limiar: se for alto demais (ex.: 230), pixels do papel
    # "claros iguais" falham no teste src > limiar do THRESH_BINARY.
    limiar = float(np.percentile(imagem_cinza, percentil))
    limiar = float(np.clip(limiar, 110.0, 200.0))

    _, mascara = cv2.threshold(
        imagem_cinza,
        limiar,
        255,
        cv2.THRESH_BINARY,
    )

    kernel = np.ones((7, 7), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contornos:
        return None

    altura, largura = imagem_cinza.shape[:2]
    area_min = 0.08 * altura * largura

    candidatos = [c for c in contornos if cv2.contourArea(c) >= area_min]

    if not candidatos:
        return None

    maior = max(candidatos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(maior)

    if w < 20 or h < 20:
        return None

    return x, y, w, h


def binarizar_adaptativo_camera(imagem_cinza, bloco=31, c=8):
    """
    Binarização adaptativa para webcam: número branco, fundo preto.

    Melhor que Otsu global quando o ROI mistura papel claro e fundo escuro.
    """

    if imagem_cinza is None or imagem_cinza.size == 0:
        return None

    suave = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)

    # Bloco precisa ser ímpar e >= 3.
    bloco = int(bloco)
    if bloco % 2 == 0:
        bloco += 1
    bloco = max(bloco, 11)

    binaria = cv2.adaptiveThreshold(
        suave,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        bloco,
        c,
    )

    kernel = np.ones((3, 3), np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=1)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=1)

    return binaria


def normalizar_binaria(imagem_binaria):
    """
    Recorta o número, centraliza em um quadrado, reduz para
    aproximadamente 20x20 e insere no centro de uma imagem 28x28.

    Retorna None se a imagem não tiver nenhum pixel branco.
    """

    # Localiza os pixels brancos, que representam
    # o número manuscrito.
    pontos = cv2.findNonZero(imagem_binaria)

    if pontos is None:
        return None

    # Calcula o menor retângulo que contém todos
    # os pixels brancos e recorta essa região.
    x, y, largura, altura = cv2.boundingRect(pontos)

    recorte = imagem_binaria[
        y:y + altura,
        x:x + largura
    ]

    # Cria uma imagem quadrada preta e coloca o
    # número no centro, preservando a proporção.
    lado = max(largura, altura)

    imagem_quadrada = np.zeros(
        (lado, lado),
        dtype=np.uint8
    )

    inicio_x = (lado - largura) // 2
    inicio_y = (lado - altura) // 2

    imagem_quadrada[
        inicio_y:inicio_y + altura,
        inicio_x:inicio_x + largura
    ] = recorte

    # Redimensiona o número para aproximadamente 20x20.
    # INTER_AREA ajuda na redução da imagem.
    numero_redimensionado = cv2.resize(
        imagem_quadrada,
        (TAMANHO_NUMERO, TAMANHO_NUMERO),
        interpolation=cv2.INTER_AREA
    )

    # Cria uma imagem completamente preta 28x28
    # e insere o número branco no centro.
    imagem_final = np.zeros(
        (TAMANHO_FINAL, TAMANHO_FINAL),
        dtype=np.uint8
    )

    posicao = (TAMANHO_FINAL - TAMANHO_NUMERO) // 2

    imagem_final[
        posicao:posicao + TAMANHO_NUMERO,
        posicao:posicao + TAMANHO_NUMERO
    ] = numero_redimensionado

    # O redimensionamento cria tons de cinza nas bordas;
    # volta para 0/255.
    return rebinarizar(imagem_final)


def preprocessar_digito(imagem_bgr_ou_path):
    """
    Transforma uma fotografia (caminho ou np.ndarray BGR/cinza)
    em uma imagem 28x28 uint8, fundo preto e número branco (0/255).

    Retorna None se a imagem não puder ser lida ou não tiver número.
    """

    imagem_cinza = carregar_imagem(imagem_bgr_ou_path)

    if imagem_cinza is None:
        return None

    imagem_binaria = binarizar_fotografia(imagem_cinza)

    return normalizar_binaria(imagem_binaria)


def _deslocar(imagem, deslocamento_x, deslocamento_y):
    """
    Desloca a imagem por um número inteiro de pixels.
    """

    altura, largura = imagem.shape

    matriz_deslocamento = np.float32([
        [1, 0, deslocamento_x],
        [0, 1, deslocamento_y]
    ])

    return cv2.warpAffine(
        imagem,
        matriz_deslocamento,
        (largura, altura),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


def aumentar_digito(
    imagem,
    quantidade=5,
    rng=None,
    angulo_maximo=ANGULO_MAXIMO,
    deslocamento_maximo=DESLOCAMENTO_MAXIMO,
    margem=MARGEM_AUMENTO
):
    """
    Gera variações leves de uma imagem 28x28 já processada.

    Para cada variação:
        1. Adiciona uma margem preta (padding) ao redor da imagem.
        2. Rotaciona levemente, sem cortar os traços.
        3. Rebinariza (0/255) e normaliza de volta para 28x28.
        4. Desloca alguns pixels (no máximo a margem de 4 px
           ao redor do número, então nunca corta os traços).
        5. Rebinariza novamente.

    Não altera brilho/contraste: a imagem é binária e deve continuar
    contendo apenas 0 e 255.
    """

    if rng is None:
        rng = random.Random(SEMENTE)

    # O deslocamento não pode ultrapassar a margem livre ao redor
    # do número, senão os traços seriam cortados.
    limite_deslocamento = (TAMANHO_FINAL - TAMANHO_NUMERO) // 2
    deslocamento_maximo = min(deslocamento_maximo, limite_deslocamento)

    imagens_aumentadas = []

    for _ in range(quantidade):

        angulo = rng.uniform(-angulo_maximo, angulo_maximo)
        deslocamento_x = rng.randint(-deslocamento_maximo, deslocamento_maximo)
        deslocamento_y = rng.randint(-deslocamento_maximo, deslocamento_maximo)

        # Adiciona borda preta para a rotação não cortar o número.
        expandida = cv2.copyMakeBorder(
            imagem,
            margem, margem, margem, margem,
            cv2.BORDER_CONSTANT,
            value=0
        )

        altura, largura = expandida.shape
        centro = (largura / 2, altura / 2)

        matriz_rotacao = cv2.getRotationMatrix2D(
            centro,
            angulo,
            1.0
        )

        rotacionada = cv2.warpAffine(
            expandida,
            matriz_rotacao,
            (largura, altura),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # A interpolação cria tons de cinza; volta para 0/255
        # e recoloca o número no padrão 20x20 dentro de 28x28.
        imagem_variada = normalizar_binaria(rebinarizar(rotacionada))

        if imagem_variada is None:
            continue

        imagem_variada = _deslocar(
            imagem_variada,
            deslocamento_x,
            deslocamento_y
        )

        imagens_aumentadas.append(rebinarizar(imagem_variada))

    return imagens_aumentadas
