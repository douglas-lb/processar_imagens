"""
Reconhecimento de dígitos manuscritos em tempo real pela webcam.

Uso:

    python camera_webcam.py
    python camera_webcam.py --camera 1          # outra webcam
    python camera_webcam.py --limiar 0.80       # confiança mínima para exibir o dígito
    python camera_webcam.py --tflite            # força modelos/digitos.tflite
    python camera_webcam.py --keras             # força modelos/digitos.keras
    python camera_webcam.py --sem-espelho       # inicia sem espelhamento horizontal

Como usar:

    - Escreva o número com caneta escura em papel claro (evite tela de celular).
    - Coloque o dígito INTEIRO dentro do retângulo verde.
    - A imagem começa espelhada; pressione h para ligar/desligar o espelho.
    - Teclas:
        q / Esc  = sair
        s        = salvar quadro + 28x28 em capturas/
        h        = espelho horizontal on/off
        + / =    = aumentar o quadro
        - / _    = diminuir o quadro

Sem opção de modelo, usa o TFLite (mais rápido) se existir e, caso contrário,
o modelo Keras.

Observação: os textos desenhados na janela não usam acentos porque as fontes
do OpenCV não os suportam; as mensagens do terminal usam acentos normalmente.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import sys
import time

from pathlib import Path

import cv2
import numpy as np

from prever import ModeloDigitos
from preprocess import (
    binarizar_adaptativo_camera,
    carregar_imagem,
    detectar_regiao_papel,
    normalizar_binaria,
)
from treinar_modelo import CAMINHO_KERAS, CAMINHO_TFLITE


# Confiança mínima padrão para mostrar o dígito.
LIMIAR_PADRAO = 0.80

# Pasta onde as capturas (tecla "s") são salvas.
PASTA_CAPTURAS = Path(__file__).resolve().parent / "capturas"

# Lado do ROI como fração do menor lado do quadro (ajustável com + / -).
FRACAO_ROI_PADRAO = 0.75
FRACAO_ROI_MIN = 0.40
FRACAO_ROI_MAX = 0.92
PASSO_ROI = 0.05

# Margem ao redor do contorno do dígito (fração do lado do bbox).
MARGEM_CONTORNO = 0.15

# Se altura/largura do blob for maior que isto, provavelmente só o "haste".
RAZAO_HASTE = 2.6

# Área do contorno em relação à área do papel.
FRACAO_AREA_MIN = 0.004
FRACAO_AREA_MAX = 0.45

# Fração de tinta (brancos) na binária do papel.
TINTA_PAPEL_MIN = 0.008
TINTA_PAPEL_MAX = 0.35

# Brilho médio do papel (região clara detectada).
BRILHO_PAPEL_MIN = 130.0

# Fração de brancos na imagem 28x28 final (padrão MNIST).
TINTA_28_MIN = 0.045
TINTA_28_MAX = 0.35

# Quantos frames seguidos com o mesmo dígito antes de aceitar.
FRAMES_ESTAVEIS = 8

# Janela e miniatura da entrada do modelo.
TITULO_JANELA = "Reconhecimento de digitos"
LADO_MINIATURA = 112

FONTE = cv2.FONT_HERSHEY_SIMPLEX

VERDE = (0, 200, 0)
AMARELO = (0, 215, 255)
VERMELHO = (0, 0, 220)
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)


def ler_argumentos():

    parser = argparse.ArgumentParser(
        description="Reconhece dígitos (0-9) escritos à mão, em tempo real, pela webcam."
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="índice da câmera (padrão: 0)",
    )

    parser.add_argument(
        "--limiar",
        type=float,
        default=LIMIAR_PADRAO,
        help=f"confiança mínima de 0 a 1 para exibir o dígito (padrão: {LIMIAR_PADRAO:.2f})",
    )

    grupo = parser.add_mutually_exclusive_group()

    grupo.add_argument(
        "--tflite",
        action="store_true",
        help="força o modelo TFLite (modelos/digitos.tflite)",
    )

    grupo.add_argument(
        "--keras",
        action="store_true",
        help="força o modelo Keras (modelos/digitos.keras)",
    )

    parser.add_argument(
        "--modelo",
        type=Path,
        help="caminho de um arquivo .keras ou .tflite específico",
    )

    parser.add_argument(
        "--capturas",
        type=Path,
        default=PASTA_CAPTURAS,
        help="pasta onde salvar as capturas (padrão: capturas/)",
    )

    parser.add_argument(
        "--sem-espelho",
        action="store_true",
        help="inicia sem espelhar a imagem horizontalmente (padrão: espelhada)",
    )

    parser.add_argument(
        "--roi",
        type=float,
        default=FRACAO_ROI_PADRAO,
        help=f"tamanho inicial do quadro (0.40–0.92, padrão: {FRACAO_ROI_PADRAO:.2f})",
    )

    argumentos = parser.parse_args()

    if not 0.0 <= argumentos.limiar <= 1.0:
        parser.error("--limiar deve estar entre 0 e 1 (ex.: 0.70).")

    if not FRACAO_ROI_MIN <= argumentos.roi <= FRACAO_ROI_MAX:
        parser.error(
            f"--roi deve estar entre {FRACAO_ROI_MIN} e {FRACAO_ROI_MAX}."
        )

    return argumentos


def carregar_modelo(argumentos):
    """
    Escolhe o modelo: --modelo > --keras > --tflite > TFLite se existir > Keras.
    """

    if argumentos.modelo:
        return ModeloDigitos(argumentos.modelo)

    if argumentos.keras:
        return ModeloDigitos(CAMINHO_KERAS)

    if argumentos.tflite or CAMINHO_TFLITE.exists():
        return ModeloDigitos(CAMINHO_TFLITE)

    return ModeloDigitos(CAMINHO_KERAS)


def abrir_camera(indice):
    """
    Abre a câmera e confirma que ela entrega quadros.
    Retorna None se não for possível usá-la.
    """

    captura = cv2.VideoCapture(indice)

    if not captura.isOpened():
        captura.release()
        return None

    ok, quadro = captura.read()

    if not ok or quadro is None:
        captura.release()
        return None

    return captura


def calcular_roi(altura, largura, fracao):
    """
    Retângulo quadrado centralizado: (x1, y1, x2, y2).
    """

    lado = int(min(altura, largura) * fracao)
    lado = max(lado, 40)
    x1 = (largura - lado) // 2
    y1 = (altura - lado) // 2

    return x1, y1, x1 + lado, y1 + lado


def escrever(imagem, texto, posicao, escala, cor, espessura=2):
    """
    Texto com contorno preto para ficar legível em qualquer fundo.
    """

    cv2.putText(imagem, texto, posicao, FONTE, escala, PRETO, espessura + 3, cv2.LINE_AA)
    cv2.putText(imagem, texto, posicao, FONTE, escala, cor, espessura, cv2.LINE_AA)


def parece_digito_28x28(imagem_28x28):
    """
    Heurística sobre a entrada do modelo (padrão MNIST).
    """

    tinta = float(np.count_nonzero(imagem_28x28)) / float(imagem_28x28.size)

    if tinta < TINTA_28_MIN or tinta > TINTA_28_MAX:
        return False

    _, rotulos = cv2.connectedComponents((imagem_28x28 > 0).astype(np.uint8))
    componentes = int(rotulos.max())

    if componentes < 1 or componentes > 6:
        return False

    return True


def pontuar_contorno_digito(contorno, area_papel, centro_papel):
    """
    Pontua um contorno: área razoável, aspecto e centralidade no papel.
    Retorna (score, incompleto) ou None se inválido.
    """

    area = float(cv2.contourArea(contorno))
    fracao = area / max(area_papel, 1.0)

    if fracao < FRACAO_AREA_MIN or fracao > FRACAO_AREA_MAX:
        return None

    x, y, w, h = cv2.boundingRect(contorno)

    if w < 4 or h < 4:
        return None

    razao = max(h / max(w, 1), w / max(h, 1))
    incompleto = razao >= RAZAO_HASTE

    cx = x + w / 2.0
    cy = y + h / 2.0
    dx = (cx - centro_papel[0]) / max(centro_papel[0], 1.0)
    dy = (cy - centro_papel[1]) / max(centro_papel[1], 1.0)
    centralidade = 1.0 / (1.0 + dx * dx + dy * dy)

    # Prefere área intermediária (nem poeira, nem mancha enorme).
    alvo = 0.08
    score_area = 1.0 - min(abs(fracao - alvo) / 0.20, 1.0)
    score_aspecto = 1.0 if razao < 2.0 else (0.35 if not incompleto else 0.05)

    score = 0.45 * score_area + 0.35 * centralidade + 0.20 * score_aspecto

    return score, incompleto, (x, y, w, h)


def extrair_digito_do_papel(cinza_roi):
    """
    Detecta o papel, binariza com limiar adaptativo e escolhe o contorno do dígito.

    Retorna (binaria_digito_ou_None, incompleto, motivo).
    """

    regiao = detectar_regiao_papel(cinza_roi)

    if regiao is None:
        return None, False, "sem papel - coloque a folha no quadro"

    px, py, pw, ph = regiao
    papel = cinza_roi[py:py + ph, px:px + pw]

    if float(papel.mean()) < BRILHO_PAPEL_MIN:
        return None, False, "papel escuro - use folha clara"

    binaria = binarizar_adaptativo_camera(papel)

    if binaria is None:
        return None, False, "nenhum digito"

    tinta = float(np.count_nonzero(binaria)) / float(binaria.size)

    if tinta < TINTA_PAPEL_MIN:
        return None, False, "nenhum digito - escreva o numero"

    if tinta > TINTA_PAPEL_MAX:
        return None, False, "muita tinta - limpe o quadro / luz"

    contornos, _ = cv2.findContours(
        binaria,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contornos:
        return None, False, "nenhum digito"

    area_papel = float(ph * pw)
    centro = (pw / 2.0, ph / 2.0)

    melhor = None
    melhor_score = -1.0

    for contorno in contornos:
        resultado = pontuar_contorno_digito(contorno, area_papel, centro)

        if resultado is None:
            continue

        score, incompleto, bbox = resultado

        if score > melhor_score:
            melhor_score = score
            melhor = (incompleto, bbox)

    if melhor is None:
        return None, False, "nenhum digito - aproxime o numero"

    incompleto, (x, y, w, h) = melhor

    margem = int(max(w, h) * MARGEM_CONTORNO) + 4
    x1 = max(0, x - margem)
    y1 = max(0, y - margem)
    x2 = min(pw, x + w + margem)
    y2 = min(ph, y + h + margem)

    digito_bin = binaria[y1:y2, x1:x2].copy()

    if digito_bin.size == 0 or not np.any(digito_bin):
        return None, False, "nenhum digito"

    return digito_bin, incompleto, ""


def reconhecer(modelo, recorte):
    """
    Pré-processa o ROI da câmera e faz a previsão.

    Retorna:
      imagem_28x28, digito, confianca, incompleto, motivo_rejeicao
    """

    cinza = carregar_imagem(recorte)

    if cinza is None:
        return None, None, None, False, "nenhum digito"

    digito_bin, incompleto, motivo = extrair_digito_do_papel(cinza)

    if digito_bin is None:
        return None, None, None, False, motivo or "nenhum digito"

    if incompleto:
        imagem_parcial = normalizar_binaria(digito_bin)
        return (
            imagem_parcial,
            None,
            None,
            True,
            "encaixe o digito INTEIRO no papel",
        )

    imagem_28x28 = normalizar_binaria(digito_bin)

    if imagem_28x28 is None:
        return None, None, None, False, "nenhum digito"

    if not parece_digito_28x28(imagem_28x28):
        return (
            imagem_28x28,
            None,
            None,
            False,
            "aproxime o digito inteiro no papel dentro do verde",
        )

    digito, confianca, _ = modelo.prever(imagem_28x28)

    return imagem_28x28, digito, confianca, False, ""


def desenhar_interface(
    quadro,
    roi,
    imagem_28x28,
    digito,
    confianca,
    limiar,
    fps,
    espelhar,
    fracao_roi,
    incompleto,
    motivo_rejeicao,
    estavel,
):
    """
    Desenha o ROI, o resultado e a miniatura da entrada do modelo.
    """

    x1, y1, x2, y2 = roi

    # Não mostra "candidato" fraco (ex.: 4 a 21%) — só aceita ou pede ajuste.
    if motivo_rejeicao:
        cor = VERMELHO if not incompleto else AMARELO
        mensagem = motivo_rejeicao
        aceitar = False
    elif imagem_28x28 is None or digito is None:
        cor = VERMELHO
        mensagem = "nenhum digito - mostre o numero no papel"
        aceitar = False
    elif confianca is not None and confianca >= limiar and estavel:
        cor = VERDE
        mensagem = f"digito {digito} ({confianca * 100:.0f}%)"
        aceitar = True
    elif confianca is not None and confianca >= limiar and not estavel:
        cor = AMARELO
        mensagem = "segure firme..."
        aceitar = False
    else:
        cor = AMARELO
        mensagem = "aproxime o digito inteiro no papel dentro do verde"
        aceitar = False

    cv2.rectangle(quadro, (x1, y1), (x2, y2), cor, 3)

    escrever(
        quadro,
        "papel claro | digito inteiro no verde | h: espelho",
        (x1, max(18, y1 - 12)),
        0.55,
        BRANCO,
        1,
    )

    if aceitar:
        escrever(quadro, mensagem, (x1, y2 + 40), 1.2, cor, 3)
    else:
        escrever(quadro, mensagem, (x1, y2 + 35), 0.70, cor, 2)

    # Miniatura do que o modelo realmente vê (canto superior direito).
    if imagem_28x28 is not None:
        miniatura = cv2.resize(
            imagem_28x28,
            (LADO_MINIATURA, LADO_MINIATURA),
            interpolation=cv2.INTER_NEAREST,
        )
        miniatura = cv2.cvtColor(miniatura, cv2.COLOR_GRAY2BGR)

        altura, largura = quadro.shape[:2]
        mx = largura - LADO_MINIATURA - 10
        my = 10

        quadro[my:my + LADO_MINIATURA, mx:mx + LADO_MINIATURA] = miniatura
        cv2.rectangle(
            quadro,
            (mx, my),
            (mx + LADO_MINIATURA, my + LADO_MINIATURA),
            BRANCO,
            1,
        )
        escrever(
            quadro,
            "entrada 28x28",
            (mx, my + LADO_MINIATURA + 18),
            0.5,
            BRANCO,
            1,
        )

    espelho_txt = "on" if espelhar else "off"
    escrever(
        quadro,
        f"{fps:.0f} fps | espelho: {espelho_txt} | roi: {fracao_roi:.0%}",
        (10, 25),
        0.55,
        BRANCO,
        1,
    )
    escrever(
        quadro,
        "q:sair  s:salvar  h:espelho  +/-:quadro",
        (10, 48),
        0.5,
        BRANCO,
        1,
    )


def salvar_captura(pasta, quadro_original, imagem_28x28):
    """
    Salva o quadro (sem desenhos) e, se houver, a imagem 28x28 pré-processada.
    """

    pasta.mkdir(parents=True, exist_ok=True)

    carimbo = time.strftime("%Y%m%d_%H%M%S")
    contador = 0
    base = pasta / f"captura_{carimbo}"

    while base.with_name(base.name + "_quadro.png").exists():
        contador += 1
        base = pasta / f"captura_{carimbo}_{contador}"

    caminho_quadro = base.with_name(base.name + "_quadro.png")
    cv2.imwrite(str(caminho_quadro), quadro_original)

    print(f"Quadro salvo em {caminho_quadro}")

    if imagem_28x28 is not None:
        caminho_28 = base.with_name(base.name + "_28x28.png")
        cv2.imwrite(str(caminho_28), imagem_28x28)
        print(f"Imagem 28x28 salva em {caminho_28}")


def executar(argumentos):

    try:
        modelo = carregar_modelo(argumentos)
    except FileNotFoundError as erro:
        print(f"Erro: {erro}")
        return 1

    print(f"Modelo: {modelo.caminho.name} | limiar: {argumentos.limiar:.2f}")

    captura = abrir_camera(argumentos.camera)

    if captura is None:
        print(
            f"Erro: não foi possível abrir a câmera {argumentos.camera}.\n"
            "Verifique se a webcam está conectada, se não está em uso por outro "
            "programa e se você tem permissão de acesso (no Linux, o usuário "
            "precisa estar no grupo 'video').\n"
            "Para usar outra câmera, rode por exemplo: python camera_webcam.py --camera 1"
        )
        return 1

    # Espelho ligado por padrão (desliga só com --sem-espelho ou tecla h).
    espelhar = not argumentos.sem_espelho
    fracao_roi = float(argumentos.roi)

    print(
        "Câmera aberta.\n"
        "  q = sair | s = salvar | h = espelho | + / - = tamanho do quadro\n"
        f"  Espelho inicial: {'ligado' if espelhar else 'desligado'} | "
        f"ROI: {fracao_roi:.0%}\n"
        "  Dica: folha clara no verde, dígito inteiro visível; confira a miniatura 28x28."
    )

    fps = 0.0
    instante_anterior = time.perf_counter()
    janela_criada = False
    digito_candidato = None
    frames_iguais = 0

    try:
        while True:
            ok, quadro = captura.read()

            if not ok or quadro is None:
                print("Erro: a câmera parou de enviar imagens.")
                return 1

            if espelhar:
                quadro = cv2.flip(quadro, 1)

            original = quadro.copy()
            altura, largura = quadro.shape[:2]
            roi = calcular_roi(altura, largura, fracao_roi)
            x1, y1, x2, y2 = roi

            imagem_28x28, digito, confianca, incompleto, motivo = reconhecer(
                modelo, original[y1:y2, x1:x2]
            )

            if (
                digito is not None
                and confianca is not None
                and confianca >= argumentos.limiar
                and not incompleto
                and not motivo
            ):
                if digito == digito_candidato:
                    frames_iguais += 1
                else:
                    digito_candidato = digito
                    frames_iguais = 1
            else:
                digito_candidato = None
                frames_iguais = 0

            estavel = frames_iguais >= FRAMES_ESTAVEIS

            agora = time.perf_counter()
            fps_atual = 1.0 / max(agora - instante_anterior, 1e-6)
            fps = 0.9 * fps + 0.1 * fps_atual if fps else fps_atual
            instante_anterior = agora

            desenhar_interface(
                quadro,
                roi,
                imagem_28x28,
                digito,
                confianca,
                argumentos.limiar,
                fps,
                espelhar,
                fracao_roi,
                incompleto,
                motivo,
                estavel,
            )

            try:
                cv2.imshow(TITULO_JANELA, quadro)
                janela_criada = True
                tecla = cv2.waitKey(1) & 0xFF
            except cv2.error:
                print(
                    "Erro: não foi possível abrir a janela de vídeo "
                    "(ambiente sem interface gráfica?). Rode em um desktop com display."
                )
                return 1

            if tecla in (ord("q"), ord("Q"), 27):
                break

            if tecla in (ord("s"), ord("S")):
                salvar_captura(argumentos.capturas, original, imagem_28x28)

            if tecla in (ord("h"), ord("H")):
                espelhar = not espelhar
                print(f"Espelho horizontal: {'ligado' if espelhar else 'desligado'}")

            if tecla in (ord("+"), ord("=")):
                fracao_roi = min(FRACAO_ROI_MAX, fracao_roi + PASSO_ROI)
                print(f"ROI: {fracao_roi:.0%}")

            if tecla in (ord("-"), ord("_")):
                fracao_roi = max(FRACAO_ROI_MIN, fracao_roi - PASSO_ROI)
                print(f"ROI: {fracao_roi:.0%}")

            if cv2.getWindowProperty(TITULO_JANELA, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("\nEncerrado pelo usuário.")

    finally:
        captura.release()

        if janela_criada:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(executar(ler_argumentos()))
