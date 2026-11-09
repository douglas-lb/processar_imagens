import cv2
import numpy as np
import random

from pathlib import Path


# ============================================================
# CONFIGURAÇÕES DO PROJETO
# ============================================================

# Pasta onde estão as fotografias originais dos números.
PASTA_ENTRADA = Path("fotos")

# Pasta onde serão armazenadas as imagens processadas.
PASTA_SAIDA = Path("dataset")

# Tamanho final das imagens.
TAMANHO_FINAL = 28

# Tamanho aproximado que o número ocupará dentro da imagem.
TAMANHO_NUMERO = 20

# Quantidade de variações geradas para cada fotografia.
QUANTIDADE_AUMENTOS = 5

# Semente para tornar os resultados reproduzíveis.
SEMENTE = 42

random.seed(SEMENTE)
np.random.seed(SEMENTE)


# ============================================================
# CRIAÇÃO DAS PASTAS
# ============================================================

def criar_pastas():
    """
    Cria uma pasta para cada classe, de 0 até 9.
    """

    PASTA_SAIDA.mkdir(exist_ok=True)

    for numero in range(10):

        pasta = PASTA_SAIDA / str(numero)

        pasta.mkdir(exist_ok=True)


# ============================================================
# PROCESSAMENTO DA IMAGEM ORIGINAL
# ============================================================

def processar_imagem(caminho):
    """
    Recebe uma fotografia e transforma em uma imagem 28x28.

    O resultado terá:
        - Fundo preto.
        - Número branco.
        - Resolução de 28x28 pixels.
    """

    # Lê a fotografia.
    imagem = cv2.imread(str(caminho))

    if imagem is None:
        print(f"Não foi possível abrir: {caminho}")
        return None

    # --------------------------------------------------------
    # 1. CONVERSÃO PARA ESCALA DE CINZA
    # --------------------------------------------------------

    imagem_cinza = cv2.cvtColor(
        imagem,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # 2. REDUÇÃO DE RUÍDOS
    # --------------------------------------------------------

    # Aplica um desfoque leve para reduzir pequenas
    # imperfeições e ruídos da fotografia.
    imagem_suave = cv2.GaussianBlur(
        imagem_cinza,
        (5, 5),
        0
    )

    # --------------------------------------------------------
    # 3. BINARIZAÇÃO
    # --------------------------------------------------------

    # O método Otsu calcula automaticamente um limite
    # para separar o número do fundo.
    #
    # THRESH_BINARY_INV:
    #   - Número escuro vira branco (255).
    #   - Fundo claro vira preto (0).
    #
    # Isso deixa o número branco e o fundo preto.

    _, imagem_binaria = cv2.threshold(
        imagem_suave,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # --------------------------------------------------------
    # 4. IDENTIFICAÇÃO DA ÁREA DO NÚMERO
    # --------------------------------------------------------

    # Localiza os pixels brancos, que representam
    # o número manuscrito.
    pontos = cv2.findNonZero(imagem_binaria)

    # Se nenhum pixel branco for encontrado,
    # não existe conteúdo para processar.
    if pontos is None:
        return None

    # Calcula o menor retângulo que contém todos
    # os pixels brancos.
    x, y, largura, altura = cv2.boundingRect(pontos)

    # Recorta somente a região que contém o número.
    recorte = imagem_binaria[
        y:y + altura,
        x:x + largura
    ]

    # --------------------------------------------------------
    # 5. CENTRALIZAÇÃO DO NÚMERO
    # --------------------------------------------------------

    # Cria uma imagem quadrada preta.
    lado = max(largura, altura)

    imagem_quadrada = np.zeros(
        (lado, lado),
        dtype=np.uint8
    )

    # Calcula as posições para centralizar o recorte.
    inicio_x = (lado - largura) // 2
    inicio_y = (lado - altura) // 2

    # Coloca o número branco no centro do quadrado preto.
    imagem_quadrada[
        inicio_y:inicio_y + altura,
        inicio_x:inicio_x + largura
    ] = recorte

    # --------------------------------------------------------
    # 6. REDIMENSIONAMENTO
    # --------------------------------------------------------

    # Redimensiona o número para aproximadamente 20x20.
    # INTER_AREA ajuda na redução da imagem.
    numero_redimensionado = cv2.resize(
        imagem_quadrada,
        (TAMANHO_NUMERO, TAMANHO_NUMERO),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # 7. CRIAÇÃO DA IMAGEM FINAL 28x28
    # --------------------------------------------------------

    # Cria uma imagem completamente preta.
    imagem_final = np.zeros(
        (TAMANHO_FINAL, TAMANHO_FINAL),
        dtype=np.uint8
    )

    # Calcula as posições para centralizar o número.
    posicao = (TAMANHO_FINAL - TAMANHO_NUMERO) // 2

    # Insere o número branco no centro.
    imagem_final[
        posicao:posicao + TAMANHO_NUMERO,
        posicao:posicao + TAMANHO_NUMERO
    ] = numero_redimensionado

    return imagem_final


# ============================================================
# AUMENTO DE DADOS (DATA AUGMENTATION)
# ============================================================

def aumentar_dados(imagem):
    """
    Gera variações da imagem original.

    As alterações são leves para preservar
    a identidade do número manuscrito.
    """

    imagens_aumentadas = []

    altura, largura = imagem.shape

    for _ in range(QUANTIDADE_AUMENTOS):

        # ----------------------------------------------------
        # 1. ROTAÇÃO
        # ----------------------------------------------------

        # Rotaciona levemente a imagem.
        angulo = random.uniform(-12, 12)

        centro = (largura // 2, altura // 2)

        matriz_rotacao = cv2.getRotationMatrix2D(
            centro,
            angulo,
            1.0
        )

        imagem_variada = cv2.warpAffine(
            imagem,
            matriz_rotacao,
            (largura, altura),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # ----------------------------------------------------
        # 2. DESLOCAMENTO
        # ----------------------------------------------------

        # Desloca o número alguns pixels para os lados.
        deslocamento_x = random.randint(-2, 2)
        deslocamento_y = random.randint(-2, 2)

        matriz_deslocamento = np.float32([
            [1, 0, deslocamento_x],
            [0, 1, deslocamento_y]
        ])

        imagem_variada = cv2.warpAffine(
            imagem_variada,
            matriz_deslocamento,
            (largura, altura),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # ----------------------------------------------------
        # 3. CONTRASTE E BRILHO
        # ----------------------------------------------------

        # Modifica levemente o contraste e o brilho.
        alpha = random.uniform(0.85, 1.15)
        beta = random.randint(-10, 10)

        imagem_variada = cv2.convertScaleAbs(
            imagem_variada,
            alpha=alpha,
            beta=beta
        )

        # Armazena a imagem aumentada.
        imagens_aumentadas.append(imagem_variada)

    return imagens_aumentadas


# ============================================================
# SALVAMENTO DAS IMAGENS
# ============================================================

def salvar_imagem(imagem, numero, contador):
    """
    Salva a imagem na pasta correspondente ao número.

    Exemplo:
        dataset/0/0_00001.png
        dataset/1/1_00001.png
    """

    pasta = PASTA_SAIDA / str(numero)

    nome_arquivo = f"{numero}_{contador:05d}.png"

    caminho = pasta / nome_arquivo

    cv2.imwrite(str(caminho), imagem)


# ============================================================
# EXIBIÇÃO DA IMAGEM PARA CONFERÊNCIA
# ============================================================

def visualizar_imagem(imagem, nome):
    """
    Amplia a imagem para facilitar a visualização.

    A imagem original possui apenas 28x28 pixels.
    """

    imagem_ampliada = cv2.resize(
        imagem,
        (280, 280),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imshow(nome, imagem_ampliada)

    cv2.waitKey(1)


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def executar():

    criar_pastas()

    # Localiza todas as fotografias da pasta de entrada.
    extensoes = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

    arquivos = []

    for extensao in extensoes:
        arquivos.extend(
            PASTA_ENTRADA.glob(extensao)
        )

    # Ordena os arquivos para manter uma sequência.
    arquivos = sorted(arquivos)

    if not arquivos:
        print("Nenhuma fotografia encontrada!")
        return

    # Contador individual para cada classe.
    contadores = {
        numero: 0 for numero in range(10)
    }

    print("=" * 50)
    print("PROCESSAMENTO DO DATASET")
    print("=" * 50)

    print(f"Fotografias encontradas: {len(arquivos)}")

    print("\nDigite o número correspondente à imagem.")
    print("Digite S para pular a fotografia.")
    print("=" * 50)

    # Percorre todas as fotografias.
    for indice, caminho in enumerate(arquivos):

        print("\n" + "=" * 50)

        print(
            f"Imagem {indice + 1}/{len(arquivos)}: "
            f"{caminho.name}"
        )

        # Processa a fotografia.
        imagem = processar_imagem(caminho)

        if imagem is None:
            print("Não foi possível processar essa imagem.")
            continue

        # Mostra a imagem processada.
        visualizar_imagem(
            imagem,
            "Imagem processada - Fundo preto"
        )

        # Solicita ao usuário o número correspondente.
        while True:

            resposta = input(
                "Qual número está na imagem (0-9 ou S)? "
            ).strip().lower()

            if resposta == "s":
                print("Imagem ignorada.")
                break

            if resposta in [str(i) for i in range(10)]:
                numero = int(resposta)
                break

            print("Digite um número entre 0 e 9 ou S.")

        # Se o usuário escolheu pular, continua.
        if resposta == "s":
            cv2.destroyAllWindows()
            continue

        # ----------------------------------------------------
        # SALVA A IMAGEM ORIGINAL PROCESSADA
        # ----------------------------------------------------

        contadores[numero] += 1

        salvar_imagem(
            imagem,
            numero,
            contadores[numero]
        )

        # ----------------------------------------------------
        # GERA E SALVA AS VARIAÇÕES
        # ----------------------------------------------------

        imagens_aumentadas = aumentar_dados(imagem)

        for imagem_aumentada in imagens_aumentadas:

            contadores[numero] += 1

            salvar_imagem(
                imagem_aumentada,
                numero,
                contadores[numero]
            )

        print(
            f"Imagem salva na classe {numero}."
        )

        print(
            f"Total de imagens geradas: "
            f"{QUANTIDADE_AUMENTOS + 1}"
        )

        cv2.destroyAllWindows()

    # --------------------------------------------------------
    # RESULTADO FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 50)
    print("PROCESSAMENTO CONCLUÍDO!")
    print("=" * 50)

    print("\nQuantidade de imagens por classe:")

    for numero in range(10):

        print(
            f"Número {numero}: "
            f"{contadores[numero]} imagens"
        )

    total = sum(contadores.values())

    print(f"\nTotal de imagens: {total}")

    cv2.destroyAllWindows()


# ============================================================
# INICIALIZAÇÃO DO PROGRAMA
# ============================================================

if __name__ == "__main__":
    executar()