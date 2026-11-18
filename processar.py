import argparse
import csv
import json
import random
import re

from pathlib import Path

import cv2
import numpy as np

from preprocess import (
    SEMENTE,
    TAMANHO_FINAL,
    aumentar_digito,
    preprocessar_digito,
)


# Pasta do projeto (permite executar o script de qualquer diretório).
PASTA_PROJETO = Path(__file__).resolve().parent

# Pasta onde estão as fotografias originais dos números.
PASTA_ENTRADA = PASTA_PROJETO / "fotos"

# Pasta onde serão armazenadas as imagens processadas.
PASTA_SAIDA = PASTA_PROJETO / "dataset"

# Arquivos gerados junto com o dataset.
ARQUIVO_METADATA = PASTA_SAIDA / "metadata.csv"
ARQUIVO_SPLITS = PASTA_SAIDA / "splits.json"

COLUNAS_METADATA = ["arquivo", "classe", "source_id", "eh_aumento"]

# Extensões aceitas (comparadas sem diferenciar maiúsculas/minúsculas).
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp"}

# Quantidade de variações geradas para cada fotografia.
QUANTIDADE_AUMENTOS = 5

# Proporções da divisão treino / validação / teste.
PROPORCAO_VALIDACAO = 0.1
PROPORCAO_TESTE = 0.1

# Número mínimo de fotografias de uma classe para que ela
# tenha pelo menos uma fotografia na validação e no teste.
MINIMO_FOTOS_PARA_DIVIDIR = 5

# Prefixo do nome do arquivo que indica o rótulo.
# Exemplos: "7_foto.jpg", "7-xxx.png", "7.jpg".
PADRAO_PREFIXO = re.compile(r"^([0-9])(?:[_\-\s.]|$)")

random.seed(SEMENTE)
np.random.seed(SEMENTE)

# Gerador próprio do aumento de dados (reprodutível).
GERADOR_AUMENTO = random.Random(SEMENTE)


def criar_pastas():
    """
    Cria uma pasta para cada classe, de 0 até 9.
    """

    PASTA_SAIDA.mkdir(exist_ok=True)

    for numero in range(10):

        pasta = PASTA_SAIDA / str(numero)

        pasta.mkdir(exist_ok=True)


def processar_imagem(caminho):
    """
    Recebe uma fotografia e transforma em uma imagem 28x28.

    O resultado terá:
        - Fundo preto.
        - Número branco.
        - Resolução de 28x28 pixels.
        - Somente os valores 0 e 255.
    """

    imagem = preprocessar_digito(caminho)

    if imagem is None:
        print(f"Não foi possível abrir ou encontrar o número em: {caminho}")

    return imagem


def aumentar_dados(imagem):
    """
    Gera variações da imagem original.

    As alterações são leves para preservar
    a identidade do número manuscrito.
    """

    return aumentar_digito(
        imagem,
        quantidade=QUANTIDADE_AUMENTOS,
        rng=GERADOR_AUMENTO
    )


def listar_fotografias():
    """
    Localiza as fotografias da pasta de entrada, aceitando
    extensões em maiúsculas ou minúsculas (.jpg, .JPG, .Png...).
    """

    if not PASTA_ENTRADA.exists():
        return []

    arquivos = [
        caminho
        for caminho in PASTA_ENTRADA.iterdir()
        if caminho.is_file() and caminho.suffix.lower() in EXTENSOES
    ]

    # Ordena os arquivos para manter uma sequência.
    return sorted(arquivos, key=lambda caminho: caminho.name.lower())


def sugerir_rotulo(caminho):
    """
    Sugere o rótulo a partir do prefixo do nome do arquivo.

    Exemplos:
        7_foto.jpg -> 7
        3-teste.png -> 3
        foto.jpg -> None
    """

    correspondencia = PADRAO_PREFIXO.match(Path(caminho).name)

    if correspondencia is None:
        return None

    return int(correspondencia.group(1))


def perguntar_rotulo(sugestao):
    """
    Pergunta ao usuário o número da imagem.

    Retorna o número (0-9) ou None se o usuário pular a imagem.
    """

    if sugestao is not None:
        print(f"Sugestão pelo nome do arquivo: {sugestao}")
        pergunta = (
            f"Enter confirma {sugestao}, digite 0-9 para corrigir "
            f"ou S para pular: "
        )
    else:
        pergunta = "Qual número está na imagem (0-9 ou S)? "

    while True:

        resposta = input(pergunta).strip().lower()

        if resposta == "" and sugestao is not None:
            return sugestao

        if resposta == "s":
            return None

        if resposta in [str(i) for i in range(10)]:
            return int(resposta)

        print("Digite um número entre 0 e 9 ou S.")


def proximo_contador(numero):
    """
    Retorna o maior índice já usado na pasta da classe, para que
    novas imagens nunca sobrescrevam imagens existentes.
    """

    padrao = re.compile(rf"^{numero}_(\d+)\.png$")

    maior = 0

    for caminho in (PASTA_SAIDA / str(numero)).glob("*.png"):

        correspondencia = padrao.match(caminho.name)

        if correspondencia:
            maior = max(maior, int(correspondencia.group(1)))

    return maior


def salvar_imagem(imagem, numero, contador):
    """
    Salva a imagem na pasta correspondente ao número.

    Exemplo:
        dataset/0/0_00001.png
        dataset/1/1_00001.png

    Retorna o caminho relativo à pasta do dataset (ex.: "0/0_00001.png").
    """

    pasta = PASTA_SAIDA / str(numero)

    nome_arquivo = f"{numero}_{contador:05d}.png"

    caminho = pasta / nome_arquivo

    cv2.imwrite(str(caminho), imagem)

    return f"{numero}/{nome_arquivo}"


def carregar_metadata():
    """
    Lê o dataset/metadata.csv, se existir.
    """

    if not ARQUIVO_METADATA.exists():
        return []

    with open(ARQUIVO_METADATA, newline="", encoding="utf-8") as arquivo:
        return list(csv.DictReader(arquivo))


def registrar_metadata(linhas):
    """
    Acrescenta linhas ao dataset/metadata.csv.

    Cada linha: arquivo, classe, source_id, eh_aumento.
    """

    novo = not ARQUIVO_METADATA.exists()

    with open(ARQUIVO_METADATA, "a", newline="", encoding="utf-8") as arquivo:

        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS_METADATA)

        if novo:
            escritor.writeheader()

        escritor.writerows(linhas)


def gerar_splits():
    """
    Divide o dataset em treino (80%), validação (10%) e teste (10%)
    POR source_id: todas as imagens derivadas da mesma fotografia
    (original + aumentos) ficam sempre no mesmo conjunto.

    A divisão é feita separadamente em cada classe (estratificada).
    Classes com menos de MINIMO_FOTOS_PARA_DIVIDIR fotografias ficam
    inteiramente no treino.
    """

    linhas = carregar_metadata()

    if not linhas:
        print("metadata.csv vazio ou inexistente; splits.json não foi gerado.")
        return None

    # Agrupa as fotografias originais por classe.
    fontes_por_classe = {}
    arquivos_por_fonte = {}

    for linha in linhas:

        source_id = linha["source_id"]

        fontes_por_classe.setdefault(int(linha["classe"]), set()).add(source_id)
        arquivos_por_fonte.setdefault(source_id, []).append(linha["arquivo"])

    gerador = random.Random(SEMENTE)

    fontes = {"treino": [], "validacao": [], "teste": []}

    for classe in sorted(fontes_por_classe):

        ids = sorted(fontes_por_classe[classe])
        gerador.shuffle(ids)

        quantidade = len(ids)

        quantidade_teste = int(quantidade * PROPORCAO_TESTE + 0.5)
        quantidade_validacao = int(quantidade * PROPORCAO_VALIDACAO + 0.5)

        if quantidade >= MINIMO_FOTOS_PARA_DIVIDIR:
            quantidade_teste = max(1, quantidade_teste)
            quantidade_validacao = max(1, quantidade_validacao)
        else:
            quantidade_teste = 0
            quantidade_validacao = 0

        fontes["teste"].extend(ids[:quantidade_teste])
        fontes["validacao"].extend(
            ids[quantidade_teste:quantidade_teste + quantidade_validacao]
        )
        fontes["treino"].extend(ids[quantidade_teste + quantidade_validacao:])

    splits = {
        "semente": SEMENTE,
        "estrategia": "por_source_id_estratificado_por_classe",
        "proporcoes": {
            "treino": 1 - PROPORCAO_VALIDACAO - PROPORCAO_TESTE,
            "validacao": PROPORCAO_VALIDACAO,
            "teste": PROPORCAO_TESTE,
        },
        "source_ids": {
            nome: sorted(ids) for nome, ids in fontes.items()
        },
    }

    for nome, ids in fontes.items():
        splits[nome] = sorted(
            arquivo
            for source_id in ids
            for arquivo in arquivos_por_fonte[source_id]
        )

    with open(ARQUIVO_SPLITS, "w", encoding="utf-8") as arquivo:
        json.dump(splits, arquivo, ensure_ascii=False, indent=2)

    print("\nDivisão salva em dataset/splits.json (por source_id):")

    for nome in ("treino", "validacao", "teste"):
        print(
            f"  {nome}: {len(fontes[nome])} fotografias, "
            f"{len(splits[nome])} imagens"
        )

    return splits


def limpar_dataset():
    """
    Remove as imagens geradas, o metadata.csv e o splits.json.
    """

    for numero in range(10):
        for caminho in (PASTA_SAIDA / str(numero)).glob("*.png"):
            caminho.unlink()

    for caminho in (ARQUIVO_METADATA, ARQUIVO_SPLITS):
        if caminho.exists():
            caminho.unlink()


def avisar_imagens_sem_metadata(linhas):
    """
    Avisa sobre imagens antigas que não estão no metadata.csv
    (por exemplo, geradas por versões anteriores do script).
    Elas não entram no splits.json.
    """

    registradas = {linha["arquivo"] for linha in linhas}

    orfas = [
        caminho
        for numero in range(10)
        for caminho in (PASTA_SAIDA / str(numero)).glob("*.png")
        if f"{numero}/{caminho.name}" not in registradas
    ]

    if orfas:
        print(
            f"\nAVISO: {len(orfas)} imagens em dataset/ não estão no "
            f"metadata.csv e serão ignoradas pelo splits.json."
        )
        print("Use 'python processar.py --refazer' para gerar tudo de novo.")


def visualizar_imagem(imagem, nome):
    """
    Amplia a imagem para facilitar a visualização.

    A imagem original possui apenas 28x28 pixels.
    """

    imagem_ampliada = cv2.resize(
        imagem,
        (TAMANHO_FINAL * 10, TAMANHO_FINAL * 10),
        interpolation=cv2.INTER_NEAREST
    )

    try:
        cv2.imshow(nome, imagem_ampliada)
        cv2.waitKey(1)
    except cv2.error:
        print("(Não foi possível abrir a janela de visualização.)")


def fechar_janelas():
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def executar():

    criar_pastas()

    arquivos = listar_fotografias()

    if not arquivos:
        print("Nenhuma fotografia encontrada!")
        return

    # Fotografias já rotuladas em execuções anteriores são puladas.
    metadata = carregar_metadata()
    ja_processadas = {linha["source_id"] for linha in metadata}

    avisar_imagens_sem_metadata(metadata)

    pendentes = [
        caminho for caminho in arquivos
        if caminho.name not in ja_processadas
    ]

    # Contador individual para cada classe (continua de onde parou).
    contadores = {
        numero: proximo_contador(numero) for numero in range(10)
    }

    geradas = {numero: 0 for numero in range(10)}

    print("=" * 50)
    print("PROCESSAMENTO DO DATASET")
    print("=" * 50)

    print(f"Fotografias encontradas: {len(arquivos)}")
    print(f"Já processadas anteriormente: {len(arquivos) - len(pendentes)}")
    print(f"Pendentes: {len(pendentes)}")

    print("\nDigite o número correspondente à imagem.")
    print("Se o nome começar com o número (ex.: 7_foto.jpg),")
    print("pressione Enter para aceitar a sugestão.")
    print("Digite S para pular a fotografia.")
    print("=" * 50)

    try:
        # Percorre todas as fotografias pendentes.
        for indice, caminho in enumerate(pendentes):

            print("\n" + "=" * 50)

            print(
                f"Imagem {indice + 1}/{len(pendentes)}: "
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
            numero = perguntar_rotulo(sugerir_rotulo(caminho))

            fechar_janelas()

            if numero is None:
                print("Imagem ignorada.")
                continue

            source_id = caminho.name

            # salva a imagem original e as variações
            imagens = [imagem] + aumentar_dados(imagem)

            linhas = []

            for posicao, imagem_salvar in enumerate(imagens):

                contadores[numero] += 1

                arquivo = salvar_imagem(
                    imagem_salvar,
                    numero,
                    contadores[numero]
                )

                linhas.append({
                    "arquivo": arquivo,
                    "classe": numero,
                    "source_id": source_id,
                    "eh_aumento": int(posicao > 0),
                })

            # Grava a cada fotografia para não perder rótulos
            # caso o programa seja interrompido.
            registrar_metadata(linhas)

            geradas[numero] += len(imagens)

            print(f"Imagem salva na classe {numero}.")
            print(f"Total de imagens geradas: {len(imagens)}")

    except KeyboardInterrupt:
        print("\n\nProcessamento interrompido pelo usuário.")

    finally:
        fechar_janelas()

    print("\n" + "=" * 50)
    print("PROCESSAMENTO CONCLUÍDO!")
    print("=" * 50)

    print("\nImagens geradas nesta execução, por classe:")

    for numero in range(10):

        print(
            f"Número {numero}: "
            f"{geradas[numero]} imagens"
        )

    print(f"\nTotal de imagens nesta execução: {sum(geradas.values())}")

    gerar_splits()


def ler_argumentos():

    parser = argparse.ArgumentParser(
        description="Cria um dataset de dígitos 28x28 a partir de fotografias."
    )

    parser.add_argument(
        "--refazer",
        action="store_true",
        help="apaga as imagens, o metadata.csv e o splits.json e processa tudo de novo",
    )

    parser.add_argument(
        "--apenas-splits",
        action="store_true",
        help="somente gera novamente o dataset/splits.json a partir do metadata.csv",
    )

    return parser.parse_args()


if __name__ == "__main__":

    argumentos = ler_argumentos()

    if argumentos.apenas_splits:
        gerar_splits()

    else:
        if argumentos.refazer:
            confirmacao = input(
                "Apagar todas as imagens de dataset/ e recomeçar? (s/n) "
            ).strip().lower()

            if confirmacao != "s":
                print("Operação cancelada.")
                raise SystemExit(0)

            criar_pastas()
            limpar_dataset()

        executar()
