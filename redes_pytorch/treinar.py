"""
Treina os modelos no MNIST.

Uso:
    python treinar.py                 # treina os três (mlp, lenet5, cnn)
    python treinar.py --modelo cnn    # treina só um
    python treinar.py --modelo cnn --epocas 30 --lr 0.001
    python treinar.py --modelo cnn --sem-aumento --sufixo _sem_aumento

O que acontece em cada época:
  1. passa por todo o treino: para cada lote calcula a saída, a perda de
     entropia cruzada, o gradiente (retropropagação) e atualiza os pesos;
  2. mede perda e acurácia na VALIDAÇÃO (sem atualizar nada);
  3. se a acurácia de validação melhorou, guarda esses pesos.
Para quando a validação não melhora por `paciencia` épocas seguidas
(parada antecipada). O teste do MNIST NÃO é usado aqui.

Saídas:
  pesos/<modelo>.pt                     melhores pesos (pela validação)
  resultados/historico_<modelo>.json    curvas de perda/acurácia por época
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.aumento import AumentoQuadro
from src.dados import SEMENTE, carregar_mnist, criar_loader
from src.modelos import contar_parametros, criar_modelo

RAIZ = Path(__file__).resolve().parent
PASTA_PESOS = RAIZ / "pesos"
PASTA_RESULTADOS = RAIZ / "resultados"

# Por padrão, só a CNN da equipe treina com aumento de dados. MLP e LeNet-5
# ficam "puros" como referência. (Pode ser mudado com --aumento / --sem-aumento.)
AUMENTO_PADRAO = {"mlp": False, "lenet5": False, "cnn": True}


def fixar_semente(semente):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)


def avaliar(modelo, loader, criterio):
    """Perda média e acurácia, sem calcular gradientes."""
    modelo.eval()
    perda_total, acertos, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            z = modelo(x)
            perda_total += criterio(z, y).item() * len(y)
            acertos += (z.argmax(1) == y).sum().item()
            n += len(y)
    return perda_total / n, acertos / n


def treinar(nome, epocas, lr, tamanho_lote, paciencia, usar_aumento, dados, sufixo=""):
    fixar_semente(SEMENTE)
    modelo = criar_modelo(nome)
    print(f"\n=== {nome} | {contar_parametros(modelo):,} parâmetros | "
          f"aumento={'sim' if usar_aumento else 'não'} ===")

    x_tr, y_tr = dados["treino"]
    x_val, y_val = dados["validacao"]
    aumento = AumentoQuadro(np.random.default_rng(SEMENTE)) if usar_aumento else None
    loader_tr = criar_loader(x_tr, y_tr, tamanho_lote, embaralhar=True, aumento=aumento)
    loader_val = criar_loader(x_val, y_val, 512)

    # CrossEntropyLoss = softmax + (-log p(y|x)), média no lote.
    criterio = nn.CrossEntropyLoss()
    # Adam: variação da descida do gradiente w <- w - lr * (gradiente ajustado)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    # Reduz a taxa de aprendizado pela metade se a validação estagnar.
    agendador = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, mode="max", factor=0.5, patience=2)

    historico = {"modelo": nome, "aumento": usar_aumento, "lr": lr,
                 "tamanho_lote": tamanho_lote, "semente": SEMENTE,
                 "parametros": contar_parametros(modelo),
                 "perda_treino": [], "acc_treino": [],
                 "perda_val": [], "acc_val": []}
    melhor_acc, melhor_epoca, sem_melhora = -1.0, 0, 0
    PASTA_PESOS.mkdir(exist_ok=True)
    caminho_pesos = PASTA_PESOS / f"{nome}{sufixo}.pt"

    for epoca in range(1, epocas + 1):
        inicio = time.time()
        modelo.train()
        perda_total, acertos, n = 0.0, 0, 0
        for x, y in loader_tr:
            z = modelo(x)                    # 1. propagação para frente
            perda = criterio(z, y)           # 2. perda de entropia cruzada
            otimizador.zero_grad()
            perda.backward()                 # 3. retropropagação: dL/dw
            otimizador.step()                # 4. atualização dos pesos
            perda_total += perda.item() * len(y)
            acertos += (z.argmax(1) == y).sum().item()
            n += len(y)

        perda_tr, acc_tr = perda_total / n, acertos / n
        perda_val, acc_val = avaliar(modelo, loader_val, criterio)
        agendador.step(acc_val)

        for chave, valor in [("perda_treino", perda_tr), ("acc_treino", acc_tr),
                             ("perda_val", perda_val), ("acc_val", acc_val)]:
            historico[chave].append(valor)

        marca = ""
        if acc_val > melhor_acc:
            melhor_acc, melhor_epoca, sem_melhora = acc_val, epoca, 0
            torch.save(modelo.state_dict(), caminho_pesos)
            marca = "  <- melhor"
        else:
            sem_melhora += 1

        print(f"época {epoca:2d} | treino: perda {perda_tr:.4f} acc {acc_tr:.4f} | "
              f"val: perda {perda_val:.4f} acc {acc_val:.4f} | "
              f"{time.time() - inicio:.0f}s{marca}", flush=True)

        if sem_melhora >= paciencia:
            print(f"Parada antecipada: {paciencia} épocas sem melhorar.")
            break

    historico["melhor_epoca"] = melhor_epoca
    historico["melhor_acc_val"] = melhor_acc
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    with open(PASTA_RESULTADOS / f"historico_{nome}{sufixo}.json", "w") as f:
        json.dump(historico, f, indent=2)
    print(f"Melhor época: {melhor_epoca} (val acc {melhor_acc:.4f}) -> {caminho_pesos}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", choices=["mlp", "lenet5", "cnn", "todos"], default="todos")
    p.add_argument("--epocas", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lote", type=int, default=128)
    p.add_argument("--paciencia", type=int, default=4)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--aumento", dest="aumento", action="store_true", default=None)
    g.add_argument("--sem-aumento", dest="aumento", action="store_false")
    p.add_argument("--sufixo", default="", help="ex.: _sem_aumento -> pesos/cnn_sem_aumento.pt")
    args = p.parse_args()

    dados = carregar_mnist()
    nomes = ["mlp", "lenet5", "cnn"] if args.modelo == "todos" else [args.modelo]
    for nome in nomes:
        usar = AUMENTO_PADRAO[nome] if args.aumento is None else args.aumento
        treinar(nome, args.epocas, args.lr, args.lote, args.paciencia, usar, dados, args.sufixo)


if __name__ == "__main__":
    main()
