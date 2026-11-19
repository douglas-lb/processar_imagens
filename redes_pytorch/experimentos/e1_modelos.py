"""
E1 - Do MLP à CNN (e o teste de deslocamento, item 4.3.5).

Uso (da pasta pp02):  python -m experimentos.e1_modelos

Para cada modelo treinado (pesos/<modelo>.pt) mede:
  - número de parâmetros;
  - acurácia no TESTE do MNIST;
  - tempo de inferência por dígito e de um lote de 30 recortes (orçamento: 20 ms);
  - acurácia com as imagens de teste deslocadas de 0 a 6 pixels.
E desenha as curvas de treino a partir de resultados/historico_<modelo>.json.

Saídas em resultados/:
  e1_tabela.md, e1_deslocamento.json, e1_curvas.png, e1_deslocamento.png
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dados import carregar_mnist
from src.modelos import carregar_modelo, contar_parametros

RAIZ = Path(__file__).resolve().parent.parent
PESOS = RAIZ / "pesos"
RES = RAIZ / "resultados"
# arquivo de pesos -> (rótulo, arquitetura)
NOMES = {"mlp": "MLP", "lenet5": "LeNet-5", "cnn_sem_aumento": "CNN equipe (sem aumento)",
         "cnn": "CNN da equipe"}
ARQUITETURA = {"mlp": "mlp", "lenet5": "lenet5", "cnn_sem_aumento": "cnn", "cnn": "cnn"}
CORES = {"mlp": "#d95f02", "lenet5": "#7570b3", "cnn_sem_aumento": "#a6761d", "cnn": "#1b9e77"}


def acuracia(modelo, x, y, lote=1000):
    acertos = 0
    with torch.no_grad():
        for i in range(0, len(x), lote):
            xb = torch.from_numpy(x[i:i + lote]).unsqueeze(1)
            acertos += (modelo(xb).argmax(1).numpy() == y[i:i + lote]).sum()
    return acertos / len(x)


def tempo_ms(modelo, n_imagens, repeticoes=200):
    """Mediana do tempo (ms) para classificar um lote de n_imagens."""
    x = torch.rand(n_imagens, 1, 28, 28)
    with torch.no_grad():
        for _ in range(20):                       # aquecimento
            modelo(x)
        tempos = []
        for _ in range(repeticoes):
            t0 = time.perf_counter()
            modelo(x)
            tempos.append((time.perf_counter() - t0) * 1000)
    return float(np.median(tempos))


def deslocar(imagens, dx, dy):
    """Desloca as imagens (N,28,28) por (dx,dy) pixels, preenchendo com preto."""
    saida = np.zeros_like(imagens)
    h, w = imagens.shape[1:]
    xs_o, xs_d = slice(max(0, -dx), min(w, w - dx)), slice(max(0, dx), min(w, w + dx))
    ys_o, ys_d = slice(max(0, -dy), min(h, h - dy)), slice(max(0, dy), min(h, h + dy))
    saida[:, ys_d, xs_d] = imagens[:, ys_o, xs_o]
    return saida


def teste_deslocamento(modelo, x, y, maximo=6):
    """Para cada d, média da acurácia deslocando d px para cima/baixo/esq./dir."""
    direcoes = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    resultado = []
    for d in range(maximo + 1):
        accs = [acuracia(modelo, deslocar(x, d * a, d * b), y) for a, b in direcoes]
        resultado.append(float(np.mean(accs)))
    return resultado


def main():
    torch.set_grad_enabled(False)
    x_teste, y_teste = carregar_mnist()["teste"]
    RES.mkdir(exist_ok=True)

    linhas, deslocamentos, historicos = [], {}, {}
    for nome in NOMES:
        caminho = PESOS / f"{nome}.pt"
        if not caminho.exists():
            print(f"[pulando {nome}: {caminho} não existe]")
            continue
        modelo = carregar_modelo(ARQUITETURA[nome], caminho)
        acc = acuracia(modelo, x_teste, y_teste)
        t1 = tempo_ms(modelo, 1)
        t30 = tempo_ms(modelo, 30)
        linhas.append((NOMES[nome], contar_parametros(modelo), acc, t1, t30))
        deslocamentos[nome] = teste_deslocamento(modelo, x_teste, y_teste)
        print(f"{NOMES[nome]:26s} params={contar_parametros(modelo):>8,} "
              f"teste={acc:.4f}  1 dígito={t1:.2f} ms  lote 30={t30:.2f} ms")
        print(f"  deslocamento 0..6 px: {[round(a, 3) for a in deslocamentos[nome]]}")

        h = RES / f"historico_{nome}.json"
        if h.exists():
            historicos[nome] = json.loads(h.read_text())

    # ---- tabela ----
    tabela = ["| Modelo | Parâmetros | Acurácia teste MNIST | Tempo 1 dígito (ms) | Lote de 30 (ms) |",
              "|---|---:|---:|---:|---:|"]
    for nome, p, acc, t1, t30 in linhas:
        tabela.append(f"| {nome} | {p:,} | {acc * 100:.2f}% | {t1:.2f} | {t30:.2f} |")
    (RES / "e1_tabela.md").write_text("\n".join(tabela) + "\n", encoding="utf-8")
    (RES / "e1_deslocamento.json").write_text(json.dumps(deslocamentos, indent=2))

    # ---- curvas de treino ----
    if historicos:
        fig, eixos = plt.subplots(1, 2, figsize=(11, 4))
        for nome, h in historicos.items():
            ep = range(1, len(h["perda_treino"]) + 1)
            eixos[0].plot(ep, h["perda_treino"], "--", color=CORES[nome], label=f"{NOMES[nome]} treino")
            eixos[0].plot(ep, h["perda_val"], "-", color=CORES[nome], label=f"{NOMES[nome]} validação")
            eixos[1].plot(ep, h["acc_treino"], "--", color=CORES[nome])
            eixos[1].plot(ep, h["acc_val"], "-", color=CORES[nome])
            eixos[1].axvline(h["melhor_epoca"], color=CORES[nome], alpha=0.25)
        eixos[0].set(title="Perda (entropia cruzada)", xlabel="época")
        eixos[1].set(title="Acurácia (linha vertical = época escolhida)", xlabel="época")
        eixos[0].legend(fontsize=8)
        for e in eixos:
            e.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RES / "e1_curvas.png", dpi=150)

    # ---- teste de deslocamento ----
    if deslocamentos:
        fig, ax = plt.subplots(figsize=(6, 4))
        for nome, accs in deslocamentos.items():
            ax.plot(range(len(accs)), np.array(accs) * 100, "o-", color=CORES[nome], label=NOMES[nome])
        ax.set(title="Teste de deslocamento (MNIST teste)", xlabel="deslocamento (pixels)",
               ylabel="acurácia (%)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(RES / "e1_deslocamento.png", dpi=150)

    print("\n" + "\n".join(tabela))
    print(f"\nArquivos salvos em {RES}")


if __name__ == "__main__":
    main()
