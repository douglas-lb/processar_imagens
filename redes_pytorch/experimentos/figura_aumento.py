"""Figura para o relatório: 10 dígitos do MNIST (1ª coluna) e 7 versões aumentadas.
Uso: python -m experimentos.figura_aumento  -> resultados/figura_aumento.png"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.aumento import AumentoQuadro
from src.dados import carregar_mnist

x, y = carregar_mnist()["treino"]
aum = AumentoQuadro(np.random.default_rng(42))
fig, eixos = plt.subplots(10, 8, figsize=(8, 10))
for c in range(10):
    img = x[np.where(y == c)[0][0]]
    for j in range(8):
        eixos[c, j].imshow(img if j == 0 else aum(img), cmap="gray", vmin=0, vmax=1)
        eixos[c, j].axis("off")
eixos[0, 0].set_title("original", fontsize=9)
for j in range(1, 8):
    eixos[0, j].set_title("aumentada", fontsize=9)
fig.tight_layout()
saida = Path(__file__).resolve().parent.parent / "resultados" / "figura_aumento.png"
saida.parent.mkdir(exist_ok=True)
fig.savefig(saida, dpi=120)
print(saida)
