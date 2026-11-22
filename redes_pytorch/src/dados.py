"""
Carregamento do MNIST e divisão treino / validação / teste.

- Treino:    55.000 imagens (do conjunto de treino original)
- Validação:  5.000 imagens (separadas do treino, com semente fixa)
- Teste:     10.000 imagens (escritores diferentes; usado UMA vez, no fim)

Escala de intensidade: pixel / 255 -> [0, 1], traço branco sobre fundo preto.
A percepção (src/percepcao) precisa entregar os recortes nessa MESMA escala.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

PASTA_DADOS = Path(__file__).resolve().parent.parent / "dados"
SEMENTE = 42
TAM_VALIDACAO = 5000


def carregar_mnist():
    """Devolve arrays float32 em [0,1] (N, 28, 28) e rótulos int64."""
    treino = datasets.MNIST(PASTA_DADOS, train=True, download=True)
    teste = datasets.MNIST(PASTA_DADOS, train=False, download=True)

    x = treino.data.numpy().astype(np.float32) / 255.0
    y = treino.targets.numpy().astype(np.int64)
    x_teste = teste.data.numpy().astype(np.float32) / 255.0
    y_teste = teste.targets.numpy().astype(np.int64)

    rng = np.random.default_rng(SEMENTE)
    ordem = rng.permutation(len(x))
    i_val, i_tr = ordem[:TAM_VALIDACAO], ordem[TAM_VALIDACAO:]

    return {
        "treino": (x[i_tr], y[i_tr]),
        "validacao": (x[i_val], y[i_val]),
        "teste": (x_teste, y_teste),
    }


class ConjuntoImagens(Dataset):
    """Dataset simples; aplica o aumento (se houver) a cada vez que lê."""

    def __init__(self, imagens, rotulos, aumento=None):
        self.imagens = imagens
        self.rotulos = rotulos
        self.aumento = aumento

    def __len__(self):
        return len(self.imagens)

    def __getitem__(self, i):
        img = self.imagens[i]
        if self.aumento is not None:
            img = self.aumento(img)
        return torch.from_numpy(img).unsqueeze(0), int(self.rotulos[i])


def criar_loader(imagens, rotulos, tamanho_lote=128, embaralhar=False,
                 aumento=None, semente=SEMENTE):
    gerador = torch.Generator().manual_seed(semente)
    return DataLoader(ConjuntoImagens(imagens, rotulos, aumento),
                      batch_size=tamanho_lote, shuffle=embaralhar,
                      generator=gerador, num_workers=0)
