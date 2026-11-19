"""
Os três modelos do projeto.

Todos recebem um lote de imagens 28x28 em tons de cinza, com valores em [0, 1]
(traço branco = 1, fundo preto = 0), no formato (N, 1, 28, 28), e devolvem os
"logits" z_c (N, 10). A probabilidade P(c | x) vem de softmax(z).

- MLP:      linha de base, enxerga a imagem como 784 números soltos.
- LeNet5:   referência histórica (LeCun et al., 1998), com simplificações modernas.
- CNNEquipe: o reconhecedor do sistema, projetado pelo grupo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Perceptron multicamadas: 784 -> 256 -> 128 -> 10.

    Cada pixel vira uma entrada independente. A rede não sabe que o pixel (3,4)
    é vizinho do (3,5): se o dígito andar 2 pixels, as entradas mudam quase
    todas. Por isso esperamos que ela sofra no teste de deslocamento.
    """

    def __init__(self, ocultas=(256, 128), dropout=0.2):
        super().__init__()
        camadas = []
        entrada = 28 * 28
        for n in ocultas:
            camadas += [nn.Linear(entrada, n), nn.ReLU(), nn.Dropout(dropout)]
            entrada = n
        camadas.append(nn.Linear(entrada, 10))
        self.rede = nn.Sequential(*camadas)

    def forward(self, x):
        return self.rede(x.flatten(1))


class LeNet5(nn.Module):
    """
    LeNet-5 conforme a Tabela 2 do enunciado.

    Simplificações declaradas em relação ao artigo de 1998:
      - ReLU no lugar da tangente hiperbólica;
      - max pooling no lugar da subamostragem treinável;
      - C3 ligada a todos os 6 mapas de S2 (o original usa uma tabela parcial);
      - softmax (via entropia cruzada) no lugar das unidades RBF de saída.

    A entrada original é 32x32 (MNIST com 2 pixels de margem). Para que o
    sistema entregue sempre 28x28 a todos os modelos, a margem é colocada
    aqui dentro com F.pad.
    """

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5)      # 32x32 -> 6 x 28x28
        self.s2 = nn.MaxPool2d(2)                     # -> 6 x 14x14
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)     # -> 16 x 10x10
        self.s4 = nn.MaxPool2d(2)                     # -> 16 x 5x5
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)   # -> 120 x 1x1
        self.f6 = nn.Linear(120, 84)
        self.saida = nn.Linear(84, 10)

    def forward(self, x):
        x = F.pad(x, (2, 2, 2, 2))                    # 28x28 -> 32x32
        x = self.s2(F.relu(self.c1(x)))
        x = self.s4(F.relu(self.c3(x)))
        x = F.relu(self.c5(x)).flatten(1)
        x = F.relu(self.f6(x))
        return self.saida(x)


def bloco(entrada, saida):
    """Convolução 3x3 com padding (mantém o tamanho) + BatchNorm + ReLU."""
    return nn.Sequential(
        nn.Conv2d(entrada, saida, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(saida),
        nn.ReLU(inplace=True),
    )


class CNNEquipe(nn.Module):
    """
    CNN do grupo. Justificativa camada a camada:

    Estágio 1 (28x28): duas convoluções 3x3 com 32 filtros.
        Duas 3x3 em sequência enxergam uma janela 5x5 (como a C1 da LeNet),
        com menos parâmetros e uma não linearidade a mais. Detectam pedaços
        de traço: bordas, pontas, curvas.
    MaxPool 2x2 -> 14x14: tolerância a pequenos deslocamentos e menos custo.

    Estágio 2 (14x14): duas convoluções 3x3 com 64 filtros.
        Combinam os pedaços em partes de dígito: laços, cruzamentos, hastes.
    MaxPool 2x2 -> 7x7.

    Estágio 3 (7x7): uma convolução 3x3 com 128 filtros.
        Cada unidade já vê quase o dígito inteiro (campo receptivo ~ 24 px).

    Média global (global average pooling): resume cada um dos 128 mapas em
        um número, sem importar ONDE a característica apareceu. Isso dá mais
        tolerância a deslocamento do que achatar o mapa 7x7 e quase não tem
        parâmetros (ao contrário de uma camada densa sobre 128*7*7 entradas).

    Dropout + camada densa 128 -> 10: classificação.

    BatchNorm em cada convolução estabiliza o treino e deixa a rede menos
    sensível a variações de contraste entre o MNIST e a câmera.

    Orçamento de tempo real: precisa classificar 30 recortes em <= 20 ms na
    CPU do notebook (medido em experimentos/e1_modelos.py).
    """

    def __init__(self, dropout=0.3):
        super().__init__()
        self.extrator = nn.Sequential(
            bloco(1, 32), bloco(32, 32), nn.MaxPool2d(2),     # 28 -> 14
            bloco(32, 64), bloco(64, 64), nn.MaxPool2d(2),    # 14 -> 7
            bloco(64, 128),                                   # 7x7
            nn.AdaptiveAvgPool2d(1),                          # 128 x 1x1
        )
        self.classificador = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classificador(self.extrator(x))


MODELOS = {
    "mlp": MLP,
    "lenet5": LeNet5,
    "cnn": CNNEquipe,
}


def criar_modelo(nome):
    return MODELOS[nome]()


def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def carregar_modelo(nome, caminho_pesos, dispositivo="cpu"):
    modelo = criar_modelo(nome)
    modelo.load_state_dict(torch.load(caminho_pesos, map_location=dispositivo))
    modelo.eval()
    return modelo
