# Redes em PyTorch: MLP, LeNet-5 e CNN (experimento E1)

Todos os comandos abaixo são executados de dentro da pasta `redes_pytorch/`.

## Instalação
```
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```
Para instalar o PyTorch só de CPU (menor):
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

## Parte 1 – Modelos (MLP, LeNet-5, CNN da equipe)
```
python treinar.py                          # treina os três no MNIST (baixa o MNIST na 1ª vez)
python -m experimentos.e1_modelos          # E1: tabela, curvas, teste de deslocamento
python -m experimentos.figura_aumento      # figura do aumento de dados
```
- Semente fixa: 42 (divisão treino/validação, inicialização, aumento).
- MNIST: 55.000 treino / 5.000 validação / 10.000 teste (teste usado só no E1).
- Pesos em `pesos/`, históricos e figuras em `resultados/`.

## Convenção de entrada dos modelos (IMPORTANTE para a percepção)
Todos os modelos recebem `float32` com forma `(N, 1, 28, 28)`:
- traço **branco (1.0)** sobre fundo **preto (0.0)**, valores em [0, 1] (pixel/255);
- dígito redimensionado para caber em 20×20 preservando a proporção;
- centralizado no 28×28 pelo **centro de massa**.

A LeNet-5 coloca internamente a margem de 2 px (28→32).

## Estrutura
```
src/modelos.py     MLP, LeNet5, CNNEquipe
src/aumento.py     aumento de dados orientado ao quadro (só no treino)
src/dados.py       carregamento e divisão do MNIST
treinar.py         treino com validação e parada antecipada
experimentos/      E1 e figuras
```
