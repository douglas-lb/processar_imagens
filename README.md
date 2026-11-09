# Processar imagens

Projeto em Python para processamento de imagens de números manuscritos, com o objetivo de criar um dataset próprio para treinar uma rede neural de classificação de dígitos de 0 a 9.

As imagens são processadas para seguir o padrão do MNIST, utilizando resolução de **28 × 28 pixels**, fundo preto e números brancos.

## Estrutura do projeto

```text
processar_imagens/
│
├── venv/                  # Ambiente virtual
├── fotos/                 # Fotografias dos números manuscritos
├── dataset/               # Dataset processado
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   ├── ...
│   └── 9/
│
├── processar.py           # Script de processamento
├── requirements.txt       # Dependências do projeto
└── README.md
```

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/douglas-lb/processar_imagens.git
```

Entre na pasta do projeto:

```bash
cd processar_imagens
```

### 2. Crie o ambiente virtual

```bash
python -m venv venv
```

### 3. Ative o ambiente virtual

No Windows (PowerShell):

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Instale as dependências

```bash
pip install -r requirements.txt
```

## Como utilizar

### 1. Adicione as fotografias

Coloque as fotografias dos números manuscritos dentro da pasta `fotos/`.

São aceitos os formatos:

- JPG
- JPEG
- PNG
- BMP

### 2. Execute o programa

Com o ambiente virtual ativado, execute:

```bash
python processar.py
```

### 3. Identifique os números

O programa processará cada fotografia e exibirá uma prévia da imagem convertida.

Em seguida, solicitará que o usuário informe qual número está representado:

- Digite um número de 0 a 9 para classificar a imagem.
- Digite `S` para ignorar a imagem.

### 4. Processamento das imagens

O programa realiza automaticamente:

1. Conversão da imagem para escala de cinza.
2. Redução de ruídos.
3. Binarização para separar o número do fundo.
4. Identificação e recorte da região do número.
5. Centralização do número.
6. Redimensionamento para 28 × 28 pixels.
7. Geração de cinco variações utilizando aumento de dados (*Data Augmentation*).

As variações incluem pequenas rotações, deslocamentos e alterações de brilho e contraste.

### 5. Organização do dataset

Após o processamento, as imagens serão armazenadas automaticamente em pastas correspondentes aos seus rótulos.

Exemplo:

```text
dataset/
│
├── 0/
│   ├── 0_00001.png
│   ├── 0_00002.png
│   └── ...
│
├── 1/
│   ├── 1_00001.png
│   ├── 1_00002.png
│   └── ...
│
...
│
└── 9/
    ├── 9_00001.png
    └── ...
```

Cada pasta contém exclusivamente imagens do número correspondente.


## Observações importantes

- Verifique visualmente as imagens processadas para garantir que os números foram identificados corretamente.
- Utilize fotografias com boa iluminação e números escritos com caneta escura em papel claro.
- Evite misturar imagens derivadas da mesma fotografia entre os conjuntos de treinamento e validação.
