# Reconhecimento de dígitos manuscritos (0–9)

Pipeline em Python: **dataset próprio** → **CNN (TensorFlow/Keras)** → **inferência e webcam**. As imagens são normalizadas para o padrão **MNIST** (28×28, fundo preto, número branco).

## Visão geral

| Etapa | O que faz | Scripts principais |
| --- | --- | --- |
| **1. Dataset** | Fotografias → rotulagem → `dataset/` + `metadata.csv` + `splits.json` | `processar.py`, `preprocess.py` |
| **2. Modelo** | Treino, avaliação, previsão em arquivo; export **TFLite** para uso futuro em mobile | `treinar_modelo.py`, `avaliar.py`, `prever.py` |
| **3. Câmera** | Reconhecimento em tempo real pela webcam | `camera_webcam.py` |

> **Escopo:** o export TFLite (`modelos/digitos.tflite`) prepara um app mobile no futuro; **desenvolvimento de app mobile está fora do escopo** deste repositório.

## Requisitos

- **Python 3.12** recomendado (ambiente do projeto testado com 3.12).
- **TensorFlow 2.21** suporta Python **3.10–3.13**; **não instala em Python 3.14** (e versões futuras podem demorar).
- Webcam: desktop com interface gráfica; no Linux, usuário no grupo `video`.

Dependências listadas em `requirements.txt`: OpenCV, NumPy, TensorFlow, scikit-learn, Matplotlib.

## Configuração do ambiente

Na pasta `processar_imagens/`, use o ambiente virtual **`.venv`** (já usado no projeto):

### Criar o `.venv`

| Sistema | Comando |
| --- | --- |
| Linux / macOS | `python3.12 -m venv .venv` (ou `python3 -m venv .venv`) |
| Windows (PowerShell) | `py -3.12 -m venv .venv` |

### Ativar

| Sistema | Comando |
| --- | --- |
| Linux / macOS | `source .venv/bin/activate` |
| Windows (PowerShell) | `.\.venv\Scripts\Activate.ps1` |
| Windows (CMD) | `.venv\Scripts\activate.bat` |

### Instalar dependências

```bash
pip install -r requirements.txt
```

Sem ativar o venv, use o interpretador direto: `.venv/bin/python` (Linux/macOS) ou `.venv\Scripts\python.exe` (Windows).

## Fluxo de uso

```text
fotos/  →  processar.py  →  dataset/ + splits.json
                ↓
        treinar_modelo.py  (MNIST ou --dataset dataset/)
                ↓
        modelos/digitos.keras, digitos.tflite, digitos_info.json
                ↓
        avaliar.py  →  métricas e modelos/matriz_confusao.png
                ↓
        prever.py <imagem>  |  camera_webcam.py
```

### 1. Dataset a partir de fotos

1. Coloque fotos em `fotos/` (JPG, PNG, BMP; números **escuros em papel claro**).
2. Execute:

```bash
python processar.py
python processar.py --refazer          # recria tudo (confirmação)
python processar.py --apenas-splits    # só regera splits.json
```

Cada foto gera até cinco variações (data augmentation) em `dataset/0` … `dataset/9`, com rastreamento em `dataset/metadata.csv`.

**Pré-processamento compartilhado** (dataset, `prever.py` e câmera):

```python
from preprocess import preprocessar_digito

imagem = preprocessar_digito("fotos/7_foto.jpg")  # ou ndarray BGR do OpenCV
# 28×28 uint8, fundo 0 e número 255; None se não achar o dígito
```

### 2. Treinar o modelo

```bash
python treinar_modelo.py                    # MNIST (padrão)
python treinar_modelo.py --dataset dataset/ # dataset próprio (usa splits.json)
python treinar_modelo.py --dataset dataset/ --com-mnist
python treinar_modelo.py --epocas 10 --lote 64
```

Artefatos em `modelos/`:

- `digitos.keras` — modelo Keras (local; ver `.gitignore`)
- `digitos.tflite` — inferência leve (webcam / mobile futuro)
- `digitos_info.json` — metadados do último treino (épocas, acurácia, etc.)

A acurácia reportada deve vir de `digitos_info.json` ou de `avaliar.py`, não de estimativas manuais.

### 3. Avaliar

```bash
python avaliar.py
python avaliar.py --dataset dataset/
python avaliar.py --tflite
```

Gera relatório por classe e salva `modelos/matriz_confusao.png`.

### 4. Prever em uma imagem

```bash
python prever.py foto.jpg
python prever.py foto.jpg --tflite
python prever.py foto.jpg --salvar-preprocessada /tmp/entrada.png
```

Aceita foto (número escuro em fundo claro) ou PNG já 28×28 estilo MNIST.

### 5. Webcam

Escreva um dígito com caneta escura em papel claro e enquadre na área indicada.

```bash
python camera_webcam.py
python camera_webcam.py --camera 1 --limiar 0.80
python camera_webcam.py --keras    # força Keras; --tflite força TFLite
python camera_webcam.py --roi 0.80 # quadro inicial maior/menor (0.40–0.92)
```

- Teclas: `q` sai; `s` salva; `h` liga/desliga espelho; `+` / `-` ajusta o tamanho do quadro verde.
- Por padrão usa TFLite se `modelos/digitos.tflite` existir.
- A imagem começa **espelhada**; use `h` ou `--sem-espelho` se o lado parecer invertido.
- O quadro verde começa maior (~75% do menor lado). Coloque o **dígito inteiro** dentro dele — se só o haste do 4 entrar, o modelo vê um 1.
- Prefira **papel claro + caneta escura**. A câmera detecta a folha, binariza com limiar adaptativo e só confirma após frames estáveis. Confira a miniatura **entrada 28x28**: o dígito precisa aparecer inteiro.

### Atalho no menu do Linux

O lançador **Reconhecedor de Dígitos** fica em `~/.local/share/applications/reconhecedor-digitos.desktop` e chama `launch-reconhecedor-digitos.sh`. Busque pelo nome no menu de aplicativos (Hyprland / rofi / wofi etc.) ou execute:

```bash
gtk-launch reconhecedor-digitos
# ou
./launch-reconhecedor-digitos.sh
```

## Estrutura de pastas

```text
processar_imagens/
├── .venv/                    # ambiente virtual (não versionado)
├── fotos/                    # fotografias brutas
├── dataset/                  # imagens 28×28 por classe
│   ├── 0/ … 9/
│   ├── metadata.csv
│   └── splits.json
├── modelos/
│   ├── digitos.keras         # gerado no treino (ignorado no Git)
│   ├── digitos.tflite
│   ├── digitos_info.json
│   └── matriz_confusao.png
├── capturas/                 # salvamentos da webcam (ignorado no Git)
├── preprocess.py
├── processar.py
├── treinar_modelo.py
├── avaliar.py
├── prever.py
├── camera_webcam.py
├── requirements.txt
└── README.md
```

## Splits e vazamento de dados

As cinco variações de uma mesma foto são quase idênticas. **Nunca** divida o dataset sorteando arquivos `.png` soltos: isso coloca variações da mesma foto em treino e teste e **infla a acurácia**.

O `splits.json` agrupa por **`source_id`** (nome da foto em `fotos/`): original e augmentations ficam no **mesmo** conjunto (treino / validação / teste), com divisão estratificada por classe (80% / 10% / 10%, semente 42). Classes com poucas fotos podem ficar só no treino.

Para métricas mais realistas na validação/teste, avalie preferencialmente imagens com `eh_aumento = 0` no `metadata.csv`.

Estrutura resumida do `splits.json`:

```json
{
  "estrategia": "por_source_id_estratificado_por_classe",
  "source_ids": { "treino": ["..."], "validacao": ["..."], "teste": ["..."] },
  "treino": ["0/0_00001.png", "..."],
  "validacao": ["..."],
  "teste": ["..."]
}
```

## Dicas de captura

- Boa iluminação; contraste alto (tinta escura, papel claro).
- Confira visualmente as amostras em `dataset/` após `processar.py`.
- Imagens antigas sem `metadata.csv` não entram nos splits; use `processar.py --refazer` se necessário.

## Clone (opcional)

```bash
git clone https://github.com/douglas-lb/processar_imagens.git
cd processar_imagens
```

Em seguida, crie/ative o `.venv` e instale `requirements.txt` como acima.