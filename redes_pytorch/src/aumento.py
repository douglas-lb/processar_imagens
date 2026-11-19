"""
Aumento de dados orientado ao quadro branco (aplicado SÓ no treino).

Cada transformação imita algo que a câmera ou o pincel produzem de verdade.
Recebe e devolve uma imagem 28x28 float32 em [0, 1] (traço branco, fundo preto).

| Transformação          | Por que existe (o que acontece no quadro)                         |
|------------------------|-------------------------------------------------------------------|
| Rotação ±15°           | o professor escreve inclinado; câmera levemente torta no tripé    |
| Escala 0,85–1,15       | a normalização 20x20 nunca sai perfeita (recorte com folga/justo) |
| Cisalhamento ±0,25     | câmera de lado (até ~30°) e caligrafia inclinada                  |
| Translação ±2 px       | o centro de massa de um recorte ruidoso não cai no centro exato   |
| Distorção elástica     | variação natural da mão (Simard et al., 2003)                     |
| Erosão (afinar)        | a 3 m o traço do pincel tem ~1 px: depois de reduzir fica fino    |
| Dilatação (engrossar)  | a 1,5 m, ou pincel novo, o traço fica grosso                      |
| Desfoque gaussiano     | foco da câmera, tremor, movimento                                 |
| Ruído                  | ruído do sensor, reflexo, resquício de escrita mal apagada        |
"""

import cv2
import numpy as np


class AumentoQuadro:
    def __init__(self, rng=None,
                 rotacao=15.0, escala=(0.85, 1.15), cisalhamento=0.25,
                 translacao=2.0,
                 p_elastica=0.5, alfa_elastica=15.0, sigma_elastica=4.0,
                 p_espessura=0.4, p_desfoque=0.3, p_ruido=0.3):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.rotacao = rotacao
        self.escala = escala
        self.cisalhamento = cisalhamento
        self.translacao = translacao
        self.p_elastica = p_elastica
        self.alfa_elastica = alfa_elastica
        self.sigma_elastica = sigma_elastica
        self.p_espessura = p_espessura
        self.p_desfoque = p_desfoque
        self.p_ruido = p_ruido

    # ---------- transformações individuais ----------

    def afim(self, img):
        """Rotação + escala + cisalhamento + translação em uma única matriz."""
        r = self.rng
        ang = np.deg2rad(r.uniform(-self.rotacao, self.rotacao))
        s = r.uniform(*self.escala)
        sh = r.uniform(-self.cisalhamento, self.cisalhamento)
        tx, ty = r.uniform(-self.translacao, self.translacao, size=2)

        rot = np.array([[np.cos(ang), -np.sin(ang)],
                        [np.sin(ang),  np.cos(ang)]])
        cis = np.array([[1.0, sh],
                        [0.0, 1.0]])
        A = s * rot @ cis

        c = np.array([13.5, 13.5])                 # centro da imagem 28x28
        b = c - A @ c + np.array([tx, ty])         # gira em torno do centro
        M = np.hstack([A, b[:, None]]).astype(np.float32)
        return cv2.warpAffine(img, M, (28, 28), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def elastica(self, img):
        """
        Distorção elástica (Simard et al., 2003): sorteia um deslocamento
        aleatório para cada pixel, suaviza esse campo com uma gaussiana
        (para que pixels vizinhos andem juntos) e reamostra a imagem.
        """
        h, w = img.shape
        dx = self.rng.uniform(-1, 1, (h, w)).astype(np.float32)
        dy = self.rng.uniform(-1, 1, (h, w)).astype(np.float32)
        dx = cv2.GaussianBlur(dx, (0, 0), self.sigma_elastica) * self.alfa_elastica
        dy = cv2.GaussianBlur(dy, (0, 0), self.sigma_elastica) * self.alfa_elastica
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        return cv2.remap(img, xs + dx, ys + dy, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def espessura(self, img):
        """Erosão (afina) ou dilatação (engrossa) com núcleo 2x2."""
        nucleo = np.ones((2, 2), np.uint8)
        if self.rng.random() < 0.5:
            return cv2.erode(img, nucleo)
        return cv2.dilate(img, nucleo)

    def desfoque(self, img):
        sigma = self.rng.uniform(0.4, 1.0)
        return cv2.GaussianBlur(img, (0, 0), sigma)

    def ruido(self, img):
        desvio = self.rng.uniform(0.02, 0.10)
        return img + self.rng.normal(0, desvio, img.shape).astype(np.float32)

    # ---------- composição ----------

    def __call__(self, img):
        img = img.astype(np.float32)
        img = self.afim(img)
        if self.rng.random() < self.p_elastica:
            img = self.elastica(img)
        if self.rng.random() < self.p_espessura:
            img = self.espessura(img)
        if self.rng.random() < self.p_desfoque:
            img = self.desfoque(img)
        if self.rng.random() < self.p_ruido:
            img = self.ruido(img)
        return np.clip(img, 0.0, 1.0)
