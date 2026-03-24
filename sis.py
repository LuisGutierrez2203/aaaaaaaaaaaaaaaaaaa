import cv2
import numpy as np

print(np.__version__)


# ------------------ Inicio de la extracción de placa -----------------#
def puntos_placa(placa_seg, placaOrg_dim):

    # Crea una imagene a negro de tamaño que la imagen original de la placa
    img_placa_bn = np.zeros(placaOrg_dim, dtype=np.uint8)

    # Verifica cuantas cajas de detección hay y elige la caja de mayor tamaño.
    if len(placa_seg.masks.xy) > 1:
        mascara_xy = []
        for i in placa_seg.masks.xy:
            mascara_xy.append(i[0] * i[1])

        puntos = np.array(
            placa_seg.masks.xy[np.argmax(mascara_xy)], np.int32
        )  # Puntos donde se encuentra la placa

    else:
        puntos = np.array(
            placa_seg.masks.xy, np.int32
        )  # Puntos donde se encuentra la placa

    cv2.fillPoly(
        img_placa_bn, [puntos], color=255
    )  # Pinta el area donde se encuentra la placa

    contornos, _ = cv2.findContours(
        img_placa_bn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )  # Detección de contornos

    if contornos:
        max_contorno = max(
            contornos, key=cv2.contourArea
        )  # Selecciona el contorno más grande.

        # Aproximar con 4 puntos (polígono)
        epsilon = 0.02 * cv2.arcLength(max_contorno, True)
        area_aprox = cv2.approxPolyDP(max_contorno, epsilon, True)

        # Si no tiene exactamente 4 puntos, ajustar con boundingRect
        if len(area_aprox) != 4:
            x, y, w, h = cv2.boundingRect(max_contorno)
            area_aprox = np.array(
                [[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]]
            )

        # Los puntos:
        puntos = [tuple(pt[0]) for pt in area_aprox]
        puntos = np.array(puntos, np.int32)

        return puntos
    else:
        return []


def ordenar_puntos(puntos):
    rect = np.zeros((4, 2), np.int32)

    s = puntos.sum(axis=1)
    d = np.diff(puntos, axis=1)

    rect[0] = puntos[np.argmin(s)]  # superior - Izquierdo
    rect[1] = puntos[np.argmin(d)]  # Superior - Derecho
    rect[2] = puntos[np.argmax(s)]  # Inferior - Derecho
    rect[3] = puntos[np.argmax(d)]  # Inferior - Izquierdo

    return rect


def extrac_placa(img_placa, placa_seg_model):
    a = placa_seg_model(img_placa, save=False, verbose=False)[0]

    if a.masks != None:

        puntos = puntos_placa(a, img_placa.shape[:2])

        pts_src = ordenar_puntos(puntos)
        (si, sd, id, ii) = (
            pts_src  # Superior Izquierda, Superior Derecha, Inferior Derecha, Inferior Izquierda
        )
        # Ancho
        max_ancho = int(max(np.linalg.norm(sd - si), np.linalg.norm(id - ii)))

        # Alto
        max_alto = int(max(np.linalg.norm(sd - id), np.linalg.norm(si - ii)))

        puntos_destino = np.array(
            [
                [0, 0],
                [max_ancho - 1, 0],
                [max_ancho - 1, max_alto - 1],
                [0, max_alto - 1],
            ],
            dtype=np.float32,
        )

        H, _ = cv2.findHomography(pts_src, puntos_destino)
        img_placa_persp = cv2.warpPerspective(img_placa, H, (max_ancho, max_alto))

        return img_placa_persp

    return []


# ------------------ Fin de la extracción de placa -----------------#
# ------------------ Inicio de la extracción de caracteres -----------------#


def detec_supcaja(box_dic, area_punts, cls):
    txt_placa = ""
    text_restr = []  # Vector para discriminar caracteres.

    for p in range(0, len(area_punts) - 1):

        xy1 = box_dic.get(area_punts[p])[
            1
        ]  # Puntos de la primera caja [x1, y1, x2, y2]

        xy2 = box_dic.get(area_punts[p + 1])[
            1
        ]  # Puntos de la segunda caja [x1, y1, x2, y2]

        w2 = int(xy2[2]) - int(xy2[0])  # Alto de la segunda caja
        h2 = int(xy2[3]) - int(xy2[1])  # Ancho de la segunda caja

        cx = int(xy2[0] + w2 / 2)  # X Centro de la segunda caja
        cy = int(xy2[1] + h2 / 2)  # Y Centro de la segunda caja

        # Valida si existe doble detección sobre un caracter y elige el que
        # mayor porcentaje tenga.
        if (cx > xy1[0] and cx < xy1[2]) and (cy > xy1[1] and cy < xy1[3]):
            # Porcentaje confianza que sea ese caracter.
            p_c1 = box_dic.get(area_punts[p])[2]  # Porcentaje del primero
            p_c2 = box_dic.get(area_punts[p + 1])[2]  # Porcentaje del segundo
            if p_c1 > p_c2:
                text_restr.append(area_punts[p + 1])
            else:
                text_restr.append(area_punts[p])

    for j in area_punts:
        if not (j in text_restr):
            txt_placa += cls.get(int(box_dic.get(j)[0]))

    return txt_placa


def extrac_caracteres(placa_extrac, caracter_seg_model):
    area_punts = []
    box_dic = {}

    d = []
    c = []

    filt_bila = cv2.bilateralFilter(
        cv2.cvtColor(placa_extrac, cv2.COLOR_RGB2GRAY), 11, 17, 17
    )
    filt_bila = cv2.cvtColor(filt_bila, cv2.COLOR_GRAY2RGB)
    b = caracter_seg_model(filt_bila, save=False, verbose=False)[0]

    for clase, xy, conf in zip(b.boxes.cls, b.boxes.xyxy, b.boxes.conf):

        # xy = [x1, y1, x2, y2]
        c.append(float(xy[2]) - float(xy[0]))
        area_punts.append(float(xy[2]))
        box_dic[float(xy[2])] = [
            clase,
            xy,
            conf,
        ]  # Se guarda en un diccionario con clave la suma resultante.

    area_punts.sort()  # Organiza el vector de menor a mayor.
    c = c[np.argmax(c)]
    for i in range(0, len(area_punts) - 1):

        x1 = box_dic.get(area_punts[i])[1][2]
        x2 = box_dic.get(area_punts[i + 1])[1][0]
        if (float(x2) - float(x1)) < c:
            d.append(float(x2) - float(x1))
        else:
            d.append(0.0)

    izq = area_punts[: np.argmax(d) + 1]
    der = area_punts[np.argmax(d) + 1 :]

    # print("Detección superpuesta: ")
    # print("Texto Izquierdo")
    txt1 = detec_supcaja(box_dic, izq, b.names)
    # print("Texto Derecho")
    txt2 = detec_supcaja(box_dic, der, b.names)

    txt_placa = txt1 + " " + txt2
    return txt_placa


# ------------------ Fin de la extracción de caracteres -----------------#
