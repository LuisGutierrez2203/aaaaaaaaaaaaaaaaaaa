print("Cargando dependencias :..................")

import csv
import os
import shutil
import threading
from datetime import datetime

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, ttk
from ultralytics import YOLO

import sis as ss

print("Cargando modelos de IA .............")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

PLACA_DIR = os.path.join(BASE_DIR, "Placa")
STATIC_DIR = os.path.join(BASE_DIR, "static")
REGISTROS_PATH = os.path.join(BASE_DIR, "registros_acceso.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "Modelos_IA")
EMBEDDING_FILE = "face_embedding.npy"
FACE_THRESHOLD_SFACE = 0.38
FACE_THRESHOLD_FALLBACK = 0.20

for carpeta in (PLACA_DIR, STATIC_DIR):
    os.makedirs(carpeta, exist_ok=True)

zzz = YOLO(os.path.join(ROOT_DIR, "Modelos_IA", "Extrac_placas.pt"))
xxx = YOLO(os.path.join(ROOT_DIR, "Modelos_IA", "Segment_caracteres_gray4.pt"))

cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)


def _primera_ruta_existente(candidatos):
    for nombre in candidatos:
        ruta = os.path.join(MODEL_DIR, nombre)
        if os.path.exists(ruta):
            return ruta
    return None


class FaceVerifier:
    def __init__(self):
        self.mode = "fallback"
        self.detector = None
        self.recognizer = None
        self.yunet_path = _primera_ruta_existente(
            (
                "face_detection_yunet_2023mar.onnx",
                "face_detection_yunet_2022mar.onnx",
            )
        )
        self.sface_path = _primera_ruta_existente(("face_recognition_sface_2021dec.onnx",))
        self.cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )

        if (
            hasattr(cv2, "FaceDetectorYN_create")
            and hasattr(cv2, "FaceRecognizerSF_create")
            and self.yunet_path
            and self.sface_path
        ):
            try:
                self.detector = cv2.FaceDetectorYN_create(
                    self.yunet_path, "", (320, 320), 0.85, 0.3, 5000
                )
                self.recognizer = cv2.FaceRecognizerSF_create(self.sface_path, "")
                self.mode = "sface"
            except Exception:
                self.mode = "fallback"

        print(f"[FaceVerifier] Modo activo: {self.mode}")
        if self.mode != "sface":
            print(
                "[FaceVerifier] Para modo rapido/preciso, agrega en Modelos_IA: "
                "face_detection_yunet_2023mar.onnx y face_recognition_sface_2021dec.onnx"
            )

    def _detectar_con_sface(self, frame_rgb):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        h, w = frame_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return None

        cara = max(faces, key=lambda f: float(f[2] * f[3]))
        x, y, cw, ch = map(int, cara[:4])
        x = max(x, 0)
        y = max(y, 0)
        cw = max(min(cw, w - x), 1)
        ch = max(min(ch, h - y), 1)
        crop_rgb = frame_rgb[y : y + ch, x : x + cw]
        if crop_rgb.size == 0:
            return None

        aligned_bgr = self.recognizer.alignCrop(frame_bgr, cara)
        return {"bbox": (x, y, cw, ch), "crop_rgb": crop_rgb, "aligned_bgr": aligned_bgr}

    def _detectar_con_fallback(self, frame_rgb):
        if self.cascade.empty():
            return None

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            return None

        x, y, cw, ch = max(faces, key=lambda f: int(f[2] * f[3]))
        crop_rgb = frame_rgb[y : y + ch, x : x + cw]
        if crop_rgb.size == 0:
            return None
        return {"bbox": (x, y, cw, ch), "crop_rgb": crop_rgb, "aligned_bgr": None}

    def detectar_rostro(self, frame_rgb):
        if frame_rgb is None:
            return None
        if self.mode == "sface":
            return self._detectar_con_sface(frame_rgb)
        return self._detectar_con_fallback(frame_rgb)

    @staticmethod
    def _embedding_fallback(crop_rgb):
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (112, 112), interpolation=cv2.INTER_AREA)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False)
        bins = 16
        bin_ids = np.int32(bins * ang / (2 * np.pi))
        hist = np.zeros(bins, dtype=np.float32)
        for b, m in zip(bin_ids.flatten(), mag.flatten()):
            hist[min(int(b), bins - 1)] += float(m)
        norm = np.linalg.norm(hist)
        if norm <= 1e-6:
            return None
        return hist / norm

    def extraer_embedding(self, frame_rgb):
        deteccion = self.detectar_rostro(frame_rgb)
        if deteccion is None:
            return None, None, "No se ha detectado cara. Acerquese y mire directo a la camara."

        if self.mode == "sface":
            feat = self.recognizer.feature(deteccion["aligned_bgr"])
            if feat is None:
                return None, None, "No se pudo extraer embedding facial."
            embedding = feat.flatten().astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm <= 1e-6:
                return None, None, "Embedding facial invalido."
            embedding = embedding / norm
        else:
            embedding = self._embedding_fallback(deteccion["crop_rgb"])
            if embedding is None:
                return None, None, "No se pudo calcular descriptor facial."

        return embedding, deteccion["crop_rgb"], None

    def comparar_embeddings(self, embedding_ref, embedding_cap):
        ref = embedding_ref.astype(np.float32).flatten()
        cap = embedding_cap.astype(np.float32).flatten()
        cosine = float(np.dot(ref, cap) / ((np.linalg.norm(ref) * np.linalg.norm(cap)) + 1e-8))
        distancia = 1.0 - cosine
        umbral = FACE_THRESHOLD_SFACE if self.mode == "sface" else FACE_THRESHOLD_FALLBACK
        return distancia <= umbral, distancia, umbral


FACE_VERIFIER = FaceVerifier()


def formatear_fecha_hora(fecha_hora):
    return fecha_hora.strftime("%Y-%m-%d"), fecha_hora.strftime("%H:%M:%S")


def registrar_evento(tipo, placa, fecha_hora, estado, detalle):
    escribir_encabezado = not os.path.exists(REGISTROS_PATH)
    with open(REGISTROS_PATH, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if escribir_encabezado:
            writer.writerow(
                ["tipo", "placa", "fecha", "hora", "estado", "detalle", "timestamp_iso"]
            )
        fecha, hora = formatear_fecha_hora(fecha_hora)
        writer.writerow([tipo, placa, fecha, hora, estado, detalle, fecha_hora.isoformat()])


def cargar_embedding(ruta_placa):
    ruta_embedding = os.path.join(ruta_placa, EMBEDDING_FILE)
    if not os.path.exists(ruta_embedding):
        return None
    try:
        return np.load(ruta_embedding)
    except Exception:
        return None


def guardar_embedding(ruta_placa, embedding):
    ruta_embedding = os.path.join(ruta_placa, EMBEDDING_FILE)
    np.save(ruta_embedding, embedding.astype(np.float32))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sistema de Control de Acceso Vehicular")
        self.geometry("980x680")
        self.configure(bg="#f4f6fb")

        estilo = ttk.Style(self)
        if "clam" in estilo.theme_names():
            estilo.theme_use("clam")
        estilo.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        estilo.configure("SubTitle.TLabel", font=("Segoe UI", 11))
        estilo.configure("Action.TButton", font=("Segoe UI", 10, "bold"), padding=8)
        estilo.configure("Status.TLabel", font=("Segoe UI", 10), foreground="#334155")

        menu_frame = tk.Frame(self, bg="#e2e8f0", height=56)
        menu_frame.pack(fill="x")
        menu_frame.pack_propagate(False)

        ttk.Button(
            menu_frame,
            text="Ingreso de Vehiculos",
            style="Action.TButton",
            command=self.mostrar_ventana_ingreso_vehiculo,
        ).pack(side="left", padx=14, pady=10)
        ttk.Button(
            menu_frame,
            text="Salida de Vehiculos",
            style="Action.TButton",
            command=self.mostrar_ventana_salida_vehiculo,
        ).pack(side="left", padx=6, pady=10)

        self.container = tk.Frame(self, bg="#f4f6fb")
        self.container.pack(fill="both", expand=True, padx=12, pady=12)

        self.pagina_activa = None
        self.mostrar_pagina(VentanaIngresoVehiculo)

    def mostrar_pagina(self, nueva_pagina):
        if self.pagina_activa is not None:
            self.pagina_activa.destroy()
        self.pagina_activa = nueva_pagina(self.container)
        self.pagina_activa.pack(fill="both", expand=True)

    def mostrar_ventana_ingreso_vehiculo(self):
        self.mostrar_pagina(VentanaIngresoVehiculo)

    def mostrar_ventana_salida_vehiculo(self):
        self.mostrar_pagina(VentanaSalidaVehiculo)


class VentanaIngresoVehiculo(tk.Frame):
    def __init__(self, contenedor):
        super().__init__(contenedor, bg="#f8fafc")

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(2, weight=1)

        ttk.Label(self, text="Ingreso de Vehiculos", style="Title.TLabel").grid(
            row=0, column=0, columnspan=3, pady=(10, 4)
        )
        ttk.Label(
            self,
            text="Capture rostro y placa para registrar un ingreso autorizado.",
            style="SubTitle.TLabel",
        ).grid(row=1, column=0, columnspan=3, pady=(0, 12))

        ttk.Label(self, text="Rostro capturado").grid(row=2, column=0, pady=(0, 4))
        ttk.Label(self, text="Placa capturada").grid(row=2, column=2, pady=(0, 4))

        self.canvas_rostro_cap = tk.Canvas(self, bg="white", highlightthickness=0)
        self.canvas_rostro_cap.grid(row=3, column=0, sticky="nsew", padx=10, pady=4)

        self.canvas_placa_cap = tk.Canvas(self, bg="white", highlightthickness=0)
        self.canvas_placa_cap.grid(row=3, column=2, sticky="nsew", padx=10, pady=4)

        self.label_placa = ttk.Label(self, text="Placa extraida: --")
        self.label_placa.grid(row=4, column=0, columnspan=3, pady=(8, 2))

        self.label_fecha_hora = ttk.Label(self, text="Fecha/Hora registro: --")
        self.label_fecha_hora.grid(row=5, column=0, columnspan=3, pady=2)

        self.label_estado = ttk.Label(
            self,
            text=f"Estado: Esperando captura | Motor facial: {FACE_VERIFIER.mode}",
            style="Status.TLabel",
        )
        self.label_estado.grid(row=6, column=0, columnspan=3, pady=(2, 8))

        ttk.Button(
            self,
            text="Registrar Vehiculo",
            style="Action.TButton",
            command=self.regs,
        ).grid(row=7, column=0, columnspan=3, pady=(4, 10))

        self.Captura = Captura_camara(
            contenedor, cap1, cap2, self.canvas_rostro_cap, self.canvas_placa_cap
        )
        self.Captura.capturar()

    def regs(self):
        resultado = self.Captura.cap()
        if not resultado["ok"]:
            self.label_estado.config(text=f"Estado: {resultado['error']}")
            messagebox.showwarning("Validacion requerida", resultado["error"])
            registrar_evento("INGRESO", "N/A", datetime.now(), "RECHAZADO", resultado["error"])
            return

        txt_placa = resultado["placa"].strip()
        img_rostro = resultado["rostro_crop"]
        face_embedding = resultado["face_embedding"]
        fecha_hora = resultado["fecha_hora"]
        fecha, hora = formatear_fecha_hora(fecha_hora)

        self.label_placa.config(text=f"Placa extraida: {txt_placa}")
        self.label_fecha_hora.config(text=f"Fecha/Hora registro: {fecha} {hora}")
        self.label_estado.config(text="Estado: Ingreso registrado correctamente")

        placa_archivo = txt_placa.replace(" ", "_")
        ruta_placa = os.path.join(PLACA_DIR, placa_archivo)
        os.makedirs(ruta_placa, exist_ok=True)

        for archivo in os.listdir(ruta_placa):
            ruta_archivo = os.path.join(ruta_placa, archivo)
            if os.path.isfile(ruta_archivo):
                os.remove(ruta_archivo)

        nombre_archivo = f"capt_rostro_{placa_archivo}_{fecha_hora.strftime('%Y%m%d_%H%M%S')}.jpg"
        ruta_img = os.path.join(ruta_placa, nombre_archivo)
        guardado = cv2.imwrite(ruta_img, cv2.cvtColor(img_rostro, cv2.COLOR_RGB2BGR))

        if not guardado:
            self.label_estado.config(text="Estado: Error al guardar la imagen")
            messagebox.showerror(
                "Error de almacenamiento", "No se pudo guardar la imagen de rostro."
            )
            registrar_evento(
                "INGRESO", txt_placa, fecha_hora, "ERROR", "Fallo al guardar imagen de rostro"
            )
            return

        guardar_embedding(ruta_placa, face_embedding)
        registrar_evento("INGRESO", txt_placa, fecha_hora, "AUTORIZADO", "Registro de ingreso exitoso")
        messagebox.showinfo(
            "Ingreso registrado", f"Vehiculo {txt_placa} registrado el {fecha} a las {hora}."
        )


class VentanaSalidaVehiculo(tk.Frame):
    def __init__(self, contenedor):
        super().__init__(contenedor, bg="#f8fafc")

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(4, weight=1)

        ttk.Label(self, text="Salida de Vehiculos", style="Title.TLabel").grid(
            row=0, column=0, columnspan=3, pady=(10, 4)
        )
        ttk.Label(
            self,
            text="Se valida el rostro capturado contra el registro de ingreso.",
            style="SubTitle.TLabel",
        ).grid(row=1, column=0, columnspan=3, pady=(0, 12))

        ttk.Label(self, text="Rostro capturado").grid(row=2, column=0, pady=(0, 4))
        ttk.Label(self, text="Rostro registrado").grid(row=2, column=2, pady=(0, 4))
        ttk.Label(self, text="Placa capturada").grid(row=4, column=0, pady=(10, 4))

        self.canvas_rostro_cap = tk.Canvas(
            self, bg="white", height=220, width=320, highlightthickness=0
        )
        self.canvas_rostro_cap.grid(row=3, column=0, padx=10, sticky="nsew")

        self.canvas_placa_comp = tk.Canvas(
            self, bg="white", height=220, width=320, highlightthickness=0
        )
        self.canvas_placa_comp.grid(row=3, column=2, padx=10, sticky="nsew")

        self.canvas_placa_cap = tk.Canvas(
            self, bg="white", height=220, width=320, highlightthickness=0
        )
        self.canvas_placa_cap.grid(row=5, column=0, padx=10, sticky="nsew")

        self.label_placa = ttk.Label(self, text="Placa extraida: --")
        self.label_placa.grid(row=5, column=2, pady=(0, 4))

        self.label_fecha_hora = ttk.Label(self, text="Fecha/Hora validacion: --")
        self.label_fecha_hora.grid(row=6, column=0, columnspan=3, pady=2)

        self.label_estado = ttk.Label(
            self,
            text=f"Estado: Esperando validacion | Motor facial: {FACE_VERIFIER.mode}",
            style="Status.TLabel",
        )
        self.label_estado.grid(row=7, column=0, columnspan=3, pady=(2, 8))

        ttk.Button(
            self,
            text="Registrar Salida",
            style="Action.TButton",
            command=self.salida,
        ).grid(row=8, column=0, columnspan=3, pady=(4, 10))

        self.Captura = Captura_camara(
            contenedor, cap1, cap2, self.canvas_rostro_cap, self.canvas_placa_cap
        )
        self.Captura.capturar()
        self.imgtk_rostro_comp = None

    def mostrar_rostro_registrado(self, ruta_img):
        img = Image.open(ruta_img).convert("RGB")
        width = max(self.canvas_placa_comp.winfo_width(), 1)
        height = max(self.canvas_placa_comp.winfo_height(), 1)
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        self.imgtk_rostro_comp = ImageTk.PhotoImage(image=img)
        self.canvas_placa_comp.delete("all")
        self.canvas_placa_comp.create_image(0, 0, anchor=tk.NW, image=self.imgtk_rostro_comp)

    def salida(self):
        resultado = self.Captura.cap()
        if not resultado["ok"]:
            self.label_estado.config(text=f"Estado: {resultado['error']}")
            messagebox.showwarning("Validacion requerida", resultado["error"])
            registrar_evento("SALIDA", "N/A", datetime.now(), "RECHAZADO", resultado["error"])
            return

        txt_placa = resultado["placa"].strip()
        face_embedding_cap = resultado["face_embedding"]
        fecha_hora = resultado["fecha_hora"]
        fecha, hora = formatear_fecha_hora(fecha_hora)

        self.label_placa.config(text=f"Placa extraida: {txt_placa}")
        self.label_fecha_hora.config(text=f"Fecha/Hora validacion: {fecha} {hora}")

        placa_archivo = txt_placa.replace(" ", "_")
        ruta_placa = os.path.join(PLACA_DIR, placa_archivo)

        if not os.path.exists(ruta_placa):
            mensaje = "No hay registro de ingreso para la placa detectada."
            self.label_estado.config(text=f"Estado: {mensaje}")
            messagebox.showwarning("Salida denegada", mensaje)
            registrar_evento("SALIDA", txt_placa, fecha_hora, "RECHAZADO", mensaje)
            return

        rostros_registrados = sorted(
            [
                archivo
                for archivo in os.listdir(ruta_placa)
                if archivo.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        if not rostros_registrados:
            mensaje = "No se encontro rostro registrado para esa placa."
            self.label_estado.config(text=f"Estado: {mensaje}")
            messagebox.showwarning("Salida denegada", mensaje)
            registrar_evento("SALIDA", txt_placa, fecha_hora, "RECHAZADO", mensaje)
            return

        ruta_rostro_registrado = os.path.join(ruta_placa, rostros_registrados[0])
        self.mostrar_rostro_registrado(ruta_rostro_registrado)

        embedding_ref = cargar_embedding(ruta_placa)
        if embedding_ref is None:
            img_ref_bgr = cv2.imread(ruta_rostro_registrado)
            if img_ref_bgr is None:
                mensaje = "No se pudo cargar el rostro registrado."
                self.label_estado.config(text=f"Estado: {mensaje}")
                messagebox.showerror("Error de validacion", mensaje)
                registrar_evento("SALIDA", txt_placa, fecha_hora, "ERROR", mensaje)
                return
            img_ref_rgb = cv2.cvtColor(img_ref_bgr, cv2.COLOR_BGR2RGB)
            embedding_ref, _, error = FACE_VERIFIER.extraer_embedding(img_ref_rgb)
            if error is not None:
                mensaje = f"Rostro registrado invalido: {error}"
                self.label_estado.config(text=f"Estado: {mensaje}")
                messagebox.showwarning("Salida denegada", mensaje)
                registrar_evento("SALIDA", txt_placa, fecha_hora, "RECHAZADO", mensaje)
                return
            guardar_embedding(ruta_placa, embedding_ref)

        verificado, distancia, umbral = FACE_VERIFIER.comparar_embeddings(
            embedding_ref, face_embedding_cap
        )
        detalle_distancia = f"dist={distancia:.4f} umbral={umbral:.4f}"

        if verificado:
            self.label_estado.config(text=f"Estado: Salida autorizada ({detalle_distancia})")
            messagebox.showinfo("Salida autorizada", f"Vehiculo {txt_placa} validado y liberado.")
            registrar_evento(
                "SALIDA",
                txt_placa,
                fecha_hora,
                "AUTORIZADO",
                f"Rostro verificado ({detalle_distancia})",
            )
            shutil.rmtree(ruta_placa)
            return

        mensaje = f"Rostro no coincide con el registro ({detalle_distancia})."
        self.label_estado.config(text=f"Estado: {mensaje}")
        messagebox.showwarning("Salida denegada", mensaje)
        registrar_evento("SALIDA", txt_placa, fecha_hora, "RECHAZADO", mensaje)


class Captura_camara:
    def __init__(self, contenedor, cap1, cap2, canv1, canv2):
        self.cap_rostro = None
        self.cap_placa = None
        self.contenedor = contenedor
        self.canvas_rostro_cap = canv1
        self.canvas_placa_cap = canv2
        self.cam_encendido = True
        self.after_id = None

    def capturar(self):
        self.hilo_rostro = threading.Thread(target=self.cap_camara, args=(cap1, 0), daemon=True)
        self.hilo_placa = threading.Thread(target=self.cap_camara, args=(cap2, 1), daemon=True)
        self.hilo_placa.start()
        self.hilo_rostro.start()
        self.actualizar_frame()

    def cap_camara(self, cap, index):
        while self.cam_encendido:
            ret, cap_cam = cap.read()
            if ret:
                frame_cap = cv2.cvtColor(cap_cam, cv2.COLOR_BGR2RGB)
                if index == 0:
                    self.cap_rostro = frame_cap
                else:
                    self.cap_placa = frame_cap

    def actualizar_frame(self):
        if self.cap_rostro is not None:
            if self.canvas_rostro_cap.winfo_exists():
                canvas_width_rostro = self.canvas_rostro_cap.winfo_width()
                canvas_height_rostro = self.canvas_rostro_cap.winfo_height()
            else:
                if self.after_id is not None:
                    self.contenedor.after_cancel(self.after_id)
                return

            frame_rostro = cv2.resize(
                self.cap_rostro,
                (canvas_width_rostro, canvas_height_rostro),
                interpolation=cv2.INTER_AREA,
            )
            img_rostro = Image.fromarray(frame_rostro)
            imgtk_rostro = ImageTk.PhotoImage(image=img_rostro)
            self.canvas_rostro_cap.create_image(0, 0, anchor=tk.NW, image=imgtk_rostro)
            self.canvas_rostro_cap.imgtk = imgtk_rostro

        if self.cap_placa is not None:
            if self.canvas_placa_cap.winfo_exists():
                canvas_width_placa = self.canvas_placa_cap.winfo_width()
                canvas_height_placa = self.canvas_placa_cap.winfo_height()
            else:
                if self.after_id is not None:
                    self.contenedor.after_cancel(self.after_id)
                return

            frame_placa = cv2.resize(
                self.cap_placa,
                (canvas_width_placa, canvas_height_placa),
                interpolation=cv2.INTER_AREA,
            )
            img_placa = Image.fromarray(frame_placa)
            imgtk_placa = ImageTk.PhotoImage(image=img_placa)
            self.canvas_placa_cap.create_image(0, 0, anchor=tk.NW, image=imgtk_placa)
            self.canvas_placa_cap.imgtk = imgtk_placa

        self.after_id = self.contenedor.after(10, self.actualizar_frame)

    def atencion(self, img, modelo_placas):
        h, w, _ = img.shape
        tile_size = 700
        overlap = 200
        step = tile_size - overlap
        placas = []

        for y in range(0, h, step):
            for x in range(0, w, step):
                crop = img[y : y + tile_size, x : x + tile_size]
                results = modelo_placas(crop, save=False, verbose=False)[0]

                for box in results.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    x1 += x
                    x2 += x
                    y1 += y
                    y2 += y
                    placas.append([y1, y2, x1, x2])
        return placas

    def cap(self):
        if self.cap_rostro is None or self.cap_placa is None:
            return {"ok": False, "error": "No hay imagen disponible de las camaras."}

        face_embedding, face_crop, error_rostro = FACE_VERIFIER.extraer_embedding(self.cap_rostro)
        if error_rostro is not None:
            return {"ok": False, "error": error_rostro}

        ruta_capt_placa = os.path.join(STATIC_DIR, "capt_placa.jpg")
        cv2.imwrite(ruta_capt_placa, cv2.cvtColor(self.cap_placa, cv2.COLOR_RGB2BGR))

        try:
            imagen_placa = cv2.cvtColor(cv2.imread(ruta_capt_placa), cv2.COLOR_BGR2RGB)
            placas = self.atencion(imagen_placa, zzz)
            if not placas:
                return {
                    "ok": False,
                    "error": "No se detecto una placa valida en la imagen actual.",
                }

            y1 = [p[0] for p in placas]
            y2 = [p[1] for p in placas]
            x1 = [p[2] for p in placas]
            x2 = [p[3] for p in placas]
            img_recortada = imagen_placa[min(y1) : max(y2), min(x1) : max(x2)]

            placa = ss.extrac_placa(img_recortada, zzz)
            if len(placa) == 0:
                return {"ok": False, "error": "La placa fue detectada pero no se pudo extraer."}

            txt_placa = ss.extrac_caracteres(placa, xxx).strip()
            if not txt_placa:
                return {"ok": False, "error": "No se pudo leer el texto de la placa."}

            return {
                "ok": True,
                "placa": txt_placa,
                "rostro": self.cap_rostro.copy(),
                "rostro_crop": face_crop.copy(),
                "face_embedding": face_embedding.copy(),
                "fecha_hora": datetime.now(),
            }
        finally:
            if os.path.exists(ruta_capt_placa):
                os.remove(ruta_capt_placa)


def centrar_ventana(ventana, ancho, alto):
    pantalla_ancho = ventana.winfo_screenwidth()
    pantalla_alto = ventana.winfo_screenheight()
    x = (pantalla_ancho // 2) - (ancho // 2)
    y = (pantalla_alto // 2) - (alto // 2)
    ventana.geometry(f"{ancho}x{alto}+{x}+{y-30}")


if __name__ == "__main__":
    app = App()
    centrar_ventana(app, 980, 680)
    app.mainloop()
