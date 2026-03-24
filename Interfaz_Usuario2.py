print("Cargando dependencias :..................")

import csv
import os
import shutil
import threading
from datetime import datetime

import cv2
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
from deepface import DeepFace
from ultralytics import YOLO

import sis as ss

print("Cargando modelos de IA .............")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

PLACA_DIR = os.path.join(BASE_DIR, "Placa")
STATIC_DIR = os.path.join(BASE_DIR, "static")
REGISTROS_PATH = os.path.join(BASE_DIR, "registros_acceso.csv")

for carpeta in (PLACA_DIR, STATIC_DIR):
    os.makedirs(carpeta, exist_ok=True)

zzz = YOLO(os.path.join(ROOT_DIR, "Modelos_IA", "Extrac_placas.pt"))
xxx = YOLO(os.path.join(ROOT_DIR, "Modelos_IA", "Segment_caracteres_gray4.pt"))

cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)


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

        self.label_estado = ttk.Label(self, text="Estado: Esperando captura", style="Status.TLabel")
        self.label_estado.grid(row=6, column=0, columnspan=3, pady=(2, 8))

        ttk.Button(
            self,
            text="Registrar Vehiculo",
            style="Action.TButton",
            command=self.regs,
        ).grid(row=7, column=0, columnspan=3, pady=(4, 10))

        self.Captura = Captura_camara(contenedor, cap1, cap2, self.canvas_rostro_cap, self.canvas_placa_cap)
        self.Captura.capturar()

    def regs(self):
        resultado = self.Captura.cap()
        if not resultado["ok"]:
            self.label_estado.config(text=f"Estado: {resultado['error']}")
            messagebox.showwarning("Validacion requerida", resultado["error"])
            registrar_evento("INGRESO", "N/A", datetime.now(), "RECHAZADO", resultado["error"])
            return

        txt_placa = resultado["placa"].strip()
        img_rostro = resultado["rostro"]
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
            messagebox.showerror("Error de almacenamiento", "No se pudo guardar la imagen de rostro.")
            registrar_evento("INGRESO", txt_placa, fecha_hora, "ERROR", "Fallo al guardar imagen de rostro")
            return

        registrar_evento("INGRESO", txt_placa, fecha_hora, "AUTORIZADO", "Registro de ingreso exitoso")
        messagebox.showinfo("Ingreso registrado", f"Vehiculo {txt_placa} registrado el {fecha} a las {hora}.")


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

        self.canvas_rostro_cap = tk.Canvas(self, bg="white", height=220, width=320, highlightthickness=0)
        self.canvas_rostro_cap.grid(row=3, column=0, padx=10, sticky="nsew")

        self.canvas_placa_comp = tk.Canvas(self, bg="white", height=220, width=320, highlightthickness=0)
        self.canvas_placa_comp.grid(row=3, column=2, padx=10, sticky="nsew")

        self.canvas_placa_cap = tk.Canvas(self, bg="white", height=220, width=320, highlightthickness=0)
        self.canvas_placa_cap.grid(row=5, column=0, padx=10, sticky="nsew")

        self.label_placa = ttk.Label(self, text="Placa extraida: --")
        self.label_placa.grid(row=5, column=2, pady=(0, 4))

        self.label_fecha_hora = ttk.Label(self, text="Fecha/Hora validacion: --")
        self.label_fecha_hora.grid(row=6, column=0, columnspan=3, pady=2)

        self.label_estado = ttk.Label(self, text="Estado: Esperando validacion", style="Status.TLabel")
        self.label_estado.grid(row=7, column=0, columnspan=3, pady=(2, 8))

        ttk.Button(
            self,
            text="Registrar Salida",
            style="Action.TButton",
            command=self.salida,
        ).grid(row=8, column=0, columnspan=3, pady=(4, 10))

        self.Captura = Captura_camara(contenedor, cap1, cap2, self.canvas_rostro_cap, self.canvas_placa_cap)
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
        img_rostro = resultado["rostro"]
        fecha_hora = resultado["fecha_hora"]
        fecha, hora = formatear_fecha_hora(fecha_hora)

        self.label_placa.config(text=f"Placa extraida: {txt_placa}")
        self.label_fecha_hora.config(text=f"Fecha/Hora validacion: {fecha} {hora}")

        placa_archivo = txt_placa.replace(" ", "_")
        ruta_placa = os.path.join(PLACA_DIR, placa_archivo)

        ruta_rostro_capt = os.path.join(
            STATIC_DIR, f"cap_rostro_{fecha_hora.strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        cv2.imwrite(ruta_rostro_capt, cv2.cvtColor(img_rostro, cv2.COLOR_RGB2BGR))

        try:
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

            valid_rostro = DeepFace.verify(
                img1_path=ruta_rostro_registrado,
                img2_path=ruta_rostro_capt,
                model_name="Facenet",
                detector_backend="mtcnn",
                enforce_detection=False,
            )

            if valid_rostro.get("verified"):
                self.label_estado.config(text="Estado: Salida autorizada")
                messagebox.showinfo("Salida autorizada", f"Vehiculo {txt_placa} validado y liberado.")
                registrar_evento(
                    "SALIDA",
                    txt_placa,
                    fecha_hora,
                    "AUTORIZADO",
                    "Rostro verificado correctamente",
                )
                shutil.rmtree(ruta_placa)
                return

            mensaje = "Rostro no coincide con el registro de ingreso."
            self.label_estado.config(text=f"Estado: {mensaje}")
            messagebox.showwarning("Salida denegada", mensaje)
            registrar_evento("SALIDA", txt_placa, fecha_hora, "RECHAZADO", mensaje)
        finally:
            if os.path.exists(ruta_rostro_capt):
                os.remove(ruta_rostro_capt)


class Captura_camara:
    def __init__(self, contenedor, cap1, cap2, canv1, canv2):
        self.cap_rostro = None
        self.cap_placa = None
        self.contenedor = contenedor
        self.canvas_rostro_cap = canv1
        self.canvas_placa_cap = canv2
        self.cam_encendido = True
        self.after_id = None
        self.face_detector = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )

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

    def detectar_rostro(self):
        if self.cap_rostro is None or self.face_detector.empty():
            return False
        gray = cv2.cvtColor(self.cap_rostro, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        rostros = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        return len(rostros) > 0

    def cap(self):
        if self.cap_rostro is None or self.cap_placa is None:
            return {"ok": False, "error": "No hay imagen disponible de las camaras."}

        if not self.detectar_rostro():
            return {
                "ok": False,
                "error": "No se ha detectado cara. Acerquese y mire directo a la camara.",
            }

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
