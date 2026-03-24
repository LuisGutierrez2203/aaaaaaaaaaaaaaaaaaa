print("Cargando dependencias :..................")
import tkinter as tk
from tkinter import ttk

from tkinter import ttk, filedialog
from PIL import Image, ImageTk

from ultralytics import YOLO


import sis as ss

from deepface import DeepFace

import os
import cv2

import threading
import shutil

print("Cargando modelos de IA .............")

base = os.path.dirname(os.path.abspath(__file__))  # /Sistema
raiz = os.path.abspath(os.path.join(base, ".."))  # /CodigoRaspberry

zzz = YOLO(os.path.join(raiz, "Modelos_IA", "Extrac_placas.pt"))
xxx = YOLO(os.path.join(raiz, "Modelos_IA", "Segment_caracteres_gray4.pt"))

carpetas = ["Placa", "static"]
for carpeta in carpetas:
    os.makedirs(carpeta, exist_ok=True)


cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Menú dinámico")
        self.geometry("800x600")

        # --- Menú superior ---
        menu_frame = tk.Frame(self)
        menu_frame.pack(fill="x")

        tk.Button(
            menu_frame,
            text="INGRESO DE VEHICULOS",
            command=self.mostrar_ventana_ingreso_vehiculo,
        ).pack(side="left", padx=10, pady=5)
        tk.Button(
            menu_frame,
            text="SALIDA DE VEHICULOS",
            command=self.mostrar_ventana_salida_vehiculo,
        ).pack(side="right", padx=10, pady=5)

        # --- Contenedor principal ---
        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)

        # Página inicial
        self.pagina_activa = None

        self.mostrar_pagina(VentanaIngresoVehiculo)

    # --- Función para limpiar y cambiar de página ---
    def mostrar_pagina(self, nueva_pagina):
        # Eliminar contenido anterior
        if self.pagina_activa is not None:
            self.pagina_activa.destroy()

        # Mostrar la nueva página dentro del contenedor
        self.pagina_activa = nueva_pagina(self.container)
        self.pagina_activa.pack(fill="both", expand=True)

    # --- Páginas diferentes ---
    def mostrar_ventana_ingreso_vehiculo(self):
        self.mostrar_pagina(VentanaIngresoVehiculo)

    def mostrar_ventana_salida_vehiculo(self):
        self.mostrar_pagina(VentanaSalidaVehiculo)


class VentanaIngresoVehiculo(tk.Frame):
    def __init__(self, contenedor):
        super().__init__(contenedor, bg="lightblue")

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=3)
        self.grid_rowconfigure(3, weight=1)
        self.grid_rowconfigure(4, weight=1)

        tk.Label(self, text="Ingreso de Vehiculos", font=("Arial", 16)).grid(
            row=0, column=0, columnspan=3
        )

        tk.Label(self, text="Rostro Capturado", font=("Arial", 10)).grid(
            row=1, column=0
        )

        tk.Label(self, text="Placa Capturada", font=("Arial", 10)).grid(row=1, column=2)

        self.label_placa = tk.Label(
            self, text="Placa Extraida: xxx - xxx", font=("Arial", 10)
        )
        self.label_placa.grid(row=3, column=2)

        self.canvas_rostro_cap = tk.Canvas(self, bg="white")
        self.canvas_rostro_cap.grid(row=2, column=0, sticky="nsew", padx=10)

        self.canvas_placa_cap = tk.Canvas(self, bg="white")
        self.canvas_placa_cap.grid(row=2, column=2, sticky="nsew", padx=10)

        tk.Button(
            self, text="Registrar Vehiculo", font=("Arial", 12), command=self.regs
        ).grid(row=4, column=0, columnspan=3)

        self.Captura = Captura_camara(
            contenedor, cap1, cap2, self.canvas_rostro_cap, self.canvas_placa_cap
        )
        self.Captura.capturar()

    def regs(self):
        self.result = self.Captura.cap()
        if len(self.result) != 0:
            
            self.txt_placa = self.result[0]
            self.img_rostro = self.result[1]
            self.label_placa.config(text="Placa Extraida: " + self.txt_placa)

            # ruta = "Placa/" + self.txt_placa.replace(" ", "_")
            self.ruta = os.path.join(base, "Placa", self.txt_placa.replace(" ", "_"))
            print(self.ruta)
            if not os.path.exists(self.ruta):
                os.makedirs(self.ruta)
                cv2.imwrite(
                    self.ruta + "/capt_rostro" + self.txt_placa.replace(" ", "_") + ".jpg",
                    self.img_rostro,
                )
                print("📁 Carpeta creada correctamente")

            else:
                print("📂 La carpeta ya existía")
        else:
            print("Esto es serio goku")


class VentanaSalidaVehiculo(tk.Frame):
    def __init__(self, contenedor):
        super().__init__(contenedor, bg="lightgreen")
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=2)
        self.grid_rowconfigure(3, weight=1)
        self.grid_rowconfigure(4, weight=2)

        self.grid_propagate(False)
        tk.Label(self, text="Salida Vehiculos", font=("Arial", 16)).grid(
            row=0, column=0, columnspan=3
        )

        tk.Label(self, text="Rostro Capturado", font=("Arial", 10)).grid(
            row=1, column=0, pady=10
        )

        self.canvas_rostro_cap = tk.Canvas(self, bg="white", height=200, width=300)
        self.canvas_rostro_cap.grid(row=2, column=0, padx=10, columnspan=1)

        tk.Label(self, text="Placa Capturada", font=("Arial", 10)).grid(
            row=3, column=0, pady=10
        )

        self.canvas_placa_cap = tk.Canvas(self, bg="white", height=200, width=300)
        self.canvas_placa_cap.grid(row=4, column=0, padx=10)

        tk.Label(self, text="Rostro Compatible", font=("Arial", 10)).grid(
            row=1, column=2
        )

        self.canvas_placa_comp = tk.Canvas(self, bg="white", height=200, width=300)
        self.canvas_placa_comp.grid(row=2, column=2, padx=10)

        self.label_placa = tk.Label(
            self, text="Placa Extraida: xxx - xxx", font=("Arial", 10)
        )
        self.label_placa.grid(row=3, column=2)

        self.Captura = Captura_camara(
            contenedor, cap1, cap2, self.canvas_rostro_cap, self.canvas_placa_cap
        )
        self.Captura.capturar()

        tk.Button(
            self, text="Registrar Vehiculo", font=("Arial", 12), command=self.salida
        ).grid(row=4, column=2, columnspan=3)

    def salida(self):
        self.result = self.Captura.cap()
        if len(self.result) != 0:
            self.txt_placa = self.result[0]
            self.img_rostro = self.result[1]

            cv2.imwrite(os.path.join(base, "static") + "/cap_rostro.jpg", self.img_rostro)

            self.label_placa.config(text="Placa Extraida: " + self.txt_placa)

            self.ruta = os.path.join(
                base, "Placa", self.txt_placa.replace(" ", "_")
            )  # "Placa/" + self.txt_placa.replace(" ", "_")

            if os.path.exists(self.ruta):
                self.ruta_rostro_registrado = os.listdir(self.ruta)
                print(self.ruta_rostro_registrado)
                self.ruta_rostro_capt = os.path.join(
                    base, "static", "cap_rostro.jpg"
                )  # "./static/capt_rostro.jpg"

                if len(self.ruta_rostro_registrado) == 1:
                    print(os.path.join(self.ruta, self.ruta_rostro_registrado[0]))
                    valid_rostro = DeepFace.verify(
                        img1_path=os.path.join(self.ruta, self.ruta_rostro_registrado[0]),
                        img2_path=self.ruta_rostro_capt,
                        model_name="Facenet",
                        detector_backend="mtcnn",
                    )

                    if valid_rostro.get("verified"):
                        print("Rostro compatibles")
                        os.remove(self.ruta_rostro_capt)
                        shutil.rmtree(self.ruta)
            else:
                print("Joa se jodio esto")


class Captura_camara:
    def __init__(self, contenedor, cap1, cap2, canv1, canv2):
        self.cap_rostro = None
        self.cap_placa = None
        self.contenedor = contenedor
        self.canvas_rostro_cap = canv1
        self.canvas_placa_cap = canv2
        self.cam_encendido = True

    def capturar(self):
        self.hilo_rostro = threading.Thread(
            target=self.cap_camara, args=(cap1, 0), daemon=True
        )
        self.hilo_placa = threading.Thread(
            target=self.cap_camara, args=(cap2, 1), daemon=True
        )

        self.hilo_placa.start()
        self.hilo_rostro.start()

        self.actualizar_frame()

    def cap_camara(self, cap, index):
        while self.cam_encendido:
            self.ret, self.cap_cam = cap.read()
            if self.ret:
                # self.frame_cap = cv2.flip(self.cap_cam, 1)
                self.frame_cap = cv2.cvtColor(self.cap_cam, cv2.COLOR_BGR2RGB)

                if index == 0:
                    self.cap_rostro = self.frame_cap

                else:
                    self.cap_placa = self.frame_cap

    def actualizar_frame(self):
        if self.cap_rostro is not None:

            if self.canvas_rostro_cap.winfo_exists():
                self.canvas_width_rostro = self.canvas_rostro_cap.winfo_width()
                self.canvas_height_rostro = self.canvas_rostro_cap.winfo_height()
            else:
                self.contenedor.after_cancel(self.after_id)
                return

            # Redimensionar el frame al tamaño del canvas
            self.frame_rostro = cv2.resize(
                self.cap_rostro,
                (self.canvas_width_rostro, self.canvas_height_rostro),
                interpolation=cv2.INTER_AREA,
            )
            # Convertir a ImageTk
            self.img_rostro = Image.fromarray(self.frame_rostro)
            self.imgtk_rostro = ImageTk.PhotoImage(image=self.img_rostro)

            # Pintar en el Canvas
            self.canvas_rostro_cap.create_image(
                0, 0, anchor=tk.NW, image=self.imgtk_rostro
            )
            self.canvas_rostro_cap.imgtk = self.imgtk_rostro

        if self.cap_placa is not None:

            if self.canvas_placa_cap.winfo_exists():
                # Obtener dimensiones del canvas
                self.canvas_width_placa = self.canvas_placa_cap.winfo_width()
                self.canvas_height_placa = self.canvas_placa_cap.winfo_height()

            else:
                self.contenedor.after_cancel(self.after_id)
                return

            self.frame_placa = cv2.resize(
                self.cap_placa,
                (self.canvas_width_placa, self.canvas_height_placa),
                interpolation=cv2.INTER_AREA,
            )

            # Convertir a ImageTk
            self.img_placa = Image.fromarray(self.frame_placa)
            self.imgtk_placa = ImageTk.PhotoImage(image=self.img_placa)

            # Pintar en el Canvas
            self.canvas_placa_cap.create_image(
                0, 0, anchor=tk.NW, image=self.imgtk_placa
            )
            self.canvas_placa_cap.imgtk = self.imgtk_placa

        # Llamar de nuevo después de 10ms
        self.after_id = self.contenedor.after(10, self.actualizar_frame)

    def atencion(self, img, zzz):
        h, w, _ = img.shape
        tile_size = 700
        overlap = 200
        step = tile_size - overlap
        placas = []

        for y in range(0, h, step):
            for x in range(0, w, step):

                crop = img[y:y+tile_size, x:x+tile_size]
                results = zzz(crop, save=False, verbose=False)[0]
                

                for r in results:
                    boxes = r.boxes.xyxy

                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)

                        # ajustar coordenadas a imagen original
                        x1 = x1 + x
                        x2 = x2 + x
                        y1 = y1 + y
                        y2 = y2 + y

                        

                        placas.append([y1, y2, x1, x2])
        return placas

    def cap(self):
        cv2.imwrite("./static/capt_placa.jpg", self.cap_placa)

        imagen_placa = cv2.cvtColor(
            cv2.imread("./static/capt_placa.jpg"), cv2.COLOR_BGR2RGB
        )

        placas = self.atencion(imagen_placa, zzz)

        y1 = []
        y2 = []
        x1 = []
        x2 = []
        if len(placas) != 0:
            for p in placas:
                y1.append(p[0])
                y2.append(p[1])
                x1.append(p[2])
                x2.append(p[3])
                
            img_recortada = imagen_placa[min(y1):max(y2), min(x1):max(x2)]

        
            placa = ss.extrac_placa(img_recortada, zzz)



            if len(placa) > 0:
                txt_placa = ss.extrac_caracteres(placa, xxx)

                print(txt_placa)
                os.remove("./static/capt_placa.jpg")

                return [txt_placa, self.cap_rostro]

        os.remove("./static/capt_placa.jpg")
        return []


def centrar_ventana(ventana, ancho, alto):
    # Obtener dimensiones de la pantalla
    pantalla_ancho = ventana.winfo_screenwidth()

    pantalla_alto = ventana.winfo_screenheight()

    # Calcular coordenadas para centrar
    x = (pantalla_ancho // 2) - (ancho // 2)
    y = (pantalla_alto // 2) - (alto // 2)

    # Asignar geometría
    ventana.geometry(f"{ancho}x{alto}+{x}+{y-30}")


# --- Ejecutar app ---
if __name__ == "__main__":
    app = App()

    centrar_ventana(app, 800, 600)
    app.mainloop()
