import os
import cv2
from rembg import remove
import numpy as np

CaraClasificador = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
class RemoverFondo:
    #Creacion de carpetas
    def __init__(self, input_folder, output_base_folder):
        self.input_folder = input_folder
        self.output_folder = os.path.join(output_base_folder, 'Sinfondo')
        self.faces_folder = os.path.join(output_base_folder, 'Infantil')
        # Asegurarse de que las carpetas existen
        os.makedirs(self.input_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.faces_folder, exist_ok=True)

    #Proceso general
    def procesar_imagenes(self):
        for file in os.listdir(self.input_folder):
            if file.endswith((".png", ".jpg", ".jpeg")):
                input_path = os.path.join(self.input_folder, file)
                output_path = os.path.join(self.output_folder, file)
                faces_output_path = os.path.join(self.faces_folder, file.replace('.jpg', '_rostro.jpg'))
                self.remover(input_path, output_path, faces_output_path)

    #Remover el fondo por primera vez para eliminar mas de 1 cara
    def remover(self, input_path, output_path, faces_output_path):
        with open(input_path, 'rb') as inp:
            contenido = inp.read()
            if len(contenido) == 0:
                print(f"El archivo {input_path} está vacío o corrupto.")
                return
            fondo_salida = remove(contenido)
            with open(output_path, 'wb') as outp:
                outp.write(fondo_salida)
            self.infantil(fondo_salida, faces_output_path)

    #Buscarar la cara y acomodarla a tamaño infantil de la imagen ya sin fondo
    def infantil(self, fondo_salida, faces_output_path):
        image = cv2.imdecode(np.frombuffer(fondo_salida, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        caras = CaraClasificador.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in caras:
            # Centro de la cara
            centro_x = x + w // 2
            centro_y = y + h // 2

            # Limites de infantil
            lado_max = int(max(w, h) * (2.5))
            x_inicio = max(centro_x - lado_max // 2, 0)
            y_inicio = max(centro_y - lado_max // 2, 0)
            x_final = min(x_inicio + lado_max, image.shape[1])
            y_final = min(y_inicio + lado_max, image.shape[0])

            # Recortar y redimensionar la imagen
            rostro = image[y_inicio:y_final, x_inicio:x_final]
            rostro = cv2.resize(rostro, (600, 600), interpolation=cv2.INTER_CUBIC)

            # Manejo de brillo
            rostro_ajustado = np.clip(rostro * 0.9, 0, 255).astype(np.uint8)

            cv2.imwrite(faces_output_path, rostro_ajustado)
            self.remover2(faces_output_path, faces_output_path)
            break  # Solo procesa el primer rostro encontrado para simplificar

    #Quitar el fondo negro del recorte
    def remover2(self, input_path, output_path):
        with open(input_path, 'rb') as inp:
            contenido = inp.read()
            if len(contenido) == 0:
                print(f"El archivo {input_path} está vacío o corrupto.")
                return
            fondo_salida = remove(contenido)
            with open(output_path, 'wb') as outp:
                outp.write(fondo_salida)


if __name__ == "__main__":
    input_folder = "input"
    output_base_folder = "output"

    remover = RemoverFondo(input_folder, output_base_folder)
    remover.procesar_imagenes()
