import cv2
import numpy as np
from ultralytics import YOLO
import gradio as gr

# 1) Cargar modelo YOLO
# Para pruebas: usa un modelo general. Cuando tengas tu modelo propio,
# cambia "yolov8n.pt" por "models/contenedores.pt"
model = YOLO("yolov8n.pt")

# Nombre de la clase que usaremos como "contenedor".
# Cuando entrenes tu modelo propio, normalmente será class 0 -> "container"
CONTAINER_CLASS_NAME = "container"   # ajústalo cuando entrenes tu modelo

def contar_contenedores(img):
    """
    Recibe una imagen (PIL) desde Gradio, detecta contenedores
    y devuelve la imagen anotada + cantidad.
    """
    # Convertir PIL -> numpy (BGR para OpenCV)
    img = np.array(img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 2) Correr detección
    results = model(img_bgr)[0]

    conteo = 0
    for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = box.tolist()
        cls_id = int(cls_id.item())
        cls_name = model.names.get(cls_id, str(cls_id))

        # Si todavía no tienes modelo entrenado,
        # puedes dejar esto como conteo total: conteo += 1
        # y ver qué clases te está detectando.
        if cls_name == CONTAINER_CLASS_NAME:
            conteo += 1
            # Dibujar recuadro verde
            cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                img_bgr, cls_name,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

    # Volver a RGB para mostrar
    img_out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    texto = f"Contenedores detectados: {conteo}"
    return img_out, texto

# 3) Interfaz con Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Contador de contenedores en imágenes satelitales")
    gr.Markdown(
        "Sube un pantallazo de Google Earth **recortado justo a la zona roja**. "
        "El modelo marcará y contará los contenedores detectados."
    )

    with gr.Row():
        input_img = gr.Image(type="pil", label="Imagen de entrada")
        output_img = gr.Image(type="numpy", label="Detecciones")
    output_text = gr.Textbox(label="Resultado", interactive=False)

    btn = gr.Button("Contar contenedores")

    btn.click(
        contar_contenedores,
        inputs=input_img,
        outputs=[output_img, output_text],
    )

# 4) Lanzar app
if __name__ == "__main__":
    demo.launch()
