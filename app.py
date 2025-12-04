import cv2
import numpy as np
import gradio as gr

from inference_sdk import InferenceHTTPClient
from config import (
    ROBOFLOW_API_KEY,
    ROBOFLOW_WORKSPACE,
    ROBOFLOW_WORKFLOW_ID,
)

# Crear cliente Roboflow
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

def detectar_contenedores(img):
    """
    Recibe una imagen PIL, la convierte a BGR, la envía a Roboflow
    y devuelve las detecciones + conteo.
    """

    # PIL → numpy RGB
    img_np = np.array(img)

    # Guardar temporalmente como JPG en memoria (Roboflow necesita archivo)
    # Pero NO guardamos en disco, hacemos encode en RAM:
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    if not ok:
        raise Exception("Error al convertir imagen a JPG")

    # Ejecutar Workflow en Roboflow
    result = client.run_workflow(
        workspace_name=ROBOFLOW_WORKSPACE,
        workflow_id=ROBOFLOW_WORKFLOW_ID,
        images={
            "image": buffer.tobytes()
        },
        use_cache=True
    )

    # --- Interpretar predicciones ---
    predictions = result["predictions"]
    img_out = img_np.copy()

    conteo = 0

    for p in predictions:
        x, y = p["x"], p["y"]
        w, h = p["width"], p["height"]
        conf = p.get("confidence", 0)

        # Convertir centro → esquinas
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)

        conteo += 1

        # Dibujar bounding box
        cv2.rectangle(
            img_out,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            img_out,
            f"{conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return img_out, f"Contenedores detectados: {conteo}"

# ----- Interfaz Gradio -----

with gr.Blocks() as demo:
    gr.Markdown("# Contador de Contenedores (Roboflow)")
    gr.Markdown(
        "Sube una imagen satelital / Google Earth con contenedores. "
        "El modelo de Roboflow detectará cada contenedor (rectángulo)."
    )

    with gr.Row():
        entrada = gr.Image(type="pil", label="Imagen de entrada")
        salida = gr.Image(type="numpy", label="Detecciones")

    texto = gr.Textbox(label="Resultado")

    boton = gr.Button("Contar contenedores")
    boton.click(
        detectar_contenedores,
        inputs=entrada,
        outputs=[salida, texto]
    )

if __name__ == "__main__":
    demo.launch()
