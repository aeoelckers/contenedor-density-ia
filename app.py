import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

from inference_sdk import InferenceHTTPClient
from config import (
    ROBOFLOW_API_KEY,
    ROBOFLOW_WORKSPACE,
    ROBOFLOW_WORKFLOW_ID,
)

# Cliente Roboflow
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

def detectar_contenedores(img_pil):
    """
    Recibe imagen PIL desde Gradio, llama a Roboflow Workflow,
    dibuja las cajas usando PIL (sin OpenCV) y retorna conteo + imagen.
    """

    # Convertir la imagen PIL a JPG binario para Roboflow
    img_bytes = img_pil.convert("RGB")
    img_buffer = np.asarray(img_bytes)

    # Roboflow requiere imagen como bytes, así que la convertimos:
    import io
    buff = io.BytesIO()
    img_pil.save(buff, format="JPEG")
    buff.seek(0)

    # Hacer la llamada al workflow
    result = client.run_workflow(
        workspace_name=ROBOFLOW_WORKSPACE,
        workflow_id=ROBOFLOW_WORKFLOW_ID,
        images={"image": buff.getvalue()},
        use_cache=True
    )

    predictions = result.get("predictions", [])
    conteo = len(predictions)

    # Dibujamos sobre una copia
    draw = ImageDraw.Draw(img_pil)

    for p in predictions:
        x, y = p["x"], p["y"]
        w, h = p["width"], p["height"]
        conf = p.get("confidence", 0)

        # Convertir centro a esquinas
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)

        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        draw.text((x1, y1 - 10), f"{conf:.2f}", fill="lime")

    return img_pil, f"Contenedores detectados: {conteo}"

# Interfaz de Gradio

with gr.Blocks() as demo:
    gr.Markdown("# Contador de Contenedores (Roboflow)")
    gr.Markdown("Sube una imagen satelital o Google Earth. La IA detectará contenedores.")

    input_img = gr.Image(type="pil", label="Imagen")
    output_img = gr.Image(type="pil", label="Detecciones")
    output_text = gr.Textbox(label="Resultado")

    btn = gr.Button("Contar contenedores")
    btn.click(detectar_contenedores, inputs=input_img, outputs=[output_img, output_text])

if __name__ == "__main__":
    demo.launch()
