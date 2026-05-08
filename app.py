from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load model
model = tf.keras.models.load_model(
    "fin_attention_resunet_best_improved.keras",
    compile=False
)

IMG_SIZE = 256

@app.get("/")
def home():
    return {"message": "OptiViewDx API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read image
    image = Image.open(file.file).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Preprocess
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)[0]

    # Binary mask
    pred = (pred > 0.5).astype(np.uint8) * 255

    if len(pred.shape) == 3:
        pred = pred[:, :, 0]

    # Convert to image
    mask = Image.fromarray(pred)

    # Send image back
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
