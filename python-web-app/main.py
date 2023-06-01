from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from PIL import Image
import numpy as np
import onnxruntime

app = FastAPI()

# endpoint以外のアクセスがあった場合はstaticディレクトリ以下のstatic(css)ファイルを返す
app.mount("/static", StaticFiles(directory="static"))

# response_class : HTMLを返すという意
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r") as f:
        return f.read()

# 固定のデータとして推論の結果を返す
@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    image_data = await image.read()
    pil_image = Image.open(BytesIO(image_data))
    
    # ◆ 推論処理 ◆
    # 前処理(predict.py参照)
    resized_image = pil_image.resize((28, 28))
    resized_arr = np.array(resized_image)
    transposed_arr = resized_arr.transpose(2, 0, 1)
    alpha_arr = transposed_arr[3]
    reshaped_arr = alpha_arr.reshape(-1)
    input = [reshaped_arr.astype(np.float32)]
    
    # 推論
    onnx_session = onnxruntime.InferenceSession("model.onnx")
    output = onnx_session.run(["probabilities"], {"float_input": input})
    res = output[0][0] 
     
    return {"probabilities": res.tolist()}