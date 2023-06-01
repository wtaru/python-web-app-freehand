import numpy as np
from PIL import Image
import onnxruntime

pil_image = Image.open("sample.png")

# 入力データとして渡す前に前処理
resized_image = pil_image.resize((28, 28))
resized_arr = np.array(resized_image)
print(f"resized_arr shape: {resized_arr.shape}")

# (28*28*4)の4の成分(RGBAのA)を取り出すため(4*28*28)に変換
transposed_arr = resized_arr.transpose(2, 0, 1)
print(f"transposed_arr shape: {transposed_arr.shape}")

# Aの中のアルファの成分を取り出す
alpha_arr = transposed_arr[3]
for i in alpha_arr:
    for j in i:
        print("%3d " % j, end="")
    print()

# 28次元のデータを1次元に変換する
reshaped_arr = alpha_arr.reshape(-1)
print("reshaped_arr")
print(reshaped_arr)

# onxxで入力データはFloat型を指定してるためfloat32
input = [reshaped_arr.astype(np.float32)]


# 推論
# モデル読み込み
onnx_session = onnxruntime.InferenceSession("model.onnx")
output = onnx_session.run(["probabilities"], {"float_input": input})
res = output[0][0]
print("output")
print(res)