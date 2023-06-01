from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# データの準備
x, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
print("mnist comp")

x = x[:700]
y = y[:700]

# データの分割
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=123
)

# 学習
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# 精度確認
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {accuracy}")

# モデルファイル名の定義
model_file = "model.onnx"

# 入力データの形式を指定(mnistは28*28=768のデータである)
inital_types = [("float_input", FloatTensorType([None, 28*28]))]

# モデルを変換する処理(optionで出力結果の形式を指定している)
onnx = convert_sklearn(model, initial_types=inital_types, options={"zipmap": False})

with open(model_file, "wb") as f:
    f.write(onnx.SerializeToString())
    
print(f"{model_file} exported")