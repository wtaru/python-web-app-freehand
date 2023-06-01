from sklearn.datasets import fetch_openml

# openmlからmnistデータをダウンロード
x, y = fetch_openml(
    "mnist_784", 
    version=1, 
    return_X_y=True, 
    as_frame=False
)

print(f"x: {len(x)}")
print(f"x: {len(y)}")