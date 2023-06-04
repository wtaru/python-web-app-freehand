// ◆ canvas内の手書き実装 ◆ 
const canvasElement = document.getElementById("drow-area");
const canvas = new HandwritingCanvas(canvasElement);


// ◆ 書き直しボタンの実装 ◆ 
const clearButtonElement = document.getElementById("clear-button");
clearButtonElement.addEventListener("click", () => canvas.clear())

// 推論関数
async function preprocess(blob) {
    // 画像を２８✖️２８に変換
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = 28;
    canvas.height = 28;
    const bitmap = await createImageBitmap(blob, {
        resizeHeight: 28,
        resizeWidth: 28
    })
    ctx.drawImage(bitmap, 0, 0);
    const imageData = ctx.getImageData(0, 0, 28, 28);

    // RGBAのA要素を取り出す
    const alphas = [];
    for (let i = 0; i < imageData.data.length; i++) {
        if (i % 4 === 3) {
            const alpha = imageData.data[i];
            alphas.push(alpha);
        }
    }
    return alphas;
}

async function predict(input) {
    // model onnx取得
    const session = await ort.InferenceSession.create("/static/model.onnx");
    // input data 準備
    const feeds = {
        float_input: new ort.Tensor("float32", input, [1, 28 * 28])
    }
    const results = await session.run(feeds);
    return results.probabilities.data;
}

// ◆ 推論実行ボタンの実装 ◆ 
const predictButtonElement = document.getElementById("predict-button");
predictButtonElement.addEventListener("click", async () => {
    if (canvas.isEmpty) { return } 
    
    // ◆ 推論を実行する
    // canvas文字をpng形式の画像に変換
    const blob = await canvas.toBlob("image/png");
    const input = await preprocess(blob);
    const probabilities = await predict(input);
    
    // ◆ 推論結果を画像に表示 ◆
    const imageUrl = URL.createObjectURL(blob);
    const imageElement = document.createElement("img");
    imageElement.src = imageUrl;
    const resultImageElement = document.getElementById("res-image");
    // 子要素がある場合は削除
    if (resultImageElement.firstChild) {
        resultImageElement.removeChild(resultImageElement.firstChild)
    }
    resultImageElement.appendChild(imageElement);
    canvas.clear()
    

    // ◆ 推論結果を画面に表示する ◆ 
    const tableBodyElement = document.getElementById("result-table-body");
    // tableBodyElementに子要素があった場合は削除
    while (tableBodyElement.firstChild) {
        tableBodyElement.removeChild(tableBodyElement.firstChild)
    }
    // 子要素の作成
    for (let i = 0; i < probabilities.length; i++) {
        const tr_tag = document.createElement("tr");
        const tdNumber = document.createElement("td");
        tdNumber.textContent = i;
        tr_tag.appendChild(tdNumber)
        const tdProbability = document.createElement("td");
        // 確率は100％表記で小数点以下1位を表示
        tdProbability.textContent = (probabilities[i] * 100).toFixed(1);
        tr_tag.appendChild(tdProbability);
        tableBodyElement.appendChild(tr_tag);
    }
})

