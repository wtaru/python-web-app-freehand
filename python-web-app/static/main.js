// ◆ canvas内の手書き実装 ◆ 
const canvasElement = document.getElementById("drow-area");
const canvas = new HandwritingCanvas(canvasElement);


// ◆ 書き直しボタンの実装 ◆ 
const clearButtonElement = document.getElementById("clear-button");
clearButtonElement.addEventListener("click", () => canvas.clear())


// ◆ 推論実行ボタンの実装 ◆ 
const predictButtonElement = document.getElementById("predict-button");
predictButtonElement.addEventListener("click", async () => {
    if (canvas.isEmpty) { return } 
    
    // ◆ 推論を実行する
    // canvas文字をpng形式の画像に変換
    const blob = await canvas.toBlob("image/png");
    const formData = new FormData()
    formData.append("image", blob, "number.png");
    const response = await fetch("/api/predict", {
        method: "POST",
        body: formData
    });
    const responseData = await response.json();


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
    const probabilities = responseData.probabilities;
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

