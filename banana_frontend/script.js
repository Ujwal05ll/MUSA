const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultDiv = document.getElementById("result");

const API_URL = "http://127.0.0.1:5000/predict";

function chooseImage() {
    imageInput.click();
}

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (!file) return;

    preview.src = URL.createObjectURL(file);
    analyzeBtn.style.display = "block";
    resultDiv.innerHTML = "";
});

async function analyze() {
    resultDiv.innerHTML = "⏳ Analyzing...";

    const formData = new FormData();
    formData.append("image", imageInput.files[0]);

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        resultDiv.innerHTML = `
            ✅ ${data.prediction}<br>
            Confidence: ${data.confidence}%
        `;
    } catch (error) {
        resultDiv.innerHTML = "❌ Server error";
    }
}
