function uploadImage() {
    let formData = new FormData();
    let fileInput = document.getElementById("imageUpload");
    formData.append("image", fileInput.files[0]);

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        let resultsDiv = document.getElementById("results");
        resultsDiv.innerHTML = "";

        if (data.error) {
            resultsDiv.innerHTML = `<p style="color: red;">${data.error}</p>`;
        } else {
            // Display detections
            data.detections.forEach(detection => {
                resultsDiv.innerHTML += `
                    <p><strong>Detected:</strong> ${detection.class} (Count: ${detection.count})</p>
                    <p><strong>Reuse Tip:</strong> ${detection.reuse}</p>
                    <p><strong>Disposal:</strong> ${detection.dispose}</p>
                    <hr>
                `;
            });

            // Display original and processed images
            let originalImageDiv = document.getElementById("originalImage");
            let processedImageDiv = document.getElementById("processedImage");

            originalImageDiv.innerHTML = `<h3>Original Image</h3><img src="${data.original_image_url}" alt="Original Image" style="width: 100%; max-width: 500px;" />`;
            processedImageDiv.innerHTML = `<h3>Processed Image</h3><img src="${data.processed_image_url}" alt="Processed Image" style="width: 100%; max-width: 500px;" />`;
        }
    })
    .catch(error => console.error("Error:", error));
}