document.getElementById("uploadForm").addEventListener("submit", async function(event) {
    event.preventDefault(); 

    const formData = new FormData();
    const fileField = document.querySelector('input[type="file"]');

    formData.append('file', fileField.files[0]);


    document.getElementById("result").innerHTML = "";
    document.getElementById("prediction").innerHTML = "Processing...";


    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(result => {
       
        document.getElementById("result").innerHTML = `<img src="${URL.createObjectURL(fileField.files[0])}" alt="Uploaded Image" />`;
        document.getElementById("prediction").innerHTML = `Prediction: ${result.prediction}`;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById("prediction").innerHTML = "Error in prediction.";
    });
});
