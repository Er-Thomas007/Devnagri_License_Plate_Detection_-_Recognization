document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const video = document.getElementById('video');
    const captureButton = document.getElementById('capture');
    const processButton = document.getElementById('process');
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');

    // Handle file input for image preview
    if (fileInput) {
        fileInput.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    imagePreview.src = e.target.result;
                };
                reader.readAsDataURL(file);
            }
        });
    }

    // Handle video capture
    if (captureButton && video) {
        // Start webcam stream
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
                video.play();
            })
            .catch(err => {
                console.error('Error accessing webcam: ', err);
            });

        captureButton.addEventListener('click', function() {
            if (video.srcObject) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL('image/jpeg').split(',')[1]; // Get base64 string

                // Display the captured image
                imagePreview.src = canvas.toDataURL('image/jpeg');
                imagePreview.style.display = 'block';
                processButton.style.display = 'block';
            } else {
                console.error('No video stream available');
            }
        });

        // Handle image processing
        if (processButton) {
            processButton.addEventListener('click', function() {
                const imageData = imagePreview.src.split(',')[1]; // Get base64 string

                // Send the image data to the server for processing
                fetch('/process-webcam-image', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: imageData
                    })
                })
                .then(response => response.json())
                .then(data => {
                    // Redirect to results page with query parameters
                    const queryParams = new URLSearchParams({
                        processed_img_data: data.processed_img_data,
                        english_text: data.english_text,
                        nepali_text: data.nepali_text,
                        is_bagmati: data.is_bagmati
                    }).toString();

                    window.location.href = `/results?${queryParams}`;
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            });
        }
    }
});
