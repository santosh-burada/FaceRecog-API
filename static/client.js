const video = document.getElementById('liveFeed');
const croppedFaceImg = document.getElementById('croppedFace');
const captureButton = document.getElementById('captureButton');
const userNameInput = document.getElementById('userName');

// Access the camera on the client side
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        video.play();
        sendFramesToServer();
    })
    .catch(err => {
        console.error('Error accessing camera:', err);
    });

// WebSocket connection for streaming video frames and receiving cropped face
const ws = new WebSocket('ws://127.0.0.1:8080/ws/camera/');

ws.onopen = () => {
    console.log('WebSocket connection opened');
};

ws.onmessage = (event) => {
    const imageUrl = URL.createObjectURL(event.data);
    croppedFaceImg.src = imageUrl;
};

ws.onerror = (error) => {
    console.error('WebSocket Error:', error);
};

ws.onclose = () => {
    console.log('WebSocket connection closed');
};

function sendFramesToServer() {
    // Capture frames from the video element and send to the server every 500ms
    setInterval(() => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(blob => {
            ws.send(blob);
        }, 'image/jpeg');
    }, 500);
}

captureButton.addEventListener('click', function() {
    const userName = userNameInput.value;
    if (!userName) {
        alert('Please provide a name before capturing.');
        return;
    }

    // Send a request to the server to save the cropped image with the given user name
    fetch('/save_image/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username: userName }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            alert(data.message);
        } else {
            alert('Failed to save image. Please try again.');
        }
    })
    .catch((error) => {
        console.error('Error saving image:', error);
    });
});