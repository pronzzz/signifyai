const video = document.getElementById('webcam');
const canvas = document.getElementById('snapshot-canvas');
const ctx = canvas.getContext('2d');
const toggleBtn = document.getElementById('toggle-cam');
const predictionLabel = document.getElementById('prediction-label');
const fpsDisplay = document.getElementById('fps');
const statusDisplay = document.getElementById('status');

let isStreaming = false;
let stream = null;
let lastFrameTime = 0;
let frameCount = 0;
let inferenceInterval = null;

// Start/Stop Webcam
toggleBtn.addEventListener('click', async () => {
    if (isStreaming) {
        stopWebcam();
    } else {
        await startWebcam();
    }
});

async function startWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        isStreaming = true;
        toggleBtn.textContent = 'Stop Webcam';
        statusDisplay.textContent = 'Streaming';
        
        // Start waiting for video to be ready
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            startInferenceLoop();
        };
    } catch (err) {
        console.error("Error accessing webcam:", err);
        statusDisplay.textContent = 'Error: ' + err.message;
    }
}

function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    isStreaming = false;
    toggleBtn.textContent = 'Start Webcam';
    statusDisplay.textContent = 'Stopped';
    clearInterval(inferenceInterval);
}

function startInferenceLoop() {
    // Run inference every 200ms (5 FPS) to balance load
    inferenceInterval = setInterval(sendFrameForPrediction, 200);
    requestAnimationFrame(updateFPS);
}

function updateFPS(now) {
    if (!isStreaming) return;
    
    if (now - lastFrameTime >= 1000) {
        fpsDisplay.textContent = frameCount;
        frameCount = 0;
        lastFrameTime = now;
    }
    frameCount++;
    requestAnimationFrame(updateFPS);
}

async function sendFrameForPrediction() {
    if (!isStreaming) return;
    
    // Draw current video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert canvas to blob
    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');
        
        try {
            const startTime = performance.now();
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            const latency = performance.now() - startTime;
            
            if (response.ok) {
                if (result.confidence > 0.5) {
                    predictionLabel.textContent = `${result.label} (${(result.confidence * 100).toFixed(1)}%)`;
                } else {
                    predictionLabel.textContent = "Uncertain";
                }
            } else {
                console.warn("Prediction error:", result);
            }
        } catch (err) {
            console.error("Network error:", err);
        }
    }, 'image/jpeg', 0.8);
}
