// ==========================================
// 1. GLOBALS & DOM ELEMENTS
// ==========================================
const video = document.getElementById('video'); // Assuming video element has id="video"
const canvas = document.getElementById('output_canvas'); // Assuming your canvas has id="output_canvas"
const ctx = canvas.getContext('2d');

let yoloSession = null;
let isDetecting = false;
let fretboardCorners = null; // Will hold [TopLeft, TopRight, BottomRight, BottomLeft]
let smoothedBox = null; // Will hold the smoothed bounding box
let handLandmarks = null; // For MediaPipe

// ==========================================
// 2. INITIALIZATION (CAMERA & AI)
// ==========================================

// Boot up the webcam
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720 }
        });
        video.srcObject = stream;
        video.play();
        
        video.onloadeddata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            requestAnimationFrame(renderLoop);
        };
    } catch (err) {
        console.error("Camera access denied:", err);
    }
}

// Boot up the YOLO ONNX Model
async function initAiEngine() {
    console.log("Loading YOLO engine...");
    try {
        // USING WASM: This bypasses the WebGL clash with MediaPipe on your Mac
        yoloSession = await ort.InferenceSession.create('./best.onnx', {
            executionProviders: ['wasm']
        });
        console.log("YOLO engine successfully armed and loaded via WASM.");
    } catch (error) {
        console.error("Failed to boot ONNX runtime:", error);
    }
}

// Boot up MediaPipe Hands (Standard Setup)
// Note: Assuming you have the MediaPipe CDN links in your index.html
if (typeof Hands !== 'undefined') {
    const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });
    hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
    hands.onResults((results) => {
        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            handLandmarks = results.multiHandLandmarks[0];
        } else {
            handLandmarks = null;
        }
    });

    // Send frames to MediaPipe
    async function sendToMediaPipe() {
        if (!video.paused && !video.ended) {
            await hands.send({ image: video });
        }
        setTimeout(sendToMediaPipe, 1000 / 30); // 30 FPS target
    }
    video.addEventListener('loadeddata', sendToMediaPipe);
}

// ==========================================
// 3. THE YOLO VISION PIPELINE
// ==========================================

async function detectGuitarNeck() {
    if (!yoloSession || isDetecting || video.paused || video.ended) return;
    isDetecting = true;

    try {
        // STEP A: PREPROCESSING (Canvas to Tensor)
        // YOLO requires a 640x640 input
        const inputSize = 640;
        const offscreenCanvas = document.createElement('canvas');
        offscreenCanvas.width = inputSize;
        offscreenCanvas.height = inputSize;
        const offCtx = offscreenCanvas.getContext('2d');
        
        // Draw the current video frame to the 640x640 canvas
        offCtx.drawImage(video, 0, 0, inputSize, inputSize);
        const imgData = offCtx.getImageData(0, 0, inputSize, inputSize).data;

        // Create the flat Float32Array [1, 3, 640, 640]
        const floatData = new Float32Array(3 * inputSize * inputSize);
        
        // Planar format conversion: RRR... GGG... BBB... and normalize to 0.0 - 1.0
        for (let i = 0; i < inputSize * inputSize; i++) {
            floatData[i] = imgData[i * 4] / 255.0;                         // Red
            floatData[i + inputSize * inputSize] = imgData[i * 4 + 1] / 255.0;     // Green
            floatData[i + 2 * inputSize * inputSize] = imgData[i * 4 + 2] / 255.0; // Blue
        }

        const tensor = new ort.Tensor('float32', floatData, [1, 3, inputSize, inputSize]);

        // STEP B: RUN INFERENCE
        // Pass the tensor to the ONNX model. The input name is usually 'images'
        const results = await yoloSession.run({ images: tensor });
        const outputData = results.output0.data; // YOLOv8/11 outputs [1, 5, 8400]
        
        // STEP C: POSTPROCESSING (Decoding the Bounding Box)
        let maxConf = 0;
        let bestBox = null;
        const numAnchors = 8400;

        // Scan through all 8400 candidate boxes to find the one with the highest confidence
        for (let i = 0; i < numAnchors; i++) {
            const confidence = outputData[4 * numAnchors + i]; // Index 4 holds the score
            
            if (confidence > maxConf && confidence > 0.5) { // 50% confidence threshold
                maxConf = confidence;
                const xc = outputData[0 * numAnchors + i];
                const yc = outputData[1 * numAnchors + i];
                const w = outputData[2 * numAnchors + i];
                const h = outputData[3 * numAnchors + i];
                bestBox = { xc, yc, w, h, confidence };
            }
        }

        // If we found a guitar neck, apply stabilization math before drawing
        if (bestBox) {
            if (!smoothedBox) {
                // First detection: lock it in immediately
                smoothedBox = bestBox; 
            } else {
                // Check for occlusion (did the box suddenly shrink because a hand is over it?)
                const areaOld = smoothedBox.w * smoothedBox.h;
                const areaNew = bestBox.w * bestBox.h;
                const sizeChange = Math.abs(areaOld - areaNew) / areaOld;

                // If the box shrinks/grows by more than 15% instantly, reject the size change
                if (sizeChange > 0.15) {
                    bestBox.w = smoothedBox.w;
                    bestBox.h = smoothedBox.h;
                }

                // Apply Exponential Moving Average (EMA) to kill the jitter
                const alpha = 0.15; // 15% trust in new frame, 85% trust in previous frame
                smoothedBox.xc = (alpha * bestBox.xc) + ((1 - alpha) * smoothedBox.xc);
                smoothedBox.yc = (alpha * bestBox.yc) + ((1 - alpha) * smoothedBox.yc);
                smoothedBox.w = (alpha * bestBox.w) + ((1 - alpha) * smoothedBox.w);
                smoothedBox.h = (alpha * bestBox.h) + ((1 - alpha) * smoothedBox.h);
            }

            // Scale the stabilized coordinates back to the main canvas size
            const scaleX = canvas.width / inputSize;
            const scaleY = canvas.height / inputSize;

            const xMin = (smoothedBox.xc - smoothedBox.w / 2) * scaleX;
            const yMin = (smoothedBox.yc - smoothedBox.h / 2) * scaleY;
            const xMax = (smoothedBox.xc + smoothedBox.w / 2) * scaleX;
            const yMax = (smoothedBox.yc + smoothedBox.h / 2) * scaleY;

            // Generate the rock-solid homography matrix source points
            fretboardCorners = [
                { x: xMin, y: yMin }, 
                { x: xMax, y: yMin }, 
                { x: xMax, y: yMax }, 
                { x: xMin, y: yMax }  
            ];
        } else {
            // If the AI completely loses the guitar for multiple frames, gracefully let it go
            smoothedBox = null;
            fretboardCorners = null; 
        }
    } catch (error) {
        console.error("Error in detectGuitarNeck:", error);
    } finally {
        isDetecting = false;
    }
}

// ==========================================
// 4. MAIN RENDER LOOP & GRAPHICS
// ==========================================

function renderLoop() {
    // 1. draw the raw video frame at full speed (60fps)
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 2. draw the automated yolo coordinates smoothly
    if (fretboardCorners) {
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(fretboardCorners[0].x, fretboardCorners[0].y);
        ctx.lineTo(fretboardCorners[1].x, fretboardCorners[1].y);
        ctx.lineTo(fretboardCorners[2].x, fretboardCorners[2].y);
        ctx.lineTo(fretboardCorners[3].x, fretboardCorners[3].y);
        ctx.closePath();
        ctx.stroke();
    }

    // 3. draw mediapipe hand landmarks
    if (handLandmarks) {
        ctx.fillStyle = '#FF0000';
        for (let i = 0; i < handLandmarks.length; i++) {
            const px = handLandmarks[i].x * canvas.width;
            const py = handLandmarks[i].y * canvas.height;
            ctx.beginPath();
            ctx.arc(px, py, 5, 0, 2 * Math.PI);
            ctx.fill();
        }
    }

    // loop the graphics instantly
    requestAnimationFrame(renderLoop);
}

// ==========================================
// 5. DECOUPLED AI LOOP
// ==========================================

async function aiLoop() {
    // run the heavy math
    await detectGuitarNeck();
    
    // wait 100 milliseconds before running it again (runs at ~10 fps)
    // this lets the main thread breathe so the video stays smooth
    setTimeout(aiLoop, 100);
}

// boot everything when the script loads
window.addEventListener('DOMContentLoaded', async () => {
    await initAiEngine();
    await startCamera();
    
    // start the slow ai engine completely separate from the fast graphics
    aiLoop(); 
});

// Boot everything when the script loads
window.addEventListener('DOMContentLoaded', () => {
    initAiEngine();
    startCamera();
});