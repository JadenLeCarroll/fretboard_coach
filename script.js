const video = document.getElementById('video'); 
const canvas = document.getElementById('output_canvas'); 
const ctx = canvas.getContext('2d');

// set up the background worker
const yoloWorker = new Worker('yolo_worker.js');
let fretboardCorners = null;
let isWorkerBusy = false;

// listen for the worker texting us back
yoloWorker.onmessage = function(e) {
    if (e.data.type === 'results') {
        fretboardCorners = e.data.corners; // update the box coordinates
        isWorkerBusy = false;              // worker is ready for the next frame
    } else if (e.data.type === 'completed') {
        isWorkerBusy = false;
    }
};

// boot the webcam
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
            
            // start the graphics loop (fast)
            requestAnimationFrame(renderLoop);
            // start the communication loop (slower)
            sendFrameToWorker();
        };
    } catch (err) {
        console.error("camera access denied:", err);
    }
}

// extract pixels and throw them to the background thread
function sendFrameToWorker() {
    if (!isWorkerBusy && !video.paused && !video.ended) {
        isWorkerBusy = true;
        
        const inputSize = 640;
        const offscreenCanvas = document.createElement('canvas');
        offscreenCanvas.width = inputSize;
        offscreenCanvas.height = inputSize;
        const offCtx = offscreenCanvas.getContext('2d');
        
        offCtx.drawImage(video, 0, 0, inputSize, inputSize);
        const imgData = offCtx.getImageData(0, 0, inputSize, inputSize);

        // transfer the memory buffer directly for massive speed gains
        yoloWorker.postMessage({ 
            pixels: imgData.data.buffer, 
            mainCanvasWidth: canvas.width, 
            mainCanvasHeight: canvas.height 
        }, [imgData.data.buffer]); 
    }
    
    // check if we should send another frame every 33ms (~30fps limit on the ai side)
    setTimeout(sendFrameToWorker, 33);
}

// pure 60 fps rendering loop
function renderLoop() {
    // draw the video instantly
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // draw the stabilized coordinates (if the worker has found them)
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

    // loop immediately 
    requestAnimationFrame(renderLoop);
}

// boot
window.addEventListener('DOMContentLoaded', startCamera);