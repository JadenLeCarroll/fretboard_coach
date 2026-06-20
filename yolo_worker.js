// import onnx runtime directly into the background thread
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');

// 1. tell onnx exactly where to find the extra wasm files on the internet
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
// 2. force single-threaded mode to bypass local server security blocks
ort.env.wasm.numThreads = 1;

let yoloSession = null;
let smoothedBox = null;

// boot the engine in the background
async function initAiEngine() {
    try {
        yoloSession = await ort.InferenceSession.create('./best.onnx', {
            executionProviders: ['wasm']
        });
        console.log("worker: yolo engine armed.");
        postMessage({ type: 'status', message: 'ready' });
    } catch (error) {
        console.error("worker: failed to boot onnx", error);
    }
}

initAiEngine();

// listen for pixel data coming from the main thread
self.onmessage = async function(e) {
    if (!yoloSession) {
        postMessage({ type: 'completed' });
        return;
    }

    const { pixels, mainCanvasWidth, mainCanvasHeight } = e.data;
    const inputSize = 640;
    
    // reconstruct the array from the raw memory buffer
    const imgData = new Uint8ClampedArray(pixels);
    const floatData = new Float32Array(3 * inputSize * inputSize);

    // format the pixels for the neural network
    for (let i = 0; i < inputSize * inputSize; i++) {
        floatData[i] = imgData[i * 4] / 255.0;                         // red
        floatData[i + inputSize * inputSize] = imgData[i * 4 + 1] / 255.0;     // green
        floatData[i + 2 * inputSize * inputSize] = imgData[i * 4 + 2] / 255.0; // blue
    }

    const tensor = new ort.Tensor('float32', floatData, [1, 3, inputSize, inputSize]);

    try {
        // run inference
        const results = await yoloSession.run({ images: tensor });
        const outputData = results.output0.data; 
        
        let maxConf = 0;
        let bestBox = null;
        const numAnchors = 8400;

        for (let i = 0; i < numAnchors; i++) {
            const confidence = outputData[4 * numAnchors + i]; 
            
            // DROP THRESHOLD TO 10% FOR DEBUGGING
            if (confidence > maxConf && confidence > 0.1) { 
                maxConf = confidence;
                bestBox = {
                    xc: outputData[0 * numAnchors + i],
                    yc: outputData[1 * numAnchors + i],
                    w: outputData[2 * numAnchors + i],
                    h: outputData[3 * numAnchors + i],
                    confidence
                };
            }
        }

        // LOG THE AI'S THOUGHT PROCESS
        if (maxConf > 0) {
            console.log(`Worker: Best detection is ${(maxConf * 100).toFixed(1)}% confident`);
        }

        // apply smoothing math
        let fretboardCorners = null;

        if (bestBox) {
            if (!smoothedBox) {
                smoothedBox = bestBox; 
            } else {
                const areaOld = smoothedBox.w * smoothedBox.h;
                const areaNew = bestBox.w * bestBox.h;
                const sizeChange = Math.abs(areaOld - areaNew) / areaOld;

                // reject sudden shrinking from hand occlusion
                if (sizeChange > 0.15) {
                    bestBox.w = smoothedBox.w;
                    bestBox.h = smoothedBox.h;
                }

                // smooth gliding
                const alpha = 0.15; 
                smoothedBox.xc = (alpha * bestBox.xc) + ((1 - alpha) * smoothedBox.xc);
                smoothedBox.yc = (alpha * bestBox.yc) + ((1 - alpha) * smoothedBox.yc);
                smoothedBox.w = (alpha * bestBox.w) + ((1 - alpha) * smoothedBox.w);
                smoothedBox.h = (alpha * bestBox.h) + ((1 - alpha) * smoothedBox.h);
            }

            // scale the coordinates to the real canvas size
            const scaleX = mainCanvasWidth / inputSize;
            const scaleY = mainCanvasHeight / inputSize;

            const xMin = (smoothedBox.xc - smoothedBox.w / 2) * scaleX;
            const yMin = (smoothedBox.yc - smoothedBox.h / 2) * scaleY;
            const xMax = (smoothedBox.xc + smoothedBox.w / 2) * scaleX;
            const yMax = (smoothedBox.yc + smoothedBox.h / 2) * scaleY;

            fretboardCorners = [
                { x: xMin, y: yMin }, 
                { x: xMax, y: yMin }, 
                { x: xMax, y: yMax }, 
                { x: xMin, y: yMax }  
            ];
        } else {
            smoothedBox = null;
        }

        // send the final processed coordinates back to the main thread
        postMessage({ 
            type: 'results', 
            corners: fretboardCorners 
        });

    } catch (err) {
        console.error("worker process error:", err);
        postMessage({ type: 'completed' });
    }
};