import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video = document.getElementById("video");
const canvas = document.getElementById("output_canvas");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("status");

let handLandmarker;
let lastVideoTime = -1;

// --- MUSIC & GRID VARIABLES ---
const CHORD_DICTIONARIES = {
    "A Minor": [
        { string: 3, fret: 2, label: "M" }, 
        { string: 2, fret: 2, label: "R" }, 
        { string: 1, fret: 1, label: "I" }  
    ],
    "C Major": [
        { string: 4, fret: 3, label: "R" },  
        { string: 3, fret: 2, label: "M" }, 
        { string: 1, fret: 1, label: "I" }  
    ]
};

const ACTIVE_CHORD = "A Minor";
const targetNotes = CHORD_DICTIONARIES[ACTIVE_CHORD];

let calibrationPoints = [];
const flatW = 500;
const flatH = 200;
const fretWidth = flatW / 5;
const stringHeight = flatH / 5;
let homographyMatrix = null;

// --- SMOOTHING VARIABLES ---
let previousPositions = { "I": null, "M": null, "R": null };
const smoothing = 0.6; 

// --- MATH ENGINES ---
function calculateJointAngle(a, b, c) {
    const ba = [a.x - b.x, a.y - b.y, a.z - b.z];
    const bc = [c.x - b.x, c.y - b.y, c.z - b.z];
    const dotProduct = ba[0]*bc[0] + ba[1]*bc[1] + ba[2]*bc[2];
    const magBA = Math.sqrt(ba[0]**2 + ba[1]**2 + ba[2]**2);
    const magBC = Math.sqrt(bc[0]**2 + bc[1]**2 + bc[2]**2);
    if (magBA * magBC === 0) return 0;
    return Math.acos(dotProduct / (magBA * magBC)) * (180 / Math.PI);
}

// Replaces OpenCV's getPerspectiveTransform
function getPerspectiveTransform(src, dst) {
    let a = [];
    for (let i = 0; i < 4; i++) {
        a.push([src[i].x, src[i].y, 1, 0, 0, 0, -src[i].x * dst[i].x, -src[i].y * dst[i].x]);
        a.push([0, 0, 0, src[i].x, src[i].y, 1, -src[i].x * dst[i].y, -src[i].y * dst[i].y]);
    }
    let b = [];
    for (let i = 0; i < 4; i++) {
        b.push(dst[i].x); b.push(dst[i].y);
    }
    for (let i = 0; i < 8; i++) {
        let maxRow = i;
        for (let k = i + 1; k < 8; k++) if (Math.abs(a[k][i]) > Math.abs(a[maxRow][i])) maxRow = k;
        let temp = a[i]; a[i] = a[maxRow]; a[maxRow] = temp;
        let tempB = b[i]; b[i] = b[maxRow]; b[maxRow] = tempB;
        for (let k = i + 1; k < 8; k++) {
            let c = -a[k][i] / a[i][i];
            for (let j = i; j < 8; j++) {
                if (i === j) a[k][j] = 0;
                else a[k][j] += c * a[i][j];
            }
            b[k] += c * b[i];
        }
    }
    let x = new Array(8);
    for (let i = 7; i >= 0; i--) {
        x[i] = b[i];
        for (let k = i + 1; k < 8; k++) x[i] -= a[i][k] * x[k];
        x[i] = x[i] / a[i][i];
    }
    return x;
}

// Replaces OpenCV's perspectiveTransform
function warpPoint(x, y, matrix) {
    let denominator = matrix[6] * x + matrix[7] * y + 1;
    return {
        x: (matrix[0] * x + matrix[1] * y + matrix[2]) / denominator,
        y: (matrix[3] * x + matrix[4] * y + matrix[5]) / denominator
    };
}

// --- MOUSE LISTENERS ---
canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    // Invert X because the canvas is mirrored via CSS
    const clickX = canvas.width - ((e.clientX - rect.left) * (canvas.width / rect.width));
    const clickY = (e.clientY - rect.top) * (canvas.height / rect.height);

    if (calibrationPoints.length === 4) calibrationPoints = []; // Reset on 5th click
    calibrationPoints.push({ x: clickX, y: clickY });

    if (calibrationPoints.length === 4) {
        const dstPoints = [
            { x: 0, y: 0 }, { x: flatW, y: 0 }, 
            { x: flatW, y: flatH }, { x: 0, y: flatH }
        ];
        homographyMatrix = getPerspectiveTransform(calibrationPoints, dstPoints);
    } else {
        homographyMatrix = null;
    }
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'c') {
        calibrationPoints = [];
        homographyMatrix = null;
    }
});

// --- AI PIPELINE ---
async function initializeAI() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.75,
        minHandPresenceConfidence: 0.75,
        minTrackingConfidence: 0.75
    });
    statusText.innerText = "System Ready. Click 4 corners of fretboard to calibrate.";
    startCamera();
}

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    } catch (err) {
        statusText.innerText = "Error: Camera access denied.";
        console.error(err);
    }
}

async function predictWebcam() {
    if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const results = handLandmarker.detectForVideo(video, performance.now());
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Draw Calibration Points
        ctx.fillStyle = "orange";
        calibrationPoints.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 5, 0, 2 * Math.PI);
            ctx.fill();
        });

        // Draw Calibration Polygon
        if (calibrationPoints.length === 4) {
            ctx.strokeStyle = "orange";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(calibrationPoints[0].x, calibrationPoints[0].y);
            for (let i = 1; i < 4; i++) ctx.lineTo(calibrationPoints[i].x, calibrationPoints[i].y);
            ctx.closePath();
            ctx.stroke();
        }

        let satisfiedTargets = new Set();

        if (results.landmarks && results.landmarks.length > 0) {
            const landmarks = results.landmarks[0]; 
            
            const fingersToTrack = [
                { label: "I", mcp: 5, pip: 6, tip: 8 },
                { label: "M", mcp: 9, pip: 10, tip: 12 },
                { label: "R", mcp: 13, pip: 14, tip: 16 }
            ];

            fingersToTrack.forEach(finger => {
                const mcp = landmarks[finger.mcp];
                const pip = landmarks[finger.pip];
                const tip = landmarks[finger.tip];

                const rawCx = tip.x * canvas.width;
                const rawCy = tip.y * canvas.height;

                let cx, cy;
                if (!previousPositions[finger.label]) {
                    cx = rawCx; cy = rawCy;
                } else {
                    const px = previousPositions[finger.label].x;
                    const py = previousPositions[finger.label].y;
                    cx = (rawCx * (1 - smoothing)) + (px * smoothing);
                    cy = (rawCy * (1 - smoothing)) + (py * smoothing);
                }
                previousPositions[finger.label] = { x: cx, y: cy };

                const bendAngle = calculateJointAngle(mcp, pip, tip);
                const isPressing = bendAngle < 150.0;

                ctx.beginPath();
                ctx.arc(cx, cy, 10, 0, 2 * Math.PI);
                ctx.fillStyle = isPressing ? "#00FF00" : "#FF0000";
                ctx.fill();

                // Process through Homography Matrix
                if (isPressing && homographyMatrix) {
                    const flatPt = warpPoint(cx, cy, homographyMatrix);
                    
                    if (flatPt.x >= 0 && flatPt.x <= flatW && flatPt.y >= 0 && flatPt.y <= flatH) {
                        const fretIdx = Math.floor(flatPt.x / fretWidth);
                        const stringIdx = 5 - Math.floor(flatPt.y / stringHeight);
                        
                        targetNotes.forEach((target, i) => {
                            if (target.string === stringIdx && target.fret === fretIdx && target.label === finger.label) {
                                satisfiedTargets.add(i);
                            }
                        });
                    }
                }
            });
        }

        // Render Status UI
        if (homographyMatrix) {
            const allPressed = satisfiedTargets.size === targetNotes.length;
            const status = allPressed ? `${ACTIVE_CHORD}: VALIDATED` : `${ACTIVE_CHORD}: INCOMPLETE (${satisfiedTargets.size}/${targetNotes.length})`;
            
            ctx.save();
            
            // The canvas is mirrored via CSS, so we "un-mirror" the drawing context 
            // specifically for the UI box so the text reads correctly.
            ctx.translate(canvas.width, 0); 
            ctx.scale(-1, 1); 
            
            ctx.fillStyle = "black";
            ctx.fillRect(20, 20, 300, 40);
            
            ctx.fillStyle = allPressed ? "#00FF00" : "#FF0000";
            ctx.font = "20px monospace";
            ctx.fillText(status, 30, 45);
            
            ctx.restore();
        }
    }
    window.requestAnimationFrame(predictWebcam);
}

initializeAI();