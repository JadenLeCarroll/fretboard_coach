const startBtn = document.getElementById('start-btn');
const pitchDisplay = document.getElementById('pitch-display');
const noteDisplay = document.getElementById('note-display');

let audioContext;
let analyser;
let microphone;

// standard notes for mapping hz to a guitar string
const notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

function hzToNote(frequency) {
    const noteNum = 12 * (Math.log(frequency / 440) / Math.log(2));
    return notes[(Math.round(noteNum) + 69) % 12];
}

// autocorrelation algorithm: the industry standard for js guitar tuners
function autoCorrelate(buffer, sampleRate) {
    let size = buffer.length;
    let sumOfSquares = 0;
    
    for (let i = 0; i < size; i++) {
        let val = buffer[i];
        sumOfSquares += val * val;
    }
    
    // if it's too quiet, ignore it (acts like a noise gate)
    let rootMeanSquare = Math.sqrt(sumOfSquares / size);
    if (rootMeanSquare < 0.01) return -1;

    let r1 = 0, r2 = size - 1, thres = 0.2;
    for (let i = 0; i < size / 2; i++)
        if (Math.abs(buffer[i]) < thres) { r1 = i; break; }
    for (let i = 1; i < size / 2; i++)
        if (Math.abs(buffer[size - i]) < thres) { r2 = size - i; break; }

    buffer = buffer.slice(r1, r2);
    size = buffer.length;

    let c = new Array(size).fill(0);
    for (let i = 0; i < size; i++)
        for (let j = 0; j < size - i; j++)
            c[i] = c[i] + buffer[j] * buffer[j + i];

    let d = 0;
    while (c[d] > c[d + 1]) d++;
    
    let maxval = -1, maxpos = -1;
    for (let i = d; i < size; i++) {
        if (c[i] > maxval) {
            maxval = c[i];
            maxpos = i;
        }
    }
    let t0 = maxpos;

    // parabolic interpolation for more accuracy
    let x1 = c[t0 - 1], x2 = c[t0], x3 = c[t0 + 1];
    let a = (x1 + x3 - 2 * x2) / 2;
    let b = (x3 - x1) / 2;
    if (a) t0 = t0 - b / (2 * a);

    return sampleRate / t0;
}

async function startAudioEngine() {
    // initialize the audio context
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        
        microphone = audioContext.createMediaStreamSource(stream);
        microphone.connect(analyser);
        
        startBtn.style.display = 'none';
        updatePitch();
        
    } catch (err) {
        pitchDisplay.innerText = "mic access denied";
        console.error(err);
    }
}

function updatePitch() {
    let buffer = new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(buffer);
    
    let pitch = autoCorrelate(buffer, audioContext.sampleRate);
    
    if (pitch !== -1) {
        pitchDisplay.innerText = Math.round(pitch) + " hz";
        noteDisplay.innerText = hzToNote(pitch);
    } else {
        pitchDisplay.innerText = "-- hz";
        noteDisplay.innerText = "--";
    }
    
    requestAnimationFrame(updatePitch);
}

startBtn.addEventListener('click', startAudioEngine);