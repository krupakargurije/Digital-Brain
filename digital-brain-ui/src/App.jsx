import { useState, useRef, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import { Send, UserCircle2, BrainCircuit, Mic, Loader2 } from 'lucide-react';

const API_URL = 'http://127.0.0.1:5000/api';

function App() {
    const webcamRef = useRef(null);
    const canvasRef = useRef(null);
    const isRecordingRef = useRef(false);
    const activeUserIdRef = useRef(null);
    const lastFrameDataRef = useRef(null);
    const motionCanvasRef = useRef(null);
    const detectedFacesRef = useRef([]);
    const isPollingRef = useRef(false);
    const isInputFocusedRef = useRef(false);
    const isAutoDetectEnabledRef = useRef(true);
    const sessionPersonMapRef = useRef({});
    // ...
    const [isAutoDetectEnabled, setIsAutoDetectEnabled] = useState(true);
    const [isRecording, setIsRecording] = useState(false);
    const [isIdentifying, setIsIdentifying] = useState(false);
    const [userProfile, setUserProfile] = useState(null);
    const [messages, setMessages] = useState([{
        sender: 'system',
        text: "Awaiting identification... Click 'Identify Me' to begin."
    }]);
    const [inputText, setInputText] = useState('');
    const [isThinking, setIsThinking] = useState(false);

    // Store detected faces for rendering
    const [detectedFaces, setDetectedFaces] = useState([]);

    // Capture Image
    const captureImage = useCallback(() => {
        if (!webcamRef.current) return null;
        return webcamRef.current.getScreenshot();
    }, [webcamRef]);

    // Handle Identify Click
    const handleIdentify = async () => {
        setIsIdentifying(true);
        const imageSrc = captureImage();

        // Start audio recording using AudioContext to extract raw WAV
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const mediaStreamSource = audioContext.createMediaStreamSource(stream);
            const processor = audioContext.createScriptProcessor(4096, 1, 1);

            mediaStreamSource.connect(processor);
            processor.connect(audioContext.destination);

            const audioData = [];
            processor.onaudioprocess = (e) => {
                if (isRecordingRef.current) {
                    const channelData = e.inputBuffer.getChannelData(0);
                    audioData.push(new Float32Array(channelData));
                }
            };

            setIsRecording(true);
            isRecordingRef.current = true;

            // Add system message
            setMessages(prev => [...prev, { sender: 'system', text: "Capturing face snapshot and recording 3 seconds of audio..." }]);

            // Stop recording after 3 seconds
            setTimeout(async () => {
                isRecordingRef.current = false;
                setIsRecording(false);

                // Disconnect
                processor.disconnect();
                mediaStreamSource.disconnect();
                stream.getTracks().forEach(track => track.stop());

                // Compile Float32Array into WAV
                const totalLength = audioData.reduce((acc, current) => acc + current.length, 0);
                const combined = new Float32Array(totalLength);
                let offset = 0;
                for (let i = 0; i < audioData.length; i++) {
                    combined.set(audioData[i], offset);
                    offset += audioData[i].length;
                }

                const wavBlob = encodeWAV(combined, audioContext.sampleRate);

                // IMPORTANT: Close the hardware audio context to prevent browser crashing after 6 clicks
                await audioContext.close();

                const reader = new FileReader();
                reader.readAsDataURL(wavBlob);
                reader.onloadend = async () => {
                    const base64Audio = reader.result;
                    setMessages(prev => [...prev, { sender: 'system', text: "Analyzing Identity in Brain (Face + Voice)..." }]);
                    await sendIdentifyRequest(imageSrc, base64Audio);
                };
            }, 3000);

        } catch (err) {
            console.error("Error accessing microphone:", err);
            setMessages(prev => [...prev, { sender: 'system', text: "Microphone access denied or unavailable." }]);
            setIsIdentifying(false);
        }
    };

    const sendIdentifyRequest = async (imageB64, audioB64) => {
        try {
            const res = await axios.post(`${API_URL}/identify`, {
                image: imageB64,
                audio: audioB64
            });

            const { user_id, profile, status, greeting } = res.data;
            setUserProfile({ user_id, ...profile, status });

            setMessages(prev => [
                ...prev,
                { sender: 'system', text: `Identification Complete. Result: ${status}` },
                { sender: 'assistant', text: greeting }
            ]);
        } catch (error) {
            console.error(error);
            const errMsg = error.response?.data?.error || "Failed to communicate with the server backend.";
            setMessages(prev => [...prev, { sender: 'system', text: `Error: ${errMsg}` }]);
        } finally {
            setIsIdentifying(false);
        }
    };

    const handleSendMessage = async (e) => {
        e?.preventDefault();
        if (!inputText.trim() || !userProfile) return;

        const newMsg = inputText.trim();
        setInputText('');
        setMessages(prev => [...prev, { sender: 'user', text: newMsg }]);
        setIsThinking(true);

        try {
            const res = await axios.post(`${API_URL}/chat`, {
                user_id: userProfile.user_id,
                message: newMsg
            });

            setMessages(prev => [...prev, { sender: 'assistant', text: res.data.response }]);
            if (res.data.profile) {
                setUserProfile(prev => ({ ...prev, ...res.data.profile }));
            }
        } catch (error) {
            console.error(error);
            setMessages(prev => [...prev, { sender: 'system', text: "Failed to send message to the brain." }]);
        } finally {
            setIsThinking(false);
        }
    };

    // --- CONTINUOUS POLLING & DRAWING ---
    const drawBoundingBoxes = (faces) => {
        const canvas = canvasRef.current;
        const video = webcamRef.current?.video;
        if (!canvas || !video) return;

        // Ensure canvas matches DOM display dimensions physically for sharp rendering and correct mouse events
        const displayWidth = video.clientWidth;
        const displayHeight = video.clientHeight;

        if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
            canvas.width = displayWidth;
            canvas.height = displayHeight;
        }

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Correct aspect ratio scaling for object-fit: cover offset bounds
        const videoRatio = video.videoWidth / video.videoHeight;
        const domRatio = displayWidth / displayHeight;

        let scale, offsetX, offsetY;
        if (domRatio > videoRatio) {
            scale = displayWidth / video.videoWidth;
            offsetX = 0;
            offsetY = (displayHeight - video.videoHeight * scale) / 2;
        } else {
            scale = displayHeight / video.videoHeight;
            offsetX = (displayWidth - video.videoWidth * scale) / 2;
            offsetY = 0;
        }

        const isPrimaryUser = (face) => face.user_id === activeUserIdRef.current;

        faces.forEach((face) => {
            const box = face.box;
            const x = box.x * scale + offsetX;
            const y = box.y * scale + offsetY;
            const w = box.w * scale;
            const h = box.h * scale;
            
            if (!sessionPersonMapRef.current[face.user_id]) {
                sessionPersonMapRef.current[face.user_id] = Object.keys(sessionPersonMapRef.current).length + 1;
            }
            const personNo = sessionPersonMapRef.current[face.user_id];
            
            const isKnown = face.name && face.name.trim() !== '' && face.name !== 'Unknown User';
            const displayName = isKnown ? face.name : `Person ${personNo} (New)`;

            // ── White corner markers ──
            const cornerLen = Math.min(w, h) * 0.25;
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.lineWidth = 2.5;
            ctx.lineCap = 'round';

            // Top-left
            ctx.beginPath();
            ctx.moveTo(x, y + cornerLen);
            ctx.lineTo(x, y);
            ctx.lineTo(x + cornerLen, y);
            ctx.stroke();

            // Top-right
            ctx.beginPath();
            ctx.moveTo(x + w - cornerLen, y);
            ctx.lineTo(x + w, y);
            ctx.lineTo(x + w, y + cornerLen);
            ctx.stroke();

            // Bottom-left
            ctx.beginPath();
            ctx.moveTo(x, y + h - cornerLen);
            ctx.lineTo(x, y + h);
            ctx.lineTo(x + cornerLen, y + h);
            ctx.stroke();

            // Bottom-right
            ctx.beginPath();
            ctx.moveTo(x + w - cornerLen, y + h);
            ctx.lineTo(x + w, y + h);
            ctx.lineTo(x + w, y + h - cornerLen);
            ctx.stroke();

            // ── Name label above the box ──
            ctx.font = 'bold 13px "Inter", "Segoe UI", sans-serif';
            const nameW = ctx.measureText(displayName).width;
            const labelPadH = 8;
            const labelH = 22;
            const labelW = nameW + labelPadH * 2;
            const labelX = x;
            const labelY = y > labelH + 4 ? y - labelH - 4 : y + h + 4;

            // Label background
            ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
            ctx.beginPath();
            ctx.roundRect(labelX, labelY, labelW, labelH, 4);
            ctx.fill();

            // Label text
            ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
            ctx.fillText(displayName, labelX + labelPadH, labelY + 15);
        });
    };

    // Polling Loop Effect
    // Auto polling effect
    const pollBackend = async () => {
        if (isRecording || isIdentifying || isPollingRef.current) return; // Don't interrupt voice recognition or overlap

        const imageSrc = captureImage();
        if (!imageSrc) return;

        isPollingRef.current = true;
        try {
            const res = await axios.post(`${API_URL}/poll_vision`, { image: imageSrc });
            const faces = res.data.faces || [];
            setDetectedFaces(faces);
            detectedFacesRef.current = faces;
            drawBoundingBoxes(faces);

            // Auto-update main user profile if someone new appears
            if (faces.length > 0) {
                const activeFaceStillPresent = faces.find(f => f.user_id === activeUserIdRef.current);
                const newUnknownFace = faces.find(f => f.is_new || f.name === 'Unknown User');
                
                if (newUnknownFace && newUnknownFace.user_id !== activeUserIdRef.current) {
                    // Update the active context to the new/unknown face
                    activeUserIdRef.current = newUnknownFace.user_id;
                    const personNo = sessionPersonMapRef.current[newUnknownFace.user_id] || '?';
                    setUserProfile({ user_id: newUnknownFace.user_id, name: `Person ${personNo}`, status: "Pending Identify" });
                    
                    setMessages(prev => [...prev, { 
                        sender: 'assistant', 
                        text: `I noticed a new face! Person ${personNo}, could you please tell me your name so I can update my memory?`
                    }]);
                } else if (!activeFaceStillPresent && !newUnknownFace) {
                    // Switch back to an existing known user if the active user left
                    const newFocus = faces[0];
                    if (newFocus.user_id !== activeUserIdRef.current) {
                        activeUserIdRef.current = newFocus.user_id;
                        setUserProfile({ user_id: newFocus.user_id, name: newFocus.name, status: "Auto-Detected" });
                        setMessages(prev => [...prev, { sender: 'assistant', text: `Welcome back, ${newFocus.name}! Switching conversation context.` }]);
                    }
                }
            }
        } catch (error) {
            // Silently fail continuous polling to not spam console
        } finally {
            isPollingRef.current = false;
        }
    };

    // Local Motion Tracker
    const checkMotion = () => {
        if (isRecording || isIdentifying || isInputFocusedRef.current) return false;
        const video = webcamRef.current?.video;
        if (!video || video.readyState !== 4) return false;

        if (!motionCanvasRef.current) {
            const canvas = document.createElement("canvas");
            canvas.width = 64;
            canvas.height = 48;
            motionCanvasRef.current = canvas;
        }

        const motionCanvas = motionCanvasRef.current;
        const ctx = motionCanvas.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(video, 0, 0, motionCanvas.width, motionCanvas.height);

        const currentFrame = ctx.getImageData(0, 0, motionCanvas.width, motionCanvas.height);
        const currentData = currentFrame.data;
        const lastData = lastFrameDataRef.current;

        let motionDetected = false;

        if (lastData) {
            let diffPixels = 0;
            const faces = detectedFacesRef.current;
            const videoW = video.videoWidth;
            const videoH = video.videoHeight;
            const motionW = motionCanvas.width;
            const motionH = motionCanvas.height;

            const userStats = faces.map(f => {
                const bx = (f.box.x / videoW) * motionW;
                const by = (f.box.y / videoH) * motionH;
                const bw = (f.box.w / videoW) * motionW;
                const bh = (f.box.h / videoH) * motionH;
                const paddingX = bw * 0.75;
                const paddingY = bh * 0.75;
                return {
                    face: f, bx, by, bw, bh,
                    minX: Math.max(0, bx - paddingX),
                    minY: Math.max(0, by - paddingY),
                    maxX: Math.min(motionW, bx + bw + paddingX),
                    maxY: Math.min(motionH, by + bh + paddingY),
                    sumX: 0, sumY: 0, count: 0
                };
            });

            for (let y = 0; y < motionH; y++) {
                for (let x = 0; x < motionW; x++) {
                    const i = (y * motionW + x) * 4;
                    const diffR = Math.abs(currentData[i] - lastData[i]);
                    const diffG = Math.abs(currentData[i + 1] - lastData[i + 1]);
                    const diffB = Math.abs(currentData[i + 2] - lastData[i + 2]);

                    if ((diffR + diffG + diffB) / 3 > 45) {
                        let matched = false;
                        for (const u of userStats) {
                            if (x >= u.minX && x <= u.maxX && y >= u.minY && y <= u.maxY) {
                                u.sumX += x;
                                u.sumY += y;
                                u.count++;
                                matched = true;
                                break;
                            }
                        }
                        if (!matched) diffPixels++;
                    }
                }
            }

            let facesMovedLocally = false;
            for (const u of userStats) {
                const paddedArea = (u.maxX - u.minX) * (u.maxY - u.minY);
                if (u.count > paddedArea * 0.02) {
                    const motionCX = u.sumX / u.count;
                    const motionCY = u.sumY / u.count;
                    const currentCX = u.bx + u.bw / 2;
                    const currentCY = u.by + u.bh / 2;
                    const newCX = currentCX + (motionCX - currentCX) * 0.15;
                    const newCY = currentCY + (motionCY - currentCY) * 0.15;

                    u.face.box.x = (newCX - u.bw / 2) * (videoW / motionW);
                    u.face.box.y = (newCY - u.bh / 2) * (videoH / motionH);
                    facesMovedLocally = true;
                }
            }

            if (facesMovedLocally) {
                setDetectedFaces([...faces]);
                drawBoundingBoxes(faces);
            }

            if (diffPixels > (motionW * motionH) * 0.03) {
                motionDetected = true;
            }
        }

        lastFrameDataRef.current = new Uint8ClampedArray(currentData);
        return motionDetected;
    };

    const checkMotionAndPoll = async () => {
        if (!isAutoDetectEnabledRef.current) return;
        const hasMotion = checkMotion();
        if (hasMotion) {
            pollBackend();
        }
    };

    // Trigger polling loop
    useEffect(() => {
        const intervalId = setInterval(checkMotionAndPoll, 500); // Check motion every 500ms
        return () => {
            clearInterval(intervalId);
        };
    }, [isRecording, isIdentifying, userProfile]);

    return (
        <div className="container">
            {/* Left Side: Interactive Video Feed */}
            <div className="left-panel">
                <div className="video-container" style={{ position: 'relative' }}>
                    <Webcam
                        audio={false}
                        ref={webcamRef}
                        screenshotFormat="image/jpeg"
                        className="videoFeed"
                        mirrored={false} /* Disabled mirror so X,Y bounding boxes match perfectly */
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                    />
                    {/* Overlay Canvas for Bounding Boxes */}
                    <canvas
                        ref={canvasRef}
                        style={{
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            width: '100%',
                            height: '100%',
                            objectFit: 'cover',
                            pointerEvents: 'none'
                        }}
                    />
                </div>
                <div className="controls">
                    <p>Welcome! Scan your face & voice to log in.</p>
                    <div style={{ display: 'flex', gap: '10px', justifyContent: 'center', marginBottom: '10px' }}>
                        <button
                            className="primary-btn"
                            onClick={handleIdentify}
                            disabled={isIdentifying || isRecording}
                        >
                            {isRecording ? (
                                <><Mic className="animate-pulse" /> Recording 3s Audio...</>
                            ) : isIdentifying ? (
                                <><Loader2 className="animate-spin" /> Processing...</>
                            ) : (
                                <><UserCircle2 /> Quick Identify</>
                            )}
                        </button>
                    </div>
                </div>
            </div>

            {/* Right Side: Interaction Chat Window */}
            <div className="right-panel">
                <div className="chat-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                        <h2>
                            <BrainCircuit className="text-blue-500" />
                            <span className="status-dot"></span>
                            Digital Brain Chat
                        </h2>
                        <div className="user-info">
                            User: {userProfile ? userProfile.name : 'Not Identified'} | {userProfile ? userProfile.status : 'Awaiting Input'}
                        </div>
                    </div>

                    {/* Top Right Buttons */}
                    <div style={{ display: 'flex', gap: '8px', zIndex: 10 }}>
                        <button
                            style={{
                                padding: '6px 12px',
                                borderRadius: '20px',
                                backgroundColor: isAutoDetectEnabled ? 'rgba(16, 185, 129, 0.9)' : 'rgba(107, 114, 128, 0.9)',
                                color: 'white',
                                border: 'none',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '6px',
                                fontSize: '12px',
                                fontWeight: 'bold',
                                boxShadow: '0 2px 5px rgba(0,0,0,0.2)',
                                transition: 'all 0.3s ease'
                            }}
                            onClick={() => {
                                const nextState = !isAutoDetectEnabled;
                                setIsAutoDetectEnabled(nextState);
                                isAutoDetectEnabledRef.current = nextState;
                                if (!nextState) {
                                    setDetectedFaces([]);
                                    detectedFacesRef.current = [];
                                    drawBoundingBoxes([]);
                                }
                            }}
                            disabled={isIdentifying || isRecording}
                        >
                            <span style={{
                                width: '8px',
                                height: '8px',
                                backgroundColor: isAutoDetectEnabled ? '#fff' : '#d1d5db',
                                borderRadius: '50%',
                                display: 'inline-block',
                                boxShadow: isAutoDetectEnabled ? '0 0 5px #fff' : 'none'
                            }} />
                            AUTO DETECT
                        </button>

                        <button
                            style={{
                                padding: '6px 12px',
                                borderRadius: '20px',
                                backgroundColor: isRecording ? 'rgba(239, 68, 68, 0.9)' : 'rgba(59, 130, 246, 0.9)',
                                color: 'white',
                                border: 'none',
                                cursor: isIdentifying ? 'not-allowed' : 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '6px',
                                fontSize: '12px',
                                fontWeight: 'bold',
                                boxShadow: '0 2px 5px rgba(0,0,0,0.2)',
                                transition: 'all 0.3s ease'
                            }}
                            onClick={handleIdentify}
                            disabled={isIdentifying || isRecording}
                        >
                            {isRecording ? <Mic size={14} className="animate-pulse" /> : <Mic size={14} />}
                            {isRecording ? "RECORDING..." : "VOICE RECOGNITION"}
                        </button>
                    </div>
                </div>

                <div className="chat-messages">
                    {messages.map((msg, idx) => (
                        <div key={idx} className={`message ${msg.sender}-msg`}>
                            {msg.text}
                        </div>
                    ))}
                    {isThinking && (
                        <div className="system-msg flex items-center gap-2">
                            <Loader2 className="animate-spin w-4 h-4" /> Digital Brain is thinking...
                        </div>
                    )}
                </div>

                <form className="chat-input" onSubmit={handleSendMessage}>
                    <input
                        type="text"
                        placeholder="Type your message here..."
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        onFocus={() => isInputFocusedRef.current = true}
                        onBlur={() => isInputFocusedRef.current = false}
                        disabled={!userProfile || isThinking}
                    />
                    <button
                        type="submit"
                        className="primary-btn icon-only"
                        disabled={!userProfile || !inputText.trim() || isThinking}
                    >
                        <Send size={18} />
                    </button>
                </form>
            </div>
        </div>
    );
}

// Helper function to encode raw audio data to standard WAV
function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const writeString = (view, offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM format
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return new Blob([view], { type: 'audio/wav' });
}

export default App;
