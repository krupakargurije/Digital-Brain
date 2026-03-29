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

        // Calculate scaling factors if the video's natural resolution differs from CSS display
        const scaleX = displayWidth / video.videoWidth;
        const scaleY = displayHeight / video.videoHeight;

        faces.forEach(face => {
            const box = face.box;
            const x = box.x * scaleX;
            const y = box.y * scaleY;
            const w = box.w * scaleX;
            const h = box.h * scaleY;

            // Draw Box
            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, w, h);

            // Draw Background for Text
            ctx.fillStyle = '#3b82f6';
            ctx.fillRect(x, y - 30, w, 30);

            // Draw Name
            ctx.fillStyle = 'white';
            ctx.font = 'bold 16px Inter, sans-serif';
            ctx.fillText(`Hi, ${face.name}!`, x + 5, y - 10);
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
                // If the currently active chatting user is no longer in the frame, but someone else is
                const activeFaceStillPresent = faces.find(f => f.user_id === activeUserIdRef.current);
                
                if (!activeFaceStillPresent) {
                    const newFocus = faces[0];
                    activeUserIdRef.current = newFocus.user_id; // Switch active context
                    
                    setUserProfile({ user_id: newFocus.user_id, name: newFocus.name, status: "Auto-Detected" });
                    
                    if (newFocus.is_new) {
                        setMessages(prev => [...prev, { sender: 'assistant', text: "Hello! I noticed you are new here. I'm Digital Brain. What is your name?" }]);
                    } else {
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

    // Motion Detection Setup and Logic
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

            // Map face bounding boxes to the motion canvas scale and add padding (75% margin to ignore shifting)
            const paddedBoxes = faces.map(f => {
                const bx = (f.box.x / videoW) * motionW;
                const by = (f.box.y / videoH) * motionH;
                const bw = (f.box.w / videoW) * motionW;
                const bh = (f.box.h / videoH) * motionH;
                
                const marginX = bw * 0.75;
                const marginY = bh * 0.75;
                
                return {
                    x: Math.max(0, bx - marginX),
                    y: Math.max(0, by - marginY),
                    w: bw + marginX * 2,
                    h: bh + marginY * 2
                };
            });

            for (let y = 0; y < motionH; y++) {
                for (let x = 0; x < motionW; x++) {
                    // Check if current pixel is inside any known face box
                    let insideUser = false;
                    for (const box of paddedBoxes) {
                        if (x >= box.x && x <= box.x + box.w && y >= box.y && y <= box.y + box.h) {
                            insideUser = true;
                            break;
                        }
                    }
                    if (insideUser) continue; // Ignore motion from existing users

                    const i = (y * motionW + x) * 4;
                    const diffR = Math.abs(currentData[i] - lastData[i]);
                    const diffG = Math.abs(currentData[i + 1] - lastData[i + 1]);
                    const diffB = Math.abs(currentData[i + 2] - lastData[i + 2]);
                    
                    // Average difference > threshold
                    if ((diffR + diffG + diffB) / 3 > 45) {
                        diffPixels++;
                    }
                }
            }

            // If more than 3% of pixels changed (outside of known users)
            const motionThreshold = (motionW * motionH) * 0.03;
            if (diffPixels > motionThreshold) {
                motionDetected = true;
            }
        }

        // Save current frame for next tick
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
