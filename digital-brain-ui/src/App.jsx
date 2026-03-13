import { useState, useRef, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import { Send, UserCircle2, BrainCircuit, Mic, Loader2 } from 'lucide-react';

const API_URL = 'http://127.0.0.1:5000/api';

function App() {
    const webcamRef = useRef(null);
    const canvasRef = useRef(null);
    const isRecordingRef = useRef(false);
    const hasGreetedRef = useRef(false);
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
                const reader = new FileReader();
                reader.readAsDataURL(wavBlob);
                reader.onloadend = async () => {
                    const base64Audio = reader.result;
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
        if (isRecording || isIdentifying) return; // Don't interrupt voice recognition

        const imageSrc = captureImage();
        if (!imageSrc) return;

        try {
            const res = await axios.post(`${API_URL}/poll_vision`, { image: imageSrc });
            const faces = res.data.faces || [];
            setDetectedFaces(faces);
            drawBoundingBoxes(faces);

            // Auto-update main user profile if someone new appears and we don't have an active user
            if (faces.length > 0 && !hasGreetedRef.current) {
                hasGreetedRef.current = true;
                // Focus on the first face
                const p = faces[0];
                setUserProfile({ user_id: p.user_id, name: p.name, status: "Auto-Detected" });
                if (p.is_new) {
                    setMessages(prev => [...prev, { sender: 'assistant', text: "Hello! I noticed you are new here. I'm Digital Brain. What is your name?" }]);
                } else {
                    setMessages(prev => [...prev, { sender: 'assistant', text: `Welcome back, ${p.name}!` }]);
                }
            }
        } catch (error) {
            // Silently fail continuous polling to not spam console
        }
    };

    // Trigger polling loop
    useEffect(() => {
        const intervalId = setInterval(pollBackend, 1500);
        return () => clearInterval(intervalId);
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
                    <p>Welcome! Click the button below and speak for 3 seconds.</p>
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
                            <><UserCircle2 /> Identify Me (Face & Voice)</>
                        )}
                    </button>
                </div>
            </div>

            {/* Right Side: Interaction Chat Window */}
            <div className="right-panel">
                <div className="chat-header">
                    <h2>
                        <BrainCircuit className="text-blue-500" />
                        <span className="status-dot"></span>
                        Digital Brain Chat
                    </h2>
                    <div className="user-info">
                        User: {userProfile ? userProfile.name : 'Not Identified'} | {userProfile ? userProfile.status : 'Awaiting Input'}
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
