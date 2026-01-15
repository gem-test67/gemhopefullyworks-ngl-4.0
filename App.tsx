
import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { Message, Role, PowerUp } from './types';
import { sendMessageToGem } from './services/geminiService';
import { GemIcon, UserIcon, EnterFullScreenIcon, ExitFullScreenIcon, EyeIcon, EyeOffIcon } from './components/Icons';
import SmartHomeControls from './components/SmartHomeControls';
import SpotifyControls from './components/SpotifyControls';
import YouTubeMusicControls from './components/YouTubeMusicControls';
import AvatarView from './components/AvatarView';
import ChatInput from './components/ChatInput';
import PowerUpControls from './components/PowerUpControls';
// FIX: Aliased `Blob` to `GenAIBlob` to prevent naming conflicts with the browser's native `Blob` type.
import { GoogleGenAI, LiveServerMessage, Modality, Blob as GenAIBlob } from '@google/genai';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { SYSTEM_INSTRUCTION } from './constants';

// --- Audio Utility Functions ---
function encode(bytes: Uint8Array): string {
    let binary = '';
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}

function decode(base64: string): Uint8Array {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
}

async function decodeAudioData(data: Uint8Array, ctx: AudioContext, sampleRate: number, numChannels: number): Promise<AudioBuffer> {
    const dataInt16 = new Int16Array(data.buffer);
    const frameCount = dataInt16.length / numChannels;
    const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

    for (let channel = 0; channel < numChannels; channel++) {
        const channelData = buffer.getChannelData(channel);
        for (let i = 0; i < frameCount; i++) {
            channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
        }
    }
    return buffer;
}

// FIX: Updated the return type to use the aliased `GenAIBlob` to ensure type compatibility with the Gemini API.
function createBlob(data: Float32Array): GenAIBlob {
    const l = data.length;
    const int16 = new Int16Array(l);
    for (let i = 0; i < l; i++) {
        int16[i] = data[i] * 32768;
    }
    return {
        data: encode(new Uint8Array(int16.buffer)),
        mimeType: 'audio/pcm;rate=16000',
    };
}
// --- End Audio Utility Functions ---

const blobToBase64 = (blob: Blob): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64data = reader.result as string;
            resolve(base64data.split(',')[1]);
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
};

interface TranscriptionEntry {
    speaker: 'user' | 'gem';
    text: string;
}

const App = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: Role.GEM,
      content: "[HAPPY] Hello there! I'm Gem, your new AI companion. I've been powered up and I'm practically buzzing with energy! What sort of fun are we getting into today?",
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'avatar' | 'controls'>('avatar');
  const [gemEmotion, setGemEmotion] = useState<string>('HAPPY');
  const [headRotation, setHeadRotation] = useState<THREE.Quaternion | null>(null);
  const [isStreamingVideo, setIsStreamingVideo] = useState(false);
  const [powerUp, setPowerUp] = useState<PowerUp>('standard');
  const overlayChatEndRef = useRef<HTMLDivElement>(null);

  // Live Chat State
  const [liveStatus, setLiveStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [transcriptionHistory, setTranscriptionHistory] = useState<TranscriptionEntry[]>([]);
  const [currentUserTranscription, setCurrentUserTranscription] = useState('');
  const [currentGemTranscription, setCurrentGemTranscription] = useState('');
  const transcriptEndRef = useRef<HTMLDivElement>(null);
  const sessionStatusRef = useRef<'idle' | 'connecting' | 'connected' | 'closing'>('idle');
  
  // App container ref for fullscreen
  const appRef = useRef<HTMLDivElement>(null);
  const [isFullScreen, setIsFullScreen] = useState(false);

  const sessionPromiseRef = useRef<Promise<any> | null>(null);
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const mediaStreamSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const nextStartTimeRef = useRef(0);
  const audioSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const analyserNodeRef = useRef<AnalyserNode | null>(null);
  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);
  const wakeLockSentinelRef = useRef<any | null>(null);

  // Vision Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const videoStreamIntervalRef = useRef<number | null>(null);

  // Face Tracking Refs
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const faceTrackingFrameIdRef = useRef<number | null>(null);

  const cleanUpAvatarFeatures = () => {
    console.log('Cleaning up avatar features...');
    sessionPromiseRef.current?.then(session => session.close()).catch(console.error);
    
    if (micStreamRef.current) {
        micStreamRef.current.getTracks().forEach(track => track.stop());
        micStreamRef.current = null;
    }
    if (scriptProcessorRef.current) {
        scriptProcessorRef.current.disconnect();
        scriptProcessorRef.current.onaudioprocess = null;
        scriptProcessorRef.current = null;
    }
    if (mediaStreamSourceRef.current) {
        mediaStreamSourceRef.current.disconnect();
        mediaStreamSourceRef.current = null;
    }
    if (inputAudioContextRef.current && inputAudioContextRef.current.state !== 'closed') {
        inputAudioContextRef.current.close().catch(console.error);
    }
    if (outputAudioContextRef.current && outputAudioContextRef.current.state !== 'closed') {
        outputAudioContextRef.current.close().catch(console.error);
    }
    
    inputAudioContextRef.current = null;
    outputAudioContextRef.current = null;
    analyserNodeRef.current = null;
    setAnalyserNode(null);

    for (const source of audioSourcesRef.current.values()) {
        try { source.stop(); } catch (e) {}
    }
    audioSourcesRef.current.clear();
    nextStartTimeRef.current = 0;
    
    if (faceTrackingFrameIdRef.current) {
      cancelAnimationFrame(faceTrackingFrameIdRef.current);
      faceTrackingFrameIdRef.current = null;
    }
    const video = videoRef.current;
    if (video && video.srcObject) {
      (video.srcObject as MediaStream).getTracks().forEach(track => track.stop());
      video.srcObject = null;
    }
    faceLandmarkerRef.current?.close();
    faceLandmarkerRef.current = null;
    setHeadRotation(null);

    // Reset state after cleanup
    sessionPromiseRef.current = null;
    setLiveStatus('disconnected');
    sessionStatusRef.current = 'idle';
  };

  const playAudio = async (base64Audio: string) => {
    const ctx = outputAudioContextRef.current;
    const localAnalyserNode = analyserNodeRef.current;

    if (!ctx || !localAnalyserNode) {
      console.error("Output audio context not available for playback.");
      return;
    }

    if (ctx.state === 'suspended') {
      await ctx.resume();
    }

    const decodedAudio = decode(base64Audio);
    const audioBuffer = await decodeAudioData(decodedAudio, ctx, 24000, 1);
    
    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(localAnalyserNode);

    const currentTime = ctx.currentTime;
    const startTime = Math.max(currentTime, nextStartTimeRef.current);
    source.start(startTime);
    nextStartTimeRef.current = startTime + audioBuffer.duration;

    audioSourcesRef.current.add(source);
    source.onended = () => audioSourcesRef.current.delete(source);
  };

  const startAvatarFeatures = async () => {
    if (sessionStatusRef.current !== 'idle') return;
    
    sessionStatusRef.current = 'connecting';
    setLiveStatus('connecting');

    // --- Face Tracking & Camera Setup ---
    try {
      const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm");
      faceLandmarkerRef.current = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
              modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
              delegate: "GPU",
          },
          outputFacialTransformationMatrixes: true,
          runningMode: "VIDEO",
          numFaces: 1,
      });
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      const video = videoRef.current;
      if (video) {
          video.srcObject = stream;
          video.onloadeddata = () => {
              video.play();
              predictWebcam();
          };
      }
    } catch(err) {
      console.error("Camera or Face Tracking failed: ", err);
      alert("[SAD] I couldn't access your camera. Please check your browser permissions!");
    }

    // --- Live Chat Setup ---
    const API_KEY = process.env.API_KEY;
    if (!API_KEY) {
        alert("[SAD] I'm having trouble connecting to my brain! My developer needs to check the API key configuration.");
        cleanUpAvatarFeatures();
        return;
    }

    setTranscriptionHistory([]);
    setCurrentUserTranscription('');
    setCurrentGemTranscription('');
    
    try {
        const audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } 
        });
        micStreamRef.current = audioStream;
        
        if (!outputAudioContextRef.current) {
            const ctx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
            outputAudioContextRef.current = ctx;
            const analyser = ctx.createAnalyser();
            analyser.fftSize = 256;
            analyser.connect(ctx.destination);
            analyserNodeRef.current = analyser;
            setAnalyserNode(analyser);
        }
        if (outputAudioContextRef.current.state === 'suspended') await outputAudioContextRef.current.resume();

        if (!inputAudioContextRef.current) {
             inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
        }
        if (inputAudioContextRef.current.state === 'suspended') await inputAudioContextRef.current.resume();

        const ai = new GoogleGenAI({ apiKey: API_KEY });
        sessionPromiseRef.current = ai.live.connect({
            model: 'gemini-2.5-flash-native-audio-preview-12-2025',
            config: {
                responseModalities: [Modality.AUDIO], inputAudioTranscription: {}, outputAudioTranscription: {},
                speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } } },
                systemInstruction: SYSTEM_INSTRUCTION,
            },
            callbacks: {
                onopen: () => {
                    console.log('Session opened.');
                    sessionStatusRef.current = 'connected';
                    setLiveStatus('connected');
                    if (!micStreamRef.current || !inputAudioContextRef.current) return;
                    mediaStreamSourceRef.current = inputAudioContextRef.current.createMediaStreamSource(micStreamRef.current);
                    scriptProcessorRef.current = inputAudioContextRef.current.createScriptProcessor(4096, 1, 1);
                    scriptProcessorRef.current.onaudioprocess = (event) => {
                        const inputData = event.inputBuffer.getChannelData(0);
                        const pcmBlob = createBlob(inputData);
                        sessionPromiseRef.current?.then((session) => session.sendRealtimeInput({ media: pcmBlob }));
                    };
                    mediaStreamSourceRef.current.connect(scriptProcessorRef.current);
                    scriptProcessorRef.current.connect(inputAudioContextRef.current.destination);
                },
                onmessage: async (message: LiveServerMessage) => {
                    if (message.serverContent?.inputTranscription) setCurrentUserTranscription(p => p + message.serverContent.inputTranscription.text);
                    if (message.serverContent?.outputTranscription) {
                        const text = message.serverContent.outputTranscription.text;
                        const emotionMatch = text.match(/\[([A-Z]+)\]/);
                        if (emotionMatch && emotionMatch[1]) setGemEmotion(emotionMatch[1]);
                        setCurrentGemTranscription(p => p + text);
                    }
                    if (message.serverContent?.turnComplete) {
                        setCurrentUserTranscription(prevUser => {
                            setCurrentGemTranscription(prevGem => {
                                setTranscriptionHistory(p => [...p, { speaker: 'user', text: prevUser }, { speaker: 'gem', text: prevGem }]);
                                return '';
                            });
                            return '';
                        });
                    }
                    const base64Audio = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
                    if (base64Audio) await playAudio(base64Audio);
                },
                onerror: (e: ErrorEvent) => {
                    console.error('Session error:', e);
                    if (sessionStatusRef.current !== 'closing') {
                        alert(`A live connection error occurred: ${e.message}`);
                        sessionStatusRef.current = 'closing';
                        cleanUpAvatarFeatures();
                    }
                },
                onclose: (e: CloseEvent) => {
                    console.log('Session closed.', `Code: ${e.code}`, `Reason: ${e.reason}`);
                    if (sessionStatusRef.current !== 'closing') {
                        console.warn('Unexpected session closure. Reconnecting...');
                        cleanUpAvatarFeatures();
                        setTimeout(() => {
                            if (activeTab === 'avatar') {
                                startAvatarFeatures();
                            }
                        }, 1000);
                    }
                },
            }
        });
    } catch (error) {
        console.error('Failed to start live chat:', error);
        alert("Could not start live chat. Please ensure microphone permissions.");
        cleanUpAvatarFeatures();
    }
  };

  let lastVideoTime = -1;
  const predictWebcam = () => {
    const video = videoRef.current;
    if (!video || !faceLandmarkerRef.current) return;

    if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const results = faceLandmarkerRef.current.detectForVideo(video, performance.now());

        if (results.facialTransformationMatrixes && results.facialTransformationMatrixes.length > 0) {
            const matrixData = results.facialTransformationMatrixes[0].data;
            const matrix = new THREE.Matrix4().fromArray(matrixData);
            const rotationQuaternion = new THREE.Quaternion().setFromRotationMatrix(matrix);
            
            rotationQuaternion.x *= -1; 
            rotationQuaternion.y *= -1;

            setHeadRotation(rotationQuaternion);
        } else {
            setHeadRotation(null);
        }
    }
    faceTrackingFrameIdRef.current = requestAnimationFrame(predictWebcam);
  };

  useEffect(() => {
    if (activeTab === 'avatar') {
        startAvatarFeatures();
    }
    return () => {
      if (sessionStatusRef.current === 'connected' || sessionStatusRef.current === 'connecting') {
          sessionStatusRef.current = 'closing';
          cleanUpAvatarFeatures();
      }
    };
  }, [activeTab]);

  const scrollToBottom = () => {
    overlayChatEndRef.current?.scrollIntoView({ behavior: 'auto' });
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading, activeTab, transcriptionHistory, currentUserTranscription, currentGemTranscription]);

  const toggleVideoStream = () => {
    if (liveStatus !== 'connected') return;
    setIsStreamingVideo(prev => !prev);
  };

  useEffect(() => {
    const startStreaming = async () => {
        if (!videoRef.current) return;
        const session = await sessionPromiseRef.current;
        if (!session) return;

        session.sendRealtimeInput({ text: "Describe what you see in the video stream." });

        videoStreamIntervalRef.current = window.setInterval(() => {
            const video = videoRef.current;
            if (!video || video.paused || video.ended) return;

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            if (!ctx) return;
            
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            canvas.toBlob(async (blob) => {
                if (blob) {
                    try {
                        const base64Data = await blobToBase64(blob);
                        const currentSession = await sessionPromiseRef.current;
                        currentSession?.sendRealtimeInput({ media: { data: base64Data, mimeType: 'image/jpeg' } });
                    } catch (error) {
                        console.error("Error processing video frame:", error);
                    }
                }
            }, 'image/jpeg', 0.7);
        }, 500); // ~2 frames per second
    };

    const stopStreaming = () => {
        if (videoStreamIntervalRef.current) {
            clearInterval(videoStreamIntervalRef.current);
            videoStreamIntervalRef.current = null;
        }
    };

    if (isStreamingVideo && liveStatus === 'connected') {
        startStreaming();
    } else {
        stopStreaming();
    }

    return () => {
        stopStreaming();
    };
  }, [isStreamingVideo, liveStatus]);

  useEffect(() => {
    const acquireWakeLock = async () => {
        if ('wakeLock' in navigator) {
            try {
                wakeLockSentinelRef.current = await navigator.wakeLock.request('screen');
                console.log('Screen Wake Lock is active.');
                wakeLockSentinelRef.current.addEventListener('release', () => {
                    console.log('Screen Wake Lock was released.');
                });
            } catch (err: any) {
                console.error(`Could not acquire wake lock: ${err.name}, ${err.message}`);
            }
        } else {
            console.log('Wake Lock API is not supported by this browser.');
        }
    };
    const handleVisibilityChange = () => {
        if (wakeLockSentinelRef.current !== null && document.visibilityState === 'visible') {
            acquireWakeLock();
        }
    };
    acquireWakeLock();
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
        if (wakeLockSentinelRef.current) {
            wakeLockSentinelRef.current.release();
            wakeLockSentinelRef.current = null;
        }
        document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);
  
  useEffect(() => {
    const handleFullScreenChange = () => {
      setIsFullScreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFullScreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullScreenChange);
  }, []);

  const toggleFullScreen = () => {
    if (!document.fullscreenElement) {
        appRef.current?.requestFullscreen().catch(err => {
        alert(`Error attempting to enable full-screen mode: ${err.message} (${err.name})`);
      });
    } else {
      document.exitFullscreen();
    }
  };

  const handleSendMessage = async (userInput: string) => {
    if (!userInput.trim() || isLoading) return;

    const newUserMessage: Message = { role: Role.USER, content: userInput };
    setMessages((prevMessages) => [...prevMessages, newUserMessage]);
    setIsLoading(true);

    try {
      const history = messages.map(msg => ({
        role: msg.role === Role.USER ? 'user' : 'model',
        parts: [{ text: msg.content }]
      }));
      
      let toolConfig: any;
      if (powerUp === 'maps') {
          try {
              const position = await new Promise<GeolocationPosition>((resolve, reject) => {
                  navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 5000 });
              });
              toolConfig = {
                  retrievalConfig: {
                      latLng: {
                          latitude: position.coords.latitude,
                          longitude: position.coords.longitude
                      }
                  }
              };
          } catch (error) {
              console.error("Could not get geolocation", error);
              const errorMessage: Message = {
                  role: Role.GEM,
                  content: "[SAD] I tried to find you, but I couldn't get your location. Please enable location permissions so I can use my Maps power-up!",
              };
              setMessages((prev) => [...prev, errorMessage]);
              setIsLoading(false);
              return;
          }
      }

      const gemResponse = await sendMessageToGem(history, userInput, powerUp, toolConfig);
      
      const match = gemResponse.text.match(/^\[([A-Z]+)\]\s*/);
      if (match && match[1]) {
        setGemEmotion(match[1]);
      }

      const newGemMessage: Message = {
        role: Role.GEM,
        content: gemResponse.text,
        sources: gemResponse.sources,
      };
      setMessages((prevMessages) => [...prevMessages, newGemMessage]);
    } catch (error: any) {
      console.error('Error sending message to Gem:', error);
      const errorMessage: Message = {
        role: Role.GEM,
        content: "[SAD] Oh dear, something went wrong and I couldn't process your message. Please try again in a moment.",
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const renderMiniMessage = (msg: Message, index: number) => {
    const isGem = msg.role === Role.GEM;
    let displayContent = msg.content;
    const match = msg.content.match(/^\[([A-Z]+)\]\s*/);
    if (isGem && match) {
      displayContent = msg.content.substring(match[0].length);
    }

    return (
      <div key={`mini-${index}`} className={`flex items-start gap-2 text-sm ${isGem ? 'text-purple-200' : 'text-blue-200'}`}>
        <div className="flex-shrink-0 w-5 h-5">
            {isGem ? <GemIcon className="w-full h-full" /> : <UserIcon className="w-full h-full" />}
        </div>
        <p className="flex-1 break-words">{displayContent}</p>
      </div>
    );
  }

  const renderTranscription = (entry: TranscriptionEntry, index: number) => {
    const isGem = entry.speaker === 'gem';
    const Icon = isGem ? GemIcon : UserIcon;
    const textColor = isGem ? 'text-white/90' : 'text-gray-300/90';
    
    return (
        <div key={index} className="flex items-start gap-3 text-sm">
            <div className={`w-6 h-6 flex items-center justify-center flex-shrink-0`}>
                <Icon className={`w-4 h-4 ${textColor}`} />
            </div>
            <div className="flex-1 pt-0.5">
                <p className={textColor}>{entry.text}</p>
            </div>
        </div>
    );
};

  return (
    <div ref={appRef} className="flex flex-col h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900">
      <video ref={videoRef} style={{ display: 'none' }} autoPlay playsInline></video>
      <header className={`flex-shrink-0 flex items-center justify-between p-4 shadow-lg bg-gray-900/50 backdrop-blur-sm border-b border-purple-500/20 z-20 ${isFullScreen ? 'hidden' : ''}`}>
        <div className="flex items-center">
            <GemIcon className="w-8 h-8 text-purple-400" />
            <h1 className="ml-3 text-2xl font-bold tracking-wider text-purple-300">Gem</h1>
        </div>
        {liveStatus === 'connected' && (
            <div className="flex items-center gap-2" title="Live chat is active">
                <span className="relative flex h-3 w-3">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                </span>
                <span className="text-red-400 text-sm font-medium">LIVE</span>
            </div>
        )}
      </header>

      {/* Tabs */}
      <div className={`flex-shrink-0 flex border-b border-purple-500/20 px-4 md:px-6 z-20 bg-gray-900/30 ${isFullScreen ? 'hidden' : ''}`}>
        <button
          onClick={() => setActiveTab('avatar')}
          className={`py-3 px-4 text-sm font-medium border-b-2 transition-colors duration-200 focus:outline-none ${
            activeTab === 'avatar'
              ? 'border-purple-400 text-purple-300'
              : 'border-transparent text-gray-400 hover:text-white'
          }`}
        >
          Avatar
        </button>
        <button
          onClick={() => setActiveTab('controls')}
          className={`py-3 px-4 text-sm font-medium border-b-2 transition-colors duration-200 focus:outline-none ${
            activeTab === 'controls'
              ? 'border-purple-400 text-purple-300'
              : 'border-transparent text-gray-400 hover:text-white'
          }`}
        >
          Controls
        </button>
      </div>
      
      <div className="flex-1 relative overflow-hidden">
        {/* Main Content Area */}
        <div className={`absolute inset-0 transition-opacity duration-300 ${activeTab === 'controls' ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
          <div className="h-full overflow-y-auto p-4 md:p-6">
            <div className="space-y-4">
              <PowerUpControls activePowerUp={powerUp} onSetPowerUp={setPowerUp} isLoading={isLoading} />
              <SmartHomeControls onSendCommand={handleSendMessage} isLoading={isLoading} />
              <SpotifyControls onSendCommand={handleSendMessage} isLoading={isLoading} />
              <YouTubeMusicControls onSendCommand={handleSendMessage} isLoading={isLoading} />
            </div>
          </div>
        </div>

        <div className={`absolute inset-0 transition-opacity duration-300 ${activeTab === 'avatar' ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
            <AvatarView currentEmotion={gemEmotion} analyserNode={analyserNode} headRotation={headRotation} />
            {/* Live Transcription Overlay */}
            <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/70 to-transparent pointer-events-none">
                <div className="max-w-4xl mx-auto p-4 bg-black/30 backdrop-blur-sm rounded-lg max-h-[12.5vh] overflow-y-auto pointer-events-auto">
                    <div className="space-y-3">
                        {transcriptionHistory.map(renderTranscription)}
                        {currentUserTranscription && (
                            <div className="flex items-start gap-3 text-sm opacity-70">
                                <div className="w-6 h-6 flex items-center justify-center flex-shrink-0"><UserIcon className="w-4 h-4 text-gray-300/90" /></div>
                                <div className="flex-1 pt-0.5"><p className="text-gray-300/90 italic">{currentUserTranscription}</p></div>
                            </div>
                        )}
                        {currentGemTranscription && (
                            <div className="flex items-start gap-3 text-sm opacity-70">
                                <div className="w-6 h-6 flex items-center justify-center flex-shrink-0"><GemIcon className="w-4 h-4 text-white/90" /></div>
                                <div className="flex-1 pt-0.5"><p className="text-white/90 italic">{currentGemTranscription}</p></div>
                            </div>
                        )}
                         <div ref={transcriptEndRef} />
                    </div>
                </div>
            </div>
             {/* UI Buttons Overlay */}
             <div className="absolute top-4 right-4 z-30 flex flex-col gap-3">
                <button
                    onClick={toggleFullScreen}
                    className="p-2 text-white/70 hover:text-white bg-black/30 rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-purple-500"
                    aria-label={isFullScreen ? 'Exit full screen' : 'Enter full screen'}
                    title={isFullScreen ? 'Exit full screen' : 'Enter full screen'}
                >
                    {isFullScreen ? <ExitFullScreenIcon className="w-6 h-6" /> : <EnterFullScreenIcon className="w-6 h-6" />}
                </button>
                <button
                    onClick={toggleVideoStream}
                    disabled={liveStatus !== 'connected'}
                    className="p-2 text-white/70 hover:text-white bg-black/30 rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-purple-500"
                    aria-label={isStreamingVideo ? "Stop video stream" : "Start video stream"}
                    title={isStreamingVideo ? "Stop video stream" : "Start video stream"}
                >
                    {isStreamingVideo ? (
                        <EyeOffIcon className="w-6 h-6 text-red-400" />
                    ) : (
                        <EyeIcon className="w-6 h-6" />
                    )}
                </button>
             </div>
        </div>
      </div>
      
      {/* Chat Overlay for Controls Tab */}
      {activeTab === 'controls' && (
          <div className="fixed bottom-4 left-4 z-10 w-full max-w-sm">
            <div className="bg-black/40 backdrop-blur-md rounded-lg shadow-2xl flex flex-col max-h-[40vh] overflow-hidden">
                <div className="flex-1 p-3 overflow-y-auto">
                    <div className="space-y-3">
                        {messages.slice(-5).map(renderMiniMessage)}
                        {isLoading && (
                           <div className="flex items-start gap-2 text-sm text-purple-200">
                             <div className="flex-shrink-0 w-5 h-5"><GemIcon className="w-full h-full" /></div>
                             <div className="flex items-center space-x-1.5 pt-1">
                                <div className="w-2 h-2 bg-purple-300 rounded-full animate-pulse [animation-delay:-0.3s]"></div>
                                <div className="w-2 h-2 bg-purple-300 rounded-full animate-pulse [animation-delay:-0.15s]"></div>
                                <div className="w-2 h-2 bg-purple-300 rounded-full animate-pulse"></div>
                              </div>
                           </div>
                        )}
                        <div ref={overlayChatEndRef} />
                    </div>
                </div>
                <div className="flex-shrink-0 p-2 border-t border-white/10">
                    <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} variant="overlay" />
                </div>
            </div>
          </div>
      )}
    </div>
  );
};

export default App;
