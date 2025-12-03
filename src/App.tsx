import React, { useState, useEffect, useRef } from 'react';
import './App.css';

interface DetectedObject {
  label: string;
  confidence: number;
  bounding_box: [number, number, number, number];
}

interface AttentionStats {
  mean: number;
  max: number;
  std: number;
}

interface ProcessingResult {
  caption: string;
  detected_objects: DetectedObject[];
  guidance: string;
  model_used: string;
  fusion_enabled: boolean;
  llm_description?: string;
  attention_stats?: AttentionStats;
}

type ProcessingMode = 'basic' | 'gpt2' | 'gpt2-mini' | 'gpt2-fusion' | 'gpt2-mini-fusion';

const API_BASE_URL = 'http://localhost:5000';

const App: React.FC = () => {
  const [cameraIndex, setCameraIndex] = useState<number>(1);
  const [isCameraRunning, setIsCameraRunning] = useState<boolean>(false);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [selectedMode, setSelectedMode] = useState<ProcessingMode>('gpt2-mini-fusion');
  const [result, setResult] = useState<ProcessingResult | null>(null);
  const [error, setError] = useState<string>('');
  const [serverStatus, setServerStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [frameCount, setFrameCount] = useState<number>(0);
  
  const imgRef = useRef<HTMLImageElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    checkServerStatus();
  }, []);

  const checkServerStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/`, { method: 'GET' });
      if (response.ok) {
        setServerStatus('online');
        setError('');
      } else {
        setServerStatus('offline');
        setError('Server returned error');
      }
    } catch (err) {
      setServerStatus('offline');
      setError('Cannot connect to server. Make sure backend is running on http://localhost:5000');
    }
  };

  const startCamera = async () => {
    setError('');
    try {
      const response = await fetch(`${API_BASE_URL}/camera/start?camera_index=${cameraIndex}`, {
        method: 'POST',
      });
      const data = await response.json();
      
      if (response.ok) {
        setIsCameraRunning(true);
        setError('');
        setTimeout(() => startFrameStreaming(), 500);
      } else {
        setError(data.error || 'Failed to start camera');
      }
    } catch (err) {
      setError('Error connecting to server');
    }
  };

  const stopCamera = async () => {
    try {
      await fetch(`${API_BASE_URL}/camera/stop`, { method: 'POST' });
      setIsCameraRunning(false);
      stopFrameStreaming();
      if (imgRef.current) {
        imgRef.current.src = '';
      }
      setFrameCount(0);
    } catch (err) {
      setError('Error stopping camera');
    }
  };

  const startFrameStreaming = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    let count = 0;
    intervalRef.current = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/camera/frame`);
        if (response.ok) {
          const blob = await response.blob();
          const url = URL.createObjectURL(blob);
          if (imgRef.current) {
            imgRef.current.src = url;
          }
          count++;
          setFrameCount(count);
        }
      } catch (err) {
        console.error('Error fetching frame:', err);
      }
    }, 100);
  };

  const stopFrameStreaming = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const processFrame = async () => {
    if (!isCameraRunning) {
      setError('Please start the camera first');
      return;
    }

    setIsProcessing(true);
    setError('');

    try {
      const response = await fetch(`${API_BASE_URL}/process_camera/${selectedMode}`, {
        method: 'POST',
      });
      
      if (response.ok) {
        const data = await response.json();
        setResult(data.data);
        setError('');
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Processing failed');
      }
    } catch (err) {
      setError('Error processing frame');
    } finally {
      setIsProcessing(false);
    }
  };

  useEffect(() => {
    return () => {
      stopFrameStreaming();
    };
  }, []);

  const getModeInfo = (mode: ProcessingMode) => {
    const info: Record<ProcessingMode, { name: string; icon: string; description: string; time: string }> = {
      basic: { name: 'Basic', icon: '⚡', description: 'Fast detection only', time: '~1s' },
      'gpt2': { name: 'GPT-2', icon: '🤖', description: 'Detailed descriptions', time: '~3-5s' },
      'gpt2-mini': { name: 'GPT-2 Mini', icon: '🏃', description: 'Fast + good quality', time: '~2-3s' },
      'gpt2-fusion': { name: 'GPT-2 + Fusion', icon: '✨', description: 'Highest quality', time: '~5-8s' },
      'gpt2-mini-fusion': { name: 'GPT-2 Mini + Fusion', icon: '⭐', description: 'Best balance', time: '~4-6s' },
    };
    return info[mode];
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🎥 Vision Processing System</h1>
        <p>Real-time AI-powered vision analysis</p>
        <div className={`server-status ${serverStatus}`}>
          {serverStatus === 'checking' && '🔄 Checking server...'}
          {serverStatus === 'online' && '✅ Server Online'}
          {serverStatus === 'offline' && '❌ Server Offline'}
        </div>
      </header>

      {serverStatus === 'offline' && (
        <div className="offline-banner">
          <h2>⚠️ Backend Server Not Running</h2>
          <p>Please start the backend server:</p>
          <code>python app.py</code>
          <button onClick={checkServerStatus} className="btn btn-primary">
            🔄 Retry Connection
          </button>
        </div>
      )}

      {serverStatus === 'online' && (
        <div className="container">
          <div className="panel camera-panel">
            <h2>📷 Camera Stream</h2>
            
            <div className="camera-controls">
              <div className="input-group">
                <label>Camera Index:</label>
                <input
                  type="number"
                  value={cameraIndex}
                  onChange={(e) => setCameraIndex(parseInt(e.target.value))}
                  disabled={isCameraRunning}
                  min="0"
                  max="5"
                />
              </div>

              <div className="button-group">
                {!isCameraRunning ? (
                  <button onClick={startCamera} className="btn btn-primary">
                    ▶️ Start Camera
                  </button>
                ) : (
                  <button onClick={stopCamera} className="btn btn-danger">
                    ⏹️ Stop Camera
                  </button>
                )}
              </div>
            </div>

            <div className="camera-stream">
              {isCameraRunning ? (
                <>
                  <img ref={imgRef} alt="Camera stream" className="camera-image" />
                  <div className="frame-counter">Frames: {frameCount}</div>
                </>
              ) : (
                <div className="placeholder">
                  <p>📷</p>
                  <p>Camera is off</p>
                  <p className="hint">Click "Start Camera" to begin</p>
                </div>
              )}
            </div>

            {error && (
              <div className="error-message">
                ⚠️ {error}
              </div>
            )}
          </div>

          <div className="panel control-panel">
            <h2>🎛️ Processing Options</h2>

            <div className="mode-selection">
              <h3>Select Processing Mode:</h3>
              
              {(['basic', 'gpt2-mini', 'gpt2', 'gpt2-mini-fusion', 'gpt2-fusion'] as ProcessingMode[]).map((mode) => {
                const info = getModeInfo(mode);
                return (
                  <div
                    key={mode}
                    className={`mode-card ${selectedMode === mode ? 'selected' : ''}`}
                    onClick={() => setSelectedMode(mode)}
                  >
                    <div className="mode-header">
                      <span className="mode-icon">{info.icon}</span>
                      <span className="mode-name">{info.name}</span>
                    </div>
                    <div className="mode-description">{info.description}</div>
                    <div className="mode-time">{info.time}</div>
                  </div>
                );
              })}
            </div>

            <button
              onClick={processFrame}
              disabled={!isCameraRunning || isProcessing}
              className="btn btn-process"
            >
              {isProcessing ? '⏳ Processing...' : '🚀 Process Frame'}
            </button>

            {result && (
              <div className="results">
                <h3>📊 Results</h3>

                <div className="result-section">
                  <h4>Caption:</h4>
                  <p className="caption">{result.caption}</p>
                </div>

                <div className="result-section">
                  <h4>Detected Objects ({result.detected_objects.length}):</h4>
                  <div className="objects-list">
                    {result.detected_objects.map((obj, idx) => (
                      <div key={idx} className="object-item">
                        <span className="object-label">{obj.label}</span>
                        <span className="object-confidence">
                          {(obj.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="result-section">
                  <h4>Spatial Guidance:</h4>
                  <p className="guidance">{result.guidance}</p>
                </div>

                {result.llm_description && (
                  <div className="result-section">
                    <h4>AI Description:</h4>
                    <p className="description">{result.llm_description}</p>
                  </div>
                )}

                {result.attention_stats && (
                  <div className="result-section">
                    <h4>Attention Statistics:</h4>
                    <div className="stats">
                      <div className="stat-item">
                        <span>Mean:</span>
                        <span>{result.attention_stats.mean.toFixed(4)}</span>
                      </div>
                      <div className="stat-item">
                        <span>Max:</span>
                        <span>{result.attention_stats.max.toFixed(4)}</span>
                      </div>
                      <div className="stat-item">
                        <span>Std:</span>
                        <span>{result.attention_stats.std.toFixed(4)}</span>
                      </div>
                    </div>
                  </div>
                )}

                <div className="result-meta">
                  <span>Model: {result.model_used}</span>
                  <span>Fusion: {result.fusion_enabled ? '✅' : '❌'}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default App;