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

interface NavigationInstruction {
  direction: 'left' | 'right' | 'forward' | 'stop';
  priority: 'safe' | 'caution' | 'danger';
  message: string;
  reason: string;
}

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
  const [showDirections, setShowDirections] = useState<boolean>(true);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState<boolean>(true);
  const [autoProcess, setAutoProcess] = useState<boolean>(false);
  const [voiceEnabled, setVoiceEnabled] = useState<boolean>(false);
  const [annotatedImage, setAnnotatedImage] = useState<string | null>(null);
  const [navigationInstructions, setNavigationInstructions] = useState<NavigationInstruction | null>(null);

  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const processIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const speechSynthRef = useRef<SpeechSynthesisUtterance | null>(null);

  useEffect(() => {
    checkServerStatus();
    // Initialize speech synthesis
    if ('speechSynthesis' in window) {
      speechSynthRef.current = new SpeechSynthesisUtterance();
      speechSynthRef.current.rate = 0.9;
      speechSynthRef.current.pitch = 1.0;
      speechSynthRef.current.volume = 1.0;
    }
  }, []);

  // Auto-process functionality
  useEffect(() => {
    if (autoProcess && isCameraRunning) {
      processIntervalRef.current = setInterval(() => {
        processFrame();
      }, 3000);
    } else {
      if (processIntervalRef.current) {
        clearInterval(processIntervalRef.current);
        processIntervalRef.current = null;
      }
    }

    return () => {
      if (processIntervalRef.current) {
        clearInterval(processIntervalRef.current);
      }
    };
  }, [autoProcess, isCameraRunning]);

  const speak = (text: string) => {
    if (voiceEnabled && 'speechSynthesis' in window && speechSynthRef.current) {
      window.speechSynthesis.cancel(); // Stop any ongoing speech
      speechSynthRef.current.text = text;
      window.speechSynthesis.speak(speechSynthRef.current);
    }
  };

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
        speak('Camera started');
        setTimeout(() => startFrameStreaming(), 500);
      } else {
        setError(data.error || 'Failed to start camera');
        speak('Camera failed to start');
      }
    } catch (err) {
      setError('Error connecting to server');
      speak('Error connecting to server');
    }
  };

  const stopCamera = async () => {
    try {
      await fetch(`${API_BASE_URL}/camera/stop`, { method: 'POST' });
      setIsCameraRunning(false);
      setAutoProcess(false);
      stopFrameStreaming();
      if (imgRef.current) {
        imgRef.current.src = '';
      }
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
      }
      setFrameCount(0);
      setResult(null);
      setNavigationInstructions(null);
      setAnnotatedImage(null);
      speak('Camera stopped');
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
            imgRef.current.onload = () => {
              drawDirectionOverlay();
            };
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

  const getObjectsInZone = (zone: 'left' | 'front' | 'right') => {
    if (!result || !result.detected_objects) return [];

    const width = imgRef.current?.naturalWidth || 640;

    return result.detected_objects.filter(obj => {
      const [x1, , x2] = obj.bounding_box;
      const centerX = (x1 + x2) / 2;

      if (zone === 'right' && centerX < width / 3) return true;
      if (zone === 'left' && centerX > 2 * width / 3) return true;
      if (zone === 'front' && centerX >= width / 3 && centerX <= 2 * width / 3) return true;

      return false;
    });
  };

  const generateNavigationInstructions = (objects: DetectedObject[]): NavigationInstruction => {
    const leftObjects = getObjectsInZone('left');
    const frontObjects = getObjectsInZone('front');
    const rightObjects = getObjectsInZone('right');

    // Check for obstacles (person, chair, table, etc.)
    const dangerousObjects = ['person', 'car', 'truck', 'bicycle', 'motorcycle', 'chair', 'table', 'bench', 'couch', 'bed'];

    const leftDanger = leftObjects.filter(obj => dangerousObjects.includes(obj.label.toLowerCase()));
    const frontDanger = frontObjects.filter(obj => dangerousObjects.includes(obj.label.toLowerCase()));
    const rightDanger = rightObjects.filter(obj => dangerousObjects.includes(obj.label.toLowerCase()));

    // Determine best direction
    if (frontDanger.length > 0) {
      // Front is blocked
      if (leftDanger.length === 0) {
        return {
          direction: 'left',
          priority: 'caution',
          message: 'Path blocked ahead. Move left.',
          reason: `Obstacle detected: ${frontDanger.map(o => o.label).join(', ')}`
        };
      } else if (rightDanger.length === 0) {
        return {
          direction: 'right',
          priority: 'caution',
          message: 'Path blocked ahead. Move right.',
          reason: `Obstacle detected: ${frontDanger.map(o => o.label).join(', ')}`
        };
      } else {
        return {
          direction: 'stop',
          priority: 'danger',
          message: 'Stop! Obstacles in all directions.',
          reason: 'Multiple obstacles detected'
        };
      }
    } else if (leftObjects.length === 0 && rightObjects.length === 0) {
      // All clear
      return {
        direction: 'forward',
        priority: 'safe',
        message: 'Clear path ahead. Safe to proceed.',
        reason: 'No obstacles detected'
      };
    } else if (leftDanger.length > 0 && rightDanger.length === 0) {
      // Left has obstacles, prefer right
      return {
        direction: 'right',
        priority: 'caution',
        message: 'Obstacle on left. Stay right.',
        reason: `Left side: ${leftDanger.map(o => o.label).join(', ')}`
      };
    } else if (rightDanger.length > 0 && leftDanger.length === 0) {
      // Right has obstacles, prefer left
      return {
        direction: 'left',
        priority: 'caution',
        message: 'Obstacle on right. Stay left.',
        reason: `Right side: ${rightDanger.map(o => o.label).join(', ')}`
      };
    } else {
      // Objects present but not dangerous, proceed with caution
      return {
        direction: 'forward',
        priority: 'caution',
        message: 'Proceed with caution. Objects nearby.',
        reason: `Detected: ${objects.slice(0, 3).map(o => o.label).join(', ')}`
      };
    }
  };

  const drawDirectionOverlay = () => {
    if (!canvasRef.current || !imgRef.current || !showDirections) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = imgRef.current.width;
    canvas.height = imgRef.current.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!result || !result.detected_objects || !showBoundingBoxes) return;

    const width = canvas.width;
    const height = canvas.height;

    // Draw zone indicators
    ctx.globalAlpha = 0.2;

    const rightObjects = getObjectsInZone('right');
    if (rightObjects.length > 0) {
      ctx.fillStyle = '#4CAF50';
      ctx.fillRect(0, 0, width / 3, height);
    }

    const frontObjects = getObjectsInZone('front');
    if (frontObjects.length > 0) {
      ctx.fillStyle = '#2196F3';
      ctx.fillRect(width / 3, 0, width / 3, height);
    }

    const leftObjects = getObjectsInZone('left');
    if (leftObjects.length > 0) {
      ctx.fillStyle = '#FF9800';
      ctx.fillRect(2 * width / 3, 0, width / 3, height);
    }

    // Draw bounding boxes
    ctx.globalAlpha = 1.0;
    result.detected_objects.forEach(obj => {
      const [x1, y1, x2, y2] = obj.bounding_box;

      const scaleX = canvas.width / (imgRef.current?.naturalWidth || 640);
      const scaleY = canvas.height / (imgRef.current?.naturalHeight || 480);

      const sx1 = x1 * scaleX;
      const sy1 = y1 * scaleY;
      const sx2 = x2 * scaleX;
      const sy2 = y2 * scaleY;

      const centerX = (x1 + x2) / 2;
      const origWidth = imgRef.current?.naturalWidth || 640;
      let color = '#2196F3';
      if (centerX < origWidth / 3) {
        color = '#4CAF50';
      } else if (centerX > 2 * origWidth / 3) {
        color = '#FF9800';
      }

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);

      const label = `${obj.label} ${(obj.confidence * 100).toFixed(0)}%`;
      ctx.font = 'bold 16px Arial';
      const textMetrics = ctx.measureText(label);
      const textHeight = 24;

      ctx.fillStyle = color;
      ctx.fillRect(sx1, sy1 - textHeight, textMetrics.width + 10, textHeight);

      ctx.fillStyle = 'white';
      ctx.fillText(label, sx1 + 5, sy1 - 6);
    });

    // Draw zone labels
    ctx.font = 'bold 20px Arial';
    ctx.textAlign = 'center';

    if (rightObjects.length > 0) {
      ctx.fillStyle = 'rgba(76, 175, 80, 0.8)';
      ctx.fillRect(width / 6 - 60, 10, 120, 40);
      ctx.fillStyle = 'white';
      ctx.fillText('➡️ RIGHT', width / 6, 37);
    }

    if (frontObjects.length > 0) {
      ctx.fillStyle = 'rgba(33, 150, 243, 0.8)';
      ctx.fillRect(width / 2 - 60, 10, 120, 40);
      ctx.fillStyle = 'white';
      ctx.fillText('⬆️ FRONT', width / 2, 37);
    }

    if (leftObjects.length > 0) {
      ctx.fillStyle = 'rgba(255, 152, 0, 0.8)';
      ctx.fillRect(5 * width / 6 - 60, 10, 120, 40);
      ctx.fillStyle = 'white';
      ctx.fillText('⬅️ LEFT', 5 * width / 6, 37);
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
      const response = await fetch(
        `${API_BASE_URL}/process_camera/${selectedMode}?annotate=true`,
        { method: 'POST' }
      );

      if (response.ok) {
        const data = await response.json();
        console.log('=== Full API Response ===');
        console.log(JSON.stringify(data, null, 2));
        console.log('========================');

        // Extract the actual result data
        let resultData: ProcessingResult | null = null;

        // Try different possible response structures
        if (data.data) {
          resultData = data.data;
          console.log('Using data.data structure');
        } else if (data.caption && data.detected_objects) {
          resultData = data;
          console.log('Using flat data structure');
        } else {
          console.error('Unknown response structure');
          setError('Invalid API response structure');
          setIsProcessing(false);
          return;
        }

        setResult(resultData);
        setError('');
        console.log('Result set:', resultData);

        // Get annotated image - try multiple possible paths
        let annotatedImageData = null;

        if (data.annotated_image_base64) {
          annotatedImageData = data.annotated_image_base64;
          console.log('Found annotated_image_base64 at top level');
        } else if (data.data && data.data.annotated_image_base64) {
          annotatedImageData = data.data.annotated_image_base64;
          console.log('Found annotated_image_base64 in data.data');
        } else if (data.annotated_image) {
          annotatedImageData = data.annotated_image;
          console.log('Found annotated_image at top level');
        } else if (data.data && data.data.annotated_image) {
          annotatedImageData = data.data.annotated_image;
          console.log('Found annotated_image in data.data');
        }

        if (annotatedImageData) {
          // Trim any whitespace
          annotatedImageData = annotatedImageData.trim();

          // Check if it already has the data URL prefix
          const imgData = annotatedImageData.startsWith('data:image')
            ? annotatedImageData
            : `data:image/jpeg;base64,${annotatedImageData}`;

          setAnnotatedImage(imgData);
          console.log('✅ Annotated image set successfully');
          console.log('Image data length:', annotatedImageData.length);
        } else {
          console.warn('❌ No annotated image found in response');
          console.log('Available keys:', Object.keys(data));
          if (data.data) {
            console.log('Available keys in data.data:', Object.keys(data.data));
          }
          setAnnotatedImage(null);
        }

        // Generate navigation instructions
        const detectedObjects = resultData.detected_objects;
        if (detectedObjects && detectedObjects.length > 0) {
          const navInstructions = generateNavigationInstructions(detectedObjects);
          setNavigationInstructions(navInstructions);
          console.log('Navigation instructions:', navInstructions);

          // Speak navigation instructions
          speak(navInstructions.message);
        } else {
          console.log('No objects detected');
          setNavigationInstructions(null);
        }

        setTimeout(() => drawDirectionOverlay(), 100);
      } else {
        const errorData = await response.json();
        console.error('API error response:', errorData);
        setError(errorData.detail || 'Processing failed');
        setAnnotatedImage(null);
      }
    } catch (err) {
      console.error('Processing error:', err);
      setError(`Error processing frame: ${err}`);
      setAnnotatedImage(null);
    } finally {
      setIsProcessing(false);
    }
  };

  useEffect(() => {
    return () => {
      stopFrameStreaming();
      if (processIntervalRef.current) {
        clearInterval(processIntervalRef.current);
      }
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
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

  const parseGuidance = (guidance: string) => {
    const zones = {
      left: [] as string[],
      front: [] as string[],
      right: [] as string[]
    };

    const parts = guidance.split('To your ');
    parts.forEach(part => {
      if (part.includes('left:')) {
        const content = part.split('left:')[1].split('.')[0].trim();
        zones.left.push(content);
      } else if (part.includes('front:')) {
        const content = part.split('front:')[1].split('.')[0].trim();
        zones.front.push(content);
      } else if (part.includes('right:')) {
        const content = part.split('right:')[1].split('.')[0].trim();
        zones.right.push(content);
      }
    });

    return zones;
  };

  const getNavigationIcon = (direction: string) => {
    switch (direction) {
      case 'left': return '⬅️';
      case 'right': return '➡️';
      case 'forward': return '⬆️';
      case 'stop': return '🛑';
      default: return '❓';
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🎥 Vision Navigation System for Blind</h1>
        <p>Real-time AI guidance with directional navigation</p>
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
          <code>python main.py</code>
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

              <div className="toggle-group">
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={showDirections}
                    onChange={(e) => {
                      setShowDirections(e.target.checked);
                      if (e.target.checked) {
                        setTimeout(() => drawDirectionOverlay(), 100);
                      }
                    }}
                  />
                  <span>Show Direction Overlay</span>
                </label>

                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={showBoundingBoxes}
                    onChange={(e) => {
                      setShowBoundingBoxes(e.target.checked);
                      setTimeout(() => drawDirectionOverlay(), 100);
                    }}
                  />
                  <span>Show Bounding Boxes</span>
                </label>

                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={autoProcess}
                    onChange={(e) => setAutoProcess(e.target.checked)}
                    disabled={!isCameraRunning}
                  />
                  <span>Auto-Process (every 3s)</span>
                </label>

                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={voiceEnabled}
                    onChange={(e) => {
                      setVoiceEnabled(e.target.checked);
                      if (e.target.checked) {
                        speak('Voice guidance enabled');
                      }
                    }}
                  />
                  <span>🔊 Voice Guidance</span>
                </label>
              </div>
            </div>

            <div className="camera-stream-wrapper">
              <div className="camera-stream">
                {isCameraRunning ? (
                  <div className="video-container">
                    <img ref={imgRef} alt="Camera stream" className="camera-image" />
                    <canvas ref={canvasRef} className="overlay-canvas" />
                    <div className="frame-counter">Frames: {frameCount}</div>
                  </div>
                ) : (
                  <div className="placeholder">
                    <p>📷</p>
                    <p>Camera is off</p>
                    <p className="hint">Click "Start Camera" to begin</p>
                  </div>
                )}
              </div>

              {/* Direction Legend */}
              {isCameraRunning && showDirections && (
                <div className="direction-legend">
                  <div className="legend-item">
                    <div className="legend-color" style={{ backgroundColor: '#FF9800' }}></div>
                    <span>⬅️ LEFT</span>
                  </div>
                  <div className="legend-item">
                    <div className="legend-color" style={{ backgroundColor: '#2196F3' }}></div>
                    <span>⬆️ FRONT</span>
                  </div>
                  <div className="legend-item">
                    <div className="legend-color" style={{ backgroundColor: '#4CAF50' }}></div>
                    <span>➡️ RIGHT</span>
                  </div>
                </div>
              )}
            </div>

            {/* Navigation Instructions for Blind Person */}
            {navigationInstructions && (
              <div className={`navigation-panel priority-${navigationInstructions.priority}`}>
                <div className="navigation-header">
                  <span className="nav-icon">{getNavigationIcon(navigationInstructions.direction)}</span>
                  <h3>Navigation Instruction</h3>
                </div>
                <div className="navigation-content">
                  <div className="nav-message">{navigationInstructions.message}</div>
                  <div className="nav-reason">{navigationInstructions.reason}</div>
                  <div className="nav-action">
                    <strong>Action:</strong> {navigationInstructions.direction.toUpperCase()}
                  </div>
                </div>
              </div>
            )}

            {/* Annotated Image with Bounding Boxes */}
            {annotatedImage ? (
              <div className="annotated-image-section">
                <h3>📦 Processed Image with Bounding Boxes</h3>
                <img
                  src={annotatedImage}
                  alt="Annotated"
                  className="annotated-image"
                  onError={(e) => {
                    console.error('Failed to load annotated image');
                    setAnnotatedImage(null);
                    setError('Failed to load annotated image');
                  }}
                />
                <button
                  className="btn btn-primary"
                  onClick={() => {
                    const link = document.createElement('a');
                    link.href = annotatedImage;
                    link.download = `annotated_${Date.now()}.jpg`;
                    link.click();
                  }}
                >
                  💾 Download Image
                </button>
              </div>
            ) : result ? (
              <div className="annotated-image-section">
                <h3>📦 Processed Image with Bounding Boxes</h3>
                <div className="image-status">
                  <p>ℹ️ No annotated image available</p>
                  <p className="hint">Make sure your backend returns 'annotated_image_base64' in the response</p>
                  <details style={{ marginTop: '10px', cursor: 'pointer' }}>
                    <summary style={{ fontWeight: 'bold', color: '#667eea' }}>Show Debug Info</summary>
                    <pre style={{
                      background: '#f5f5f5',
                      padding: '10px',
                      borderRadius: '5px',
                      fontSize: '0.85em',
                      overflow: 'auto',
                      maxHeight: '200px'
                    }}>
                      {JSON.stringify(result, null, 2)}
                    </pre>
                  </details>
                </div>
              </div>
            ) : null}

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
              disabled={!isCameraRunning || isProcessing || autoProcess}
              className="btn btn-process"
            >
              {isProcessing ? '⏳ Processing...' : autoProcess ? '🔄 Auto-Processing' : '🚀 Process Frame'}
            </button>

            {/* Spatial Guidance Panel */}
            {result && result.guidance && (
              <div className="guidance-panel">
                <h3>🧭 Spatial Guidance</h3>
                <div className="guidance-zones">
                  {(() => {
                    const zones = parseGuidance(result.guidance);
                    return (
                      <>
                        <div className="zone-card left-zone">
                          <div className="zone-header">
                            <span className="zone-icon">⬅️</span>
                            <span className="zone-title">LEFT</span>
                          </div>
                          <div className="zone-content">
                            {zones.left.length > 0 ? (
                              zones.left.map((item, idx) => (
                                <div key={idx} className="zone-item">{item}</div>
                              ))
                            ) : (
                              <div className="zone-empty">Clear</div>
                            )}
                          </div>
                        </div>

                        <div className="zone-card front-zone">
                          <div className="zone-header">
                            <span className="zone-icon">⬆️</span>
                            <span className="zone-title">FRONT</span>
                          </div>
                          <div className="zone-content">
                            {zones.front.length > 0 ? (
                              zones.front.map((item, idx) => (
                                <div key={idx} className="zone-item">{item}</div>
                              ))
                            ) : (
                              <div className="zone-empty">Clear</div>
                            )}
                          </div>
                        </div>

                        <div className="zone-card right-zone">
                          <div className="zone-header">
                            <span className="zone-icon">➡️</span>
                            <span className="zone-title">RIGHT</span>
                          </div>
                          <div className="zone-content">
                            {zones.right.length > 0 ? (
                              zones.right.map((item, idx) => (
                                <div key={idx} className="zone-item">{item}</div>
                              ))
                            ) : (
                              <div className="zone-empty">Clear</div>
                            )}
                          </div>
                        </div>
                      </>
                    );
                  })()}
                </div>
                <p className="guidance-text">{result.guidance}</p>
              </div>
            )}

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