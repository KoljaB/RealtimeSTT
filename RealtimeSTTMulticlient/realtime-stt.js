/**
 * RealtimeSTT - Framework-agnostic JavaScript library for real-time speech-to-text
 * 
 * Features:
 * - Framework agnostic (vanilla JS with backward compatibility)
 * - Configurable backend URLs and settings
 * - Event-driven architecture using EventEmitter
 * - Built-in microphone icon component
 * - Automatic sentence boundary detection
 * - Cross-browser compatibility
 * 
 * @version 1.1.0
 */

(function(global, factory) {
    // UMD pattern for maximum compatibility
    if (typeof exports === 'object' && typeof module !== 'undefined') {
        // CommonJS
        module.exports = factory();
    } else if (typeof define === 'function' && define.amd) {
        // AMD
        define(factory);
    } else {
        // Browser globals
        global.RealtimeSTT = factory();
    }
})(typeof globalThis !== 'undefined' ? globalThis : typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : typeof self !== 'undefined' ? self : this, function() {
    'use strict';

    // Simple EventEmitter implementation for maximum compatibility
    function EventEmitter() {
        this.events = {};
    }

    EventEmitter.prototype.on = function(event, callback) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(callback);
        return this;
    };

    EventEmitter.prototype.off = function(event, callback) {
        if (!this.events[event]) return this;
        
        if (!callback) {
            delete this.events[event];
            return this;
        }

        this.events[event] = this.events[event].filter(function(cb) {
            return cb !== callback;
        });
        return this;
    };

    EventEmitter.prototype.emit = function(event) {
        if (!this.events[event]) return this;
        
        var args = Array.prototype.slice.call(arguments, 1);
        this.events[event].forEach(function(callback) {
            try {
                callback.apply(null, args);
            } catch (error) {
                console.error('EventEmitter error:', error);
            }
        });
        return this;
    };

    EventEmitter.prototype.once = function(event, callback) {
        var self = this;
        function onceWrapper() {
            callback.apply(null, arguments);
            self.off(event, onceWrapper);
        }
        return this.on(event, onceWrapper);
    };

    /**
     * Main RealtimeSTT class
     */
    function RealtimeSTT(config) {
        EventEmitter.call(this);
        
        // Default configuration
        this.config = this._mergeConfig({
            controlUrl: 'ws://172.31.10.139:8011',
            dataUrl: 'ws://172.31.10.139:8012',
            language: 'en',
            autoConnect: true,
            reconnectAttempts: 3,
            reconnectDelay: 1000,
            micIconSize: 32,
            micIconColor: '#007bff',
            micIconActiveColor: '#dc3545',
            debug: false
        }, config || {});

        // Internal state
        this.state = {
            isConnected: false,
            isRecording: false,
            isListening: false,
            currentText: '',
            fullSentences: [],
            reconnectCount: 0,
            sessionId: null
        };

        // WebSocket connections
        this.controlSocket = null;
        this.dataSocket = null;
        
        // Audio context and processing
        this.audioContext = null;
        this.mediaStream = null;
        this.mediaProcessor = null;
        
        // DOM elements
        this.micIcon = null;
        
        this._log('RealtimeSTT initialized with config:', this.config);
        
        if (this.config.autoConnect) {
            this._initializeConnections();
        }
    }

    // Inherit from EventEmitter
    RealtimeSTT.prototype = Object.create(EventEmitter.prototype);
    RealtimeSTT.prototype.constructor = RealtimeSTT;

    /**
     * Public API Methods
     */
    
    RealtimeSTT.prototype.connect = function() {
        this._log('Connecting to STT server...');
        this._initializeConnections();
        return this;
    };

    RealtimeSTT.prototype.disconnect = function() {
        this._log('Disconnecting from STT server...');
        this._manualDisconnect = true;  // Set flag to prevent reconnection
        this._cleanup();
        this.emit('disconnected');
        
        // Clear manual disconnect flag after cleanup
        var self = this;
        setTimeout(function() {
            self._manualDisconnect = false;
        }, 200);
        
        return this;
    };

    RealtimeSTT.prototype.startListening = function() {
        var self = this;
        
        if (this.state.isListening) {
            this._log('Already listening');
            return Promise.resolve();
        }

        this._log('Starting to listen...');
        
        return this._initializeAudio()
            .then(function() {
                self.state.isListening = true;
                self.state.isRecording = true;
                self._updateMicIcon();
                self.emit('listening-started');
                self._log('Listening started successfully');
            })
            .catch(function(error) {
                self._log('Error starting to listen:', error);
                self.emit('error', { type: 'microphone', message: error.message });
                throw error;
            });
    };

    RealtimeSTT.prototype.stopListening = function() {
        if (!this.state.isListening) {
            this._log('Not currently listening');
            return this;
        }

        this._log('Stopping listening...');
        this._stopAudioCapture();
        this.state.isListening = false;
        this.state.isRecording = false;
        this._updateMicIcon();
        this.emit('listening-stopped');
        return this;
    };

    RealtimeSTT.prototype.toggleListening = function() {
        if (this.state.isListening) {
            return this.stopListening();
        } else {
            return this.startListening();
        }
    };

    RealtimeSTT.prototype.createMicIcon = function(container) {
        if (typeof container === 'string') {
            container = document.getElementById(container) || document.querySelector(container);
        }
        
        if (!container) {
            throw new Error('Container not found for mic icon');
        }

        this.micIcon = this._createMicIconElement();
        container.appendChild(this.micIcon);
        
        return this.micIcon;
    };

    RealtimeSTT.prototype.setConfig = function(newConfig) {
        this.config = this._mergeConfig(this.config, newConfig);
        this.emit('config-updated', this.config);
        return this;
    };

    RealtimeSTT.prototype.getState = function() {
        return Object.assign({}, this.state);
    };

    RealtimeSTT.prototype.sendControlCommand = function(command) {
        if (!this.controlSocket || this.controlSocket.readyState !== WebSocket.OPEN) {
            this._log('Control socket not ready');
            return Promise.reject(new Error('Control socket not ready'));
        }
        
        // Add session ID to command if available
        if (this.state.sessionId) {
            command.session_id = this.state.sessionId;
        }
        
        var message = JSON.stringify(command);
        this.controlSocket.send(message);
        
        return Promise.resolve();
    };
    
    RealtimeSTT.prototype.getSessionId = function() {
        return this.state.sessionId;
    };

    /**
     * Private Methods
     */
    
    RealtimeSTT.prototype._mergeConfig = function(target, source) {
        var result = {};
        for (var key in target) {
            result[key] = target[key];
        }
        for (var key in source) {
            result[key] = source[key];
        }
        return result;
    };

    RealtimeSTT.prototype._log = function() {
        if (this.config.debug) {
            var args = Array.prototype.slice.call(arguments);
            args.unshift('[RealtimeSTT]');
            console.log.apply(console, args);
        }
    };

    RealtimeSTT.prototype._initializeConnections = function() {
        var self = this;
        
        // Prevent connection attempts during cleanup
        if (this._isCleaningUp) {
            this._log('Skipping connection - cleanup in progress');
            return;
        }
        
        // First cleanup any existing connections to prevent overlap
        this._cleanup();
        
        // Connect to control WebSocket
        try {
            this.controlSocket = new WebSocket(this.config.controlUrl);
            
            this.controlSocket.onopen = function() {
                self._log('Control socket connected');
                self._checkConnectionStatus();
            };
            
            this.controlSocket.onmessage = function(event) {
                try {
                    var message = JSON.parse(event.data);
                    if (message.type === 'session_init' && message.session_id) {
                        self.state.sessionId = message.session_id;
                        self._log('Session ID received:', self.state.sessionId);
                        self.emit('session-initialized', message.session_id);
                    } else if (message.status === 'error' && message.message && message.message.includes('maximum capacity')) {
                        // Handle server capacity limit
                        self._log('Server at capacity:', message.message);
                        self.emit('error', { type: 'capacity-limit', message: message.message });
                        self.controlSocket.close();
                    } else {
                        // Handle other control messages
                        self._log('Control message received:', message);
                        self.emit('control-message', message);
                    }
                } catch (error) {
                    self._log('Error parsing control message:', error);
                }
            };
            
            this.controlSocket.onclose = function(event) {
                self._log('Control socket closed:', event.code, event.reason);
                self.state.isConnected = false;
                self._handleReconnection();
            };
            
            this.controlSocket.onerror = function(error) {
                self._log('Control socket error:', error);
                self.emit('error', { type: 'control-connection', message: 'Control WebSocket connection failed' });
            };
            
        } catch (error) {
            this._log('Error creating control socket:', error);
            this.emit('error', { type: 'control-connection', message: error.message });
        }

        // Connect to data WebSocket
        try {
            this.dataSocket = new WebSocket(this.config.dataUrl);
            
            this.dataSocket.onopen = function() {
                self._log('Data socket connected');
                // Send session ID to data socket after connection
                self._sendSessionIdToDataSocket();
                self._checkConnectionStatus();
            };
            
            this.dataSocket.onmessage = function(event) {
                self._handleDataMessage(event.data);
            };
            
            this.dataSocket.onclose = function(event) {
                self._log('Data socket closed:', event.code, event.reason);
                self.state.isConnected = false;
                self._handleReconnection();
            };
            
            this.dataSocket.onerror = function(error) {
                self._log('Data socket error:', error);
                self.emit('error', { type: 'data-connection', message: 'Data WebSocket connection failed' });
            };
            
        } catch (error) {
            this._log('Error creating data socket:', error);
            this.emit('error', { type: 'data-connection', message: error.message });
        }
    };

    RealtimeSTT.prototype._sendSessionIdToDataSocket = function() {
        var self = this;
        // Wait a bit for session ID to be received from control socket
        var waitForSessionId = function(attempts) {
            if (attempts <= 0) {
                self._log('ERROR: Failed to get session ID after 5 seconds, data socket may not work properly');
                self.emit('error', { type: 'session-timeout', message: 'Session ID not received in time' });
                return;
            }
            
            if (self.state.sessionId) {
                try {
                    var sessionMessage = JSON.stringify({
                        session_id: self.state.sessionId
                    });
                    self._log('Sending session ID to data socket:', self.state.sessionId);
                    self.dataSocket.send(sessionMessage);
                    self._log('Session ID sent successfully to data socket');
                } catch (error) {
                    self._log('ERROR: Failed to send session ID to data socket:', error);
                    self.emit('error', { type: 'session-send-error', message: error.message });
                }
            } else {
                self._log('Waiting for session ID... (attempts remaining:', attempts + ')');
                setTimeout(function() {
                    waitForSessionId(attempts - 1);
                }, 100);
            }
        };
        
        waitForSessionId(50); // Try for up to 5 seconds
    };

    RealtimeSTT.prototype._checkConnectionStatus = function() {
        var controlReady = this.controlSocket && this.controlSocket.readyState === WebSocket.OPEN;
        var dataReady = this.dataSocket && this.dataSocket.readyState === WebSocket.OPEN;
        
        if (controlReady && dataReady && !this.state.isConnected) {
            this.state.isConnected = true;
            this.state.reconnectCount = 0;
            this._log('Both sockets connected, STT ready');
            this.emit('connected');
        }
    };

    RealtimeSTT.prototype._handleReconnection = function() {
        var self = this;
        
        // Don't attempt reconnection if we're cleaning up or if manual disconnect
        if (this._isCleaningUp || this._manualDisconnect) {
            this._log('Skipping reconnection - cleanup in progress or manual disconnect');
            return;
        }
        
        if (this.state.reconnectCount >= this.config.reconnectAttempts) {
            this._log('Maximum reconnection attempts reached');
            this.emit('connection-failed');
            return;
        }

        this.state.reconnectCount++;
        this._log('Attempting reconnection', this.state.reconnectCount, '/', this.config.reconnectAttempts);
        
        // Store timer reference so it can be cancelled
        this._reconnectTimer = setTimeout(function() {
            self._reconnectTimer = null;
            self._initializeConnections();
        }, this.config.reconnectDelay * this.state.reconnectCount);
    };

    RealtimeSTT.prototype._handleDataMessage = function(data) {
        try {
            var message = JSON.parse(data);
            this._log('Received message:', message);
            
            switch (message.type) {
                case 'realtime':
                    this.state.currentText = message.text;
                    this.emit('realtime-text', message.text);
                    break;
                    
                case 'fullSentence':
                    this.state.fullSentences.push(message.text);
                    this.state.currentText = '';
                    this.emit('sentence-end', message.text);
                    // Note: full-sentence event removed to prevent duplicate processing
                    // Applications should use 'sentence-end' for final transcriptions
                    break;
                    
                case 'recording_start':
                    this.state.isRecording = true;
                    this.emit('recording-started');
                    break;
                    
                case 'recording_stop':
                    this.state.isRecording = false;
                    this.emit('recording-stopped');
                    break;
                    
                case 'vad_detect_start':
                    this.emit('voice-detected');
                    break;
                    
                case 'vad_detect_stop':
                    this.emit('voice-stopped');
                    break;
                    
                case 'wakeword_detected':
                    this.emit('wakeword-detected');
                    break;
                    
                case 'transcription_start':
                    this.emit('transcription-started', message.audio_bytes_base64);
                    break;
                    
                case 'start_turn_detection':
                    this.emit('turn-detection-started');
                    break;
                    
                case 'stop_turn_detection':
                    this.emit('turn-detection-stopped');
                    break;
                    
                case 'wakeword_detection_start':
                    this.emit('wakeword-detection-started');
                    break;
                    
                case 'wakeword_detection_end':
                    this.emit('wakeword-detection-ended');
                    break;
                    
                default:
                    this._log('Unknown message type:', message.type);
            }
            
        } catch (error) {
            this._log('Error parsing message:', error);
            this.emit('error', { type: 'message-parse', message: error.message });
        }
    };

    RealtimeSTT.prototype._initializeAudio = function() {
        var self = this;
        
        return new Promise(function(resolve, reject) {
            // Check for browser support
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                reject(new Error('Browser does not support audio capture'));
                return;
            }

            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(function(stream) {
                    self.mediaStream = stream;
                    
                    // Create audio context (with fallback for older browsers)
                    var AudioContextClass = window.AudioContext || window['webkitAudioContext'];
                    self.audioContext = new AudioContextClass();
                    
                    var input = self.audioContext.createMediaStreamSource(stream);
                    
                    // Use ScriptProcessorNode for backward compatibility
                    self.mediaProcessor = self.audioContext.createScriptProcessor(1024, 1, 1);
                    
                    self.mediaProcessor.onaudioprocess = function(event) {
                        if (self.state.isListening) {
                            var audioData = event.inputBuffer.getChannelData(0);
                            self._sendAudioChunk(audioData, self.audioContext.sampleRate);
                        }
                    };
                    
                    input.connect(self.mediaProcessor);
                    self.mediaProcessor.connect(self.audioContext.destination);
                    
                    resolve();
                })
                .catch(function(error) {
                    self._log('Error accessing microphone:', error);
                    reject(error);
                });
        });
    };

    RealtimeSTT.prototype._sendAudioChunk = function(audioData, sampleRate) {
        if (!this.dataSocket || this.dataSocket.readyState !== WebSocket.OPEN) {
            return;
        }

        // Convert float32 to int16 PCM
        var float32Array = new Float32Array(audioData);
        var pcm16Data = new Int16Array(float32Array.length);

        for (var i = 0; i < float32Array.length; i++) {
            pcm16Data[i] = Math.max(-1, Math.min(1, float32Array[i])) * 0x7FFF;
        }

        // Create metadata
        var metadata = JSON.stringify({ sampleRate: sampleRate });
        var metadataLength = new Uint32Array([metadata.length]);
        var metadataBuffer = new TextEncoder ? 
            new TextEncoder().encode(metadata) : 
            this._stringToUint8Array(metadata);

        // Combine all data
        var message = new Uint8Array(
            metadataLength.byteLength + metadataBuffer.byteLength + pcm16Data.byteLength
        );
        
        message.set(new Uint8Array(metadataLength.buffer), 0);
        message.set(metadataBuffer, metadataLength.byteLength);
        message.set(new Uint8Array(pcm16Data.buffer), metadataLength.byteLength + metadataBuffer.byteLength);

        this.dataSocket.send(message);
    };

    RealtimeSTT.prototype._stringToUint8Array = function(str) {
        var arr = new Uint8Array(str.length);
        for (var i = 0; i < str.length; i++) {
            arr[i] = str.charCodeAt(i);
        }
        return arr;
    };

    RealtimeSTT.prototype._stopAudioCapture = function() {
        if (this.mediaProcessor) {
            this.mediaProcessor.disconnect();
            this.mediaProcessor = null;
        }

        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(function(track) {
                track.stop();
            });
            this.mediaStream = null;
        }
    };

    RealtimeSTT.prototype._createMicIconElement = function() {
        var self = this;
        var button = document.createElement('button');
        
        button.innerHTML = this._getMicIconSVG();
        button.style.cssText = [
            'border: none',
            'background: transparent',
            'cursor: pointer',
            'padding: 8px',
            'border-radius: 50%',
            'transition: all 0.3s ease',
            'display: inline-flex',
            'align-items: center',
            'justify-content: center',
            'width: ' + (this.config.micIconSize + 16) + 'px',
            'height: ' + (this.config.micIconSize + 16) + 'px'
        ].join(';');
        
        button.onclick = function() {
            self.toggleListening();
        };
        
        // Accessibility
        button.setAttribute('aria-label', 'Toggle speech recognition');
        button.setAttribute('role', 'button');
        
        this._updateMicIconStyle(button);
        
        return button;
    };

    RealtimeSTT.prototype._getMicIconSVG = function() {
        var size = this.config.micIconSize;
        var color = this.state.isListening ? this.config.micIconActiveColor : this.config.micIconColor;
        
        return '<svg width="' + size + '" height="' + size + '" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">' +
               '<path d="M12 1C10.34 1 9 2.34 9 4V12C9 13.66 10.34 15 12 15C13.66 15 15 13.66 15 12V4C15 2.34 13.66 1 12 1Z" fill="' + color + '"/>' +
               '<path d="M19 10V12C19 15.866 15.866 19 12 19C8.134 19 5 15.866 5 12V10H7V12C7 14.761 9.239 17 12 17C14.761 17 17 14.761 17 12V10H19Z" fill="' + color + '"/>' +
               '<path d="M11 21H13V23H11V21Z" fill="' + color + '"/>' +
               '<path d="M8 23H16V24H8V23Z" fill="' + color + '"/>' +
               '</svg>';
    };

    RealtimeSTT.prototype._updateMicIcon = function() {
        if (this.micIcon) {
            this.micIcon.innerHTML = this._getMicIconSVG();
            this._updateMicIconStyle(this.micIcon);
        }
    };

    RealtimeSTT.prototype._updateMicIconStyle = function(button) {
        if (this.state.isListening) {
            button.style.backgroundColor = 'rgba(220, 53, 69, 0.1)';
            button.style.boxShadow = '0 0 0 3px rgba(220, 53, 69, 0.2)';
        } else {
            button.style.backgroundColor = 'rgba(0, 123, 255, 0.1)';
            button.style.boxShadow = '0 0 0 3px rgba(0, 123, 255, 0.2)';
        }
    };

    RealtimeSTT.prototype._cleanup = function() {
        // Set cleanup flag to prevent race conditions
        this._isCleaningUp = true;
        
        this._stopAudioCapture();
        
        // Clear any pending reconnection timers
        if (this._reconnectTimer) {
            clearTimeout(this._reconnectTimer);
            this._reconnectTimer = null;
        }
        
        // Close WebSocket connections with proper event removal
        if (this.controlSocket) {
            // Remove event listeners to prevent onclose from triggering reconnection
            this.controlSocket.onopen = null;
            this.controlSocket.onmessage = null;
            this.controlSocket.onclose = null;
            this.controlSocket.onerror = null;
            
            if (this.controlSocket.readyState === WebSocket.OPEN || 
                this.controlSocket.readyState === WebSocket.CONNECTING) {
                this.controlSocket.close();
            }
            this.controlSocket = null;
        }
        
        if (this.dataSocket) {
            // Remove event listeners to prevent onclose from triggering reconnection
            this.dataSocket.onopen = null;
            this.dataSocket.onmessage = null;
            this.dataSocket.onclose = null;
            this.dataSocket.onerror = null;
            
            if (this.dataSocket.readyState === WebSocket.OPEN || 
                this.dataSocket.readyState === WebSocket.CONNECTING) {
                this.dataSocket.close();
            }
            this.dataSocket = null;
        }
        
        // Complete state reset to prevent multiple client instances
        this.state.isConnected = false;
        this.state.isListening = false;
        this.state.isRecording = false;
        this.state.currentText = '';
        this.state.fullSentences = [];
        this.state.reconnectCount = 0;  // Reset reconnection attempts
        this.state.sessionId = null;    // Clear session ID to prevent reuse
        
        this._updateMicIcon();
        
        // Clear cleanup flag after a brief delay to allow for WebSocket close events
        var self = this;
        setTimeout(function() {
            self._isCleaningUp = false;
        }, 100);
    };

    // Static method to check browser support
    RealtimeSTT.isSupported = function() {
        return !!(navigator.mediaDevices && 
                 navigator.mediaDevices.getUserMedia && 
                 window.WebSocket &&
                 (window.AudioContext || window['webkitAudioContext']));
    };

    return RealtimeSTT;
});