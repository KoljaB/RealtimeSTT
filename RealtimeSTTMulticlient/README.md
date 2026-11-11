# RealtimeSTT

A framework-agnostic JavaScript library for real-time speech-to-text transcription that can be imported into any frontend application.

## Features

- **Framework Agnostic**: Works with vanilla JS, React, Vue, Angular, or any other frontend framework
- **Real-time Transcription**: Live speech-to-text conversion with low latency
- **Event-driven Architecture**: Built-in EventEmitter for handling transcription events
- **Cross-browser Compatibility**: Supports all modern browsers with microphone access
- **Easy Integration**: Simple API that works out of the box
- **Configurable**: Customizable backend URLs and settings

## Installation

Simply include the library in your project:

```html
<script src="realtime-stt.js"></script>
```

Or import as a module:

```javascript
import RealtimeSTT from './realtime-stt.js';
```

## Quick Start

```javascript
// Initialize the library
const stt = new RealtimeSTT({
    serverUrl: 'ws://localhost:8080'
});

// Listen for transcription results
stt.on('transcript', (text) => {
    console.log('Transcribed:', text);
});

// Start recording
stt.startRecording();

// Stop recording
stt.stopRecording();
```

## Usage Examples

The library includes example applications:
- `test-app.html` - Basic test application
- `chatbot-demo.html` - Chatbot integration example

## Browser Support

- Chrome/Edge 60+
- Firefox 55+
- Safari 11+

Requires microphone permissions and HTTPS in production.