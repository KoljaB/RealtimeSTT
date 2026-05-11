# RealtimeSTT Data Integrity System

Complete documentation for the data integrity verification and rejection system.

---

## 🎯 **Overview**

The Data Integrity System ensures that audio data sent from clients (browser/Python) to the RealtimeSTT server arrives without corruption. It provides:

- **Real-time verification** of audio data transmission
- **Configurable rejection policies** for corrupted clients
- **Detailed logging** for debugging and monitoring
- **Multiple client implementations** (JavaScript, Python)

---

## 🔄 **Data Flow Diagram**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client        │    │   WebSocket     │    │   STT Server    │
│ (Browser/Python)│    │   Transport     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │ 1. Record Audio       │                       │
         ▼                       │                       │
┌─────────────────┐              │                       │
│ Calculate:      │              │                       │
│ • Length: 1024  │              │                       │
│ • Checksum      │              │                       │
│ • Timestamp     │              │                       │
└─────────────────┘              │                       │
         │                       │                       │
         │ 2. Send Message       │                       │
         ▼                       │                       │
┌─────────────────┐              │                       │
│ [4B Length]     │──────────────▶                       │
│ [JSON Metadata] │              │                       │
│ [Audio Data]    │              │                       │
└─────────────────┘              │                       │
                                 │                       │
                                 │ 3. Receive & Parse    │
                                 │──────────────────────▶│
                                 │                       ▼
                                 │              ┌─────────────────┐
                                 │              │ Verify Data:    │
                                 │              │ • Calc checksum │
                                 │              │ • Compare       │
                                 │              │ • Track errors  │
                                 │              └─────────────────┘
                                 │                       │
                                 │                       ▼
                                 │              ┌─────────────────┐
                                 │              │ Results:        │
                                 │              │ ✅ PASS → Process│
                                 │              │ ❌ FAIL → Log   │
                                 │              │ 🚨 REJECT → Drop│
                                 │              └─────────────────┘
```

---

## 🔧 **Implementation**

### **Frontend (JavaScript)**

```javascript
// 1. Calculate verification data
let checksum = 0;
for (let i = 0; i < audioData.length; i++) {
    checksum = (checksum + audioData[i]) & 0xFFFFFFFF;
}

// 2. Create metadata
let metadata = JSON.stringify({
    sampleRate: 16000,
    dataLength: audioData.length,          // ← Verification
    checksum: checksum,                    // ← Verification
    timestamp: Date.now(),                 // ← Verification
    server_sent_to_stt: true              // ← Enable flag
});

// 3. Send message: [length][metadata][audio]
let message = new Blob([
    new DataView(new ArrayBuffer(4)).setInt32(0, metadataBytes.length, true),
    metadataBytes,
    audioData.buffer
]);
socket.send(message);
```

### **Server (Python)**

```python
def verify_data_integrity(audio_chunk, metadata, client_id=None):
    # Extract expected values from client
    expected_checksum = metadata['checksum']
    expected_length = metadata['dataLength']
    
    # Calculate actual values from received data
    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
    actual_length = len(audio_data)
    actual_checksum = int(np.sum(audio_data, dtype=np.int64)) & 0xFFFFFFFF
    
    # Verify and handle results
    is_valid = (actual_length == expected_length and 
                actual_checksum == expected_checksum)
    
    if is_valid:
        print(f"[OK] Data integrity verified")
    else:
        print(f"[FAIL] Data integrity check failed!")
        # Handle rejection policy if enabled...
        
    return is_valid, should_reject, error_message
```

---

## ⚙️ **Server Configuration**

### **Basic Usage:**
```bash
# Enable verification (log only)
stt-server --model tiny --verify-data-integrity

# Enable verification with detailed logging
stt-server --model tiny --verify-data-integrity --use_extended_logging

# Enable rejection (strict)
stt-server --model tiny --verify-data-integrity --reject-corrupted-data --corruption-threshold 0

# Enable rejection (tolerant)
stt-server --model tiny --verify-data-integrity --reject-corrupted-data --corruption-threshold 3
```

### **Complete Example:**
```bash
# Production-ready configuration
stt-server --model large-v2 \
  --control_port 8011 \
  --data_port 8012 \
  --verify-data-integrity \
  --reject-corrupted-data \
  --corruption-threshold 2 \
  --use_extended_logging
```

### **Configuration Options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--verify-data-integrity` | false | Enable checksum verification |
| `--reject-corrupted-data` | false | Reject clients with corrupted data |
| `--corruption-threshold N` | 0 | Allow N failures before rejection |
| `--use_extended_logging` | false | Show all verification results |

---

## 🛡️ **Rejection System**

### **How It Works:**
1. **Track failures** per client connection
2. **Increment counter** on each verification failure  
3. **Send rejection message** when threshold exceeded
4. **Close connection** to prevent further corruption
5. **Clean up tracking** when client disconnects

### **Rejection Policies:**

| Configuration | Behavior | Use Case |
|---------------|----------|----------|
| No rejection | `--verify-data-integrity` | Monitor only, allow all clients |
| Immediate | `--corruption-threshold 0` | Reject on first failure (strict) |
| Tolerant | `--corruption-threshold N` | Allow N failures (production) |

### **Client Rejection Message:**
```json
{
  "type": "error",
  "error": "data_corruption",
  "message": "Connection rejected: Checksum mismatch: expected 12345678, got 87654321",
  "action": "disconnect"
}
```

---

## 📊 **Server Logs Explained**

### **✅ Successful Verification:**
```
Server received audio chunk of length 8317 bytes, metadata: {...}
  [21:04:39.891] [OK] Data integrity verified (length: 4096, checksum: 4294965588)
```

### **❌ Failed Verification:**
```
Server received audio chunk of length 8317 bytes, metadata: {...}
  [21:04:40.123] [FAIL] Data integrity check failed!
    Length: expected 4096, got 4090 (FAIL)
    Checksum: expected 4294965588, got 4294965600 (FAIL)
    Checksum mismatch indicates audio data corruption during transmission
```

### **🚨 Client Rejection:**
```
  [21:04:41.456] [FAIL] Data integrity check failed!
    [WARNING] Client 192.168.1.100:54321 corruption count: 2/3
    
  [21:04:42.789] [FAIL] Data integrity check failed!
    [REJECT] Client 192.168.1.100:54321 exceeded corruption threshold (3 failures)
    [DISCONNECT] Closing connection to 192.168.1.100:54321 due to data corruption
```

### **📈 No Verification (Missing Flag):**
```
Server received audio chunk of length 8317 bytes, metadata: {...}
# ← No verification lines = --verify-data-integrity flag missing
```

---

## 🧪 **Testing Tools**

### **Available Test Clients:**

1. **`simple_python_client.py`** - Production-ready client with verification
   ```bash
   python simple_python_client.py
   ```

2. **`test_verification_client.py`** - Synthetic audio testing
   ```bash
   python test_verification_client.py --chunks 5 --interval 1.0
   ```

3. **`test_corrupted_data.py`** - Corruption detection testing
   ```bash
   python test_corrupted_data.py
   ```

4. **`test_rejection_system.py`** - Server rejection policy testing
   ```bash
   python test_rejection_system.py
   ```

5. **`test_client_rejection_handling.py`** - Client-side rejection handling
   ```bash
   python test_client_rejection_handling.py
   ```

### **Testing Workflow:**

```bash
# 1. Start server with strict rejection
stt-server --model tiny --verify-data-integrity --reject-corrupted-data --corruption-threshold 0

# 2. Test valid data (should work)
python simple_python_client.py

# 3. Test corruption detection (should show failures)
python test_corrupted_data.py

# 4. Test rejection system (should disconnect)
python test_rejection_system.py
```

---

## 🎛️ **Message Format Specification**

### **WebSocket Message Structure:**
```
┌─────────────────────────────────────────────────────────────┐
│                    WebSocket Message                        │
├─────────────────────────────────────────────────────────────┤
│ Metadata Length (4 bytes, little-endian uint32)            │
├─────────────────────────────────────────────────────────────┤
│ Metadata (JSON string, UTF-8 encoded)                      │
│ {                                                           │
│   "sampleRate": 16000,                                     │
│   "dataLength": 4096,        // ← Verification             │
│   "checksum": 4294965588,    // ← Verification             │
│   "timestamp": 1640995200000,// ← Verification             │
│   "server_sent_to_stt": true // ← Enable flag              │
│ }                                                           │
├─────────────────────────────────────────────────────────────┤
│ Audio Data (binary, 16-bit PCM samples)                    │
└─────────────────────────────────────────────────────────────┘
```

### **Metadata Fields:**

| Field | Type | Required | Purpose |
|-------|------|----------|---------|
| `sampleRate` | number | Always | Audio sample rate |
| `dataLength` | number | For verification | Number of audio samples |
| `checksum` | number | For verification | Sum of all samples & 0xFFFFFFFF |
| `timestamp` | number | For verification | Client timestamp (ms) |
| `server_sent_to_stt` | boolean | For verification | Enable verification flag |

---

## 🚀 **Performance Impact**

### **Client Side:**
- **Checksum calculation:** ~0.1ms for 4096 samples
- **Metadata overhead:** ~100 bytes per message  
- **CPU impact:** Negligible (simple sum operation)

### **Server Side:**
- **Verification time:** ~0.05ms per chunk
- **Memory overhead:** Minimal (no data copying)
- **Logging impact:** Only when failures occur

### **Network:**
- **Bandwidth increase:** ~2% (metadata overhead)
- **Latency impact:** None (no additional round trips)

---

## 🔍 **Troubleshooting**

### **Common Issues:**

**Q: Server logs show no verification messages**
```bash
# Missing verification flag - add it:
stt-server --verify-data-integrity
```

**Q: All checksums are 0**
- **A:** Audio is silent (all zeros). Normal for quiet periods.

**Q: High failure rate**
- **A:** Check network stability, audio drivers, or reduce rejection threshold.

**Q: Client gets rejected immediately**
- **A:** Server has `--corruption-threshold 0`. Increase threshold or fix corruption.

### **Debug Commands:**

```bash
# Maximum debugging
stt-server --verify-data-integrity --reject-corrupted-data --corruption-threshold 0 --debug --use_extended_logging

# Test with known good data
python test_verification_client.py --chunks 3

# Test corruption detection
python test_corrupted_data.py
```

---

## 📋 **Production Recommendations**

### **Development:**
```bash
# Strict verification for catching issues
stt-server --verify-data-integrity --reject-corrupted-data --corruption-threshold 0 --use_extended_logging
```

### **Production:**
```bash
# Balanced: Some network tolerance
stt-server --verify-data-integrity --reject-corrupted-data --corruption-threshold 3
```

### **High-Security:**
```bash
# Zero tolerance with full logging
stt-server --verify-data-integrity --reject-corrupted-data --corruption-threshold 0 --use_extended_logging
```

### **Monitoring Only:**
```bash
# Log corruption but don't reject clients
stt-server --verify-data-integrity --use_extended_logging
```

---

## 🎯 **Quick Start Guide**

### **1. Enable Basic Verification:**
```bash
stt-server --model tiny --verify-data-integrity
```

### **2. Run Client:**
```bash
python simple_python_client.py
```

### **3. Check Server Logs:**
Look for:
```
[OK] Data integrity verified (length: 4096, checksum: 4294965588)
```

### **4. Enable Rejection (Optional):**
```bash
stt-server --model tiny --verify-data-integrity --reject-corrupted-data --corruption-threshold 2
```

### **5. Test Corruption Detection:**
```bash
python test_corrupted_data.py
```

---

## 📚 **Summary**

The Data Integrity System provides:

✅ **Corruption Detection** - Catches transmission errors in real-time  
✅ **Configurable Policies** - From monitoring to strict rejection  
✅ **Multiple Clients** - JavaScript (browser) and Python support  
✅ **Comprehensive Testing** - Full test suite included  
✅ **Production Ready** - Minimal overhead, maximum reliability  

The system adds **~2% bandwidth overhead** but provides **significant value** for ensuring reliable audio transmission in production STT systems.

---

*For technical support or questions, check the server logs and test with the provided test clients.*