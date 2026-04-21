# Architecture Overview

## System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface (Streamlit)               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Upload X-ray → Predict → Visualize → Download Report│   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Processing Pipeline                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │Preprocess│→ │Model Inf.│→ │Saliency  │→ │Visualize  │  │
│  │          │  │(ResNet50)│  │(Grad-CAM)│  │(3D Heatmap)   │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Deep Learning Model                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │ ResNet-50 Encoder (Feature Extraction)             │    │
│  │ → Residual Blocks (50 layers, skip connections)   │    │
│  │ → Global Average Pooling (2048-dim features)      │    │
│  │ → Classification Head (Dense → Sigmoid)           │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Input Processing
1. **Upload**: User uploads chest X-ray (JPG/PNG)
2. **Validation**: Check file type, size, format
3. **Resize**: Scale to 256×256 pixels
4. **Normalize**: Convert to [0, 1] range

### Model Inference
1. **Feature Extraction**: ResNet-50 encoder processes image
2. **Prediction**: Classification head outputs pneumonia probability
3. **Attention**: Gradient-based saliency map generation
4. **Output**: Confidence score + visualization

### Visualization
1. **2D Saliency**: Heatmap overlay on original image
2. **3D Heatmap**: Interactive 3D surface visualization
3. **Risk Zones**: Highlight infection hotspots
4. **Report**: PDF with results and analysis

---

## Model Architecture

### ResNet-50 Base

```
Input (256, 256, 3)
    ↓
Conv(7×7, stride=2) + MaxPool
    ↓
Residual Block × 3 (layer 1)
    ↓
Residual Block × 4 (layer 2)
    ↓
Residual Block × 6 (layer 3)
    ↓
Residual Block × 3 (layer 4)
    ↓
Global Average Pooling → (2048,)
    ↓
Dense(256) → ReLU → Dropout(0.5)
    ↓
Dense(128) → ReLU → Dropout(0.3)
    ↓
Dense(1) → Sigmoid → [0, 1]
```

### Skip Connections (Residual Blocks)

```
Input
  ↓
┌─────────────────┐
│  Conv3×3        │
│  Batch Norm     │
│  ReLU           │
│  Conv3×3        │
│  Batch Norm     │
└────────┬────────┘
         │
         ↓
      Add ← Input (Skip Connection)
         │
         ↓
       ReLU
         │
         ↓
      Output
```

---

## Attention Mechanism

### Gradient-based Saliency Map

```python
# Process input through model
y = model(input_image)

# Compute gradients w.r.t. input
dy/dx = compute_gradients(y, input_image)

# Reduce over color channels to get saliency
saliency = max(|dy/dx|, axis=color_channels)

# Normalize to [0, 1]
saliency_normalized = (saliency - min) / (max - min)
```

---

## Performance Characteristics

### Computational Requirements

| Component | CPU Time | GPU Time |
|-----------|----------|----------|
| Preprocessing | 50ms | 50ms |
| Model Inference | 600ms | 150ms |
| Saliency Map | 1200ms | 300ms |
| 3D Visualization | 300ms | 300ms |
| **Total** | **2.15s** | **0.8s** |

### Memory Usage

| Component | Size |
|-----------|------|
| Model Weights | 450MB |
| Loaded Model | 1.2GB |
| Per-Image Processing | 500MB |
| **Total** | ~2.5GB |

---

## Scalability Design

### Horizontal Scaling
```
┌─────────────┐
│ Load Balancer│
└────┬────────┘
     │
  ┌──┴──┬──────┬──────┐
  │     │      │      │
┌─▼─┐ ┌─▼─┐ ┌─▼─┐ ┌─▼─┐
│App1│ │App2│ │App3│ │App4│
└───┘ └───┘ └───┘ └───┘
  │     │      │      │
  └──┬──┴──────┴──────┘
     │
┌────▼────────────┐
│ Shared Resources:│
│ - Model Weights │
│ - Cache         │
└─────────────────┘
```

### Caching Strategy
- **Session Cache**: Model loaded once per user session (Streamlit)
- **Memory Cache**: Frequently used data in memory
- **Result Cache**: Store recent predictions

---

## Security Architecture

### Data Flow
```
User Upload
    │
    ├─→ Validation
    ├─→ Encryption (optional)
    ├─→ Processing
    │
    ├─→ Model Inference (no storage)
    │
    └─→ Result Display
        └─→ Secure Download (signed URL)
```

### Access Control
- Authentication (future: OAuth2)
- Rate limiting
- API key management
- Audit logging

---

## Error Handling

### Graceful Degradation
```python
try:
    model = load_model()
    prediction = model.predict(processed_img)
except ModelNotFoundError:
    show_error("Model not available")
    suggest_action("Download model")
except MemoryError:
    show_error("Insufficient memory")
    suggest_action("Reduce image size")
except Exception as e:
    log_error(e)
    show_generic_error()
```

---

## Future Enhancements

1. **Ensemble Methods**: Multiple models for higher accuracy
2. **Federated Learning**: Train on decentralized data
3. **Model Optimization**: Quantization, pruning for faster inference
4. **Multi-disease Detection**: Expand to other pneumonia types
5. **Real-time Analysis**: Video frame processing
6. **Mobile Support**: TensorFlow Lite deployment

---

## References

- He, K., et al. (2015). Deep Residual Learning for Image Recognition
- Selvaraju, R. R., et al. (2016). Grad-CAM: Visual Explanations from Deep Networks
- NIH Chest X-ray Dataset Documentation
