# API Reference

## Core Functions

### `load_pneumonia_model(model_path: Path) -> Optional[Model]`

Load and cache the pneumonia detection model.

**Parameters:**
- `model_path` (Path): Path to the saved Keras model file (.h5 format)

**Returns:**
- Model: Loaded Keras model or None if loading fails

**Example:**
```python
from app.helpers import load_pneumonia_model
from pathlib import Path

model_path = Path("app/model/resnet_model.h5")
model = load_pneumonia_model(model_path)

if model:
    prediction = model.predict(processed_image)
```

---

### `preprocess_image(img: Image.Image, target_size: Tuple[int, int]) -> np.ndarray`

Preprocess image for model inference.

**Parameters:**
- `img` (Image.Image): PIL Image object to preprocess
- `target_size` (Tuple[int, int]): Target image dimensions. Default: (256, 256)

**Returns:**
- np.ndarray: Array of shape (1, height, width, 3) with normalized values [0, 1]

**Example:**
```python
from PIL import Image
from app.helpers import preprocess_image

img = Image.open("chest_xray.jpg")
processed = preprocess_image(img, target_size=(256, 256))
```

---

### `create_saliency_map(model: Model, img_array: np.ndarray) -> np.ndarray`

Generate a saliency map showing regions of interest.

**Parameters:**
- `model` (Model): Trained Keras model
- `img_array` (np.ndarray): Input image array of shape (1, height, width, 3)

**Returns:**
- np.ndarray: Saliency map of shape (height, width) with values in [0, 1]

**Example:**
```python
from app.helpers import create_saliency_map

saliency = create_saliency_map(model, processed_image)
```

---

### `create_3d_heatmap(img_array, saliency_map, is_pneumonia, height_scale=35.0, downsample_factor=2)`

Create an interactive 3D heatmap visualization.

**Parameters:**
- `img_array` (np.ndarray): Input image
- `saliency_map` (np.ndarray): Saliency map from attention mechanism
- `is_pneumonia` (bool): Whether image shows pneumonia
- `height_scale` (float): Height scaling factor for 3D effect
- `downsample_factor` (int): Downsampling factor for performance

**Returns:**
- plotly.graph_objects.Figure: Interactive 3D visualization

---

## Configuration

Configuration is centralized in `app/config.py`.

### Key Constants

- `MODEL_PATH`: Path to the trained model
- `INPUT_SIZE`: Model input dimensions (256, 256)
- `CLASS_LABELS`: Mapping of class indices to labels
- `MODEL_METRICS`: Baseline performance metrics
- `COLORS`: Application color scheme

---

## Error Handling

All functions include proper error handling:

```python
from app.helpers import load_pneumonia_model

try:
    model = load_pneumonia_model(model_path)
    if model is None:
        print("Model loading failed")
except FileNotFoundError:
    print("Model file not found")
except Exception as e:
    print(f"Error: {str(e)}")
```

---

## Performance Benchmarks

- **Single Prediction:** ~800ms (CPU), ~200ms (GPU)
- **Saliency Map:** ~1.2s (CPU), ~300ms (GPU)
- **3D Visualization:** Instant (rendering)

---

## Examples

### Complete Prediction Pipeline

```python
from pathlib import Path
from PIL import Image
from app.helpers import (
    load_pneumonia_model,
    preprocess_image,
    create_saliency_map,
    create_3d_heatmap,
)
from app.config import MODEL_PATH

# Load model
model = load_pneumonia_model(MODEL_PATH)

# Load and preprocess image
img = Image.open("chest_xray.jpg")
processed = preprocess_image(img)

# Get prediction
prediction = model.predict(processed)
confidence = float(prediction[0][0])
is_pneumonia = confidence > 0.5

# Generate visualizations
saliency = create_saliency_map(model, processed)
fig_3d = create_3d_heatmap(processed, saliency, is_pneumonia)

# Display results
print(f"Pneumonia Risk: {confidence*100:.1f}%")
fig_3d.show()
```
