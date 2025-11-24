#### EEG-to-Hand-Rotation Prediction with Transformer Neural Networks


A cutting-edge Django web application that harnesses Transformer-based deep learning to predict robotic hand rotations from EEG (electroencephalography) brain signals. Features real-time 3D visualization, intelligent preprocessing, and seamless Blender integration for photorealistic rendering.

##  Project Overview

BrainHand demonstrates how artificial intelligence can bridge neuroscience and robotics by converting raw brain electrical activity into precise control commands for robotic systems. The system processes 62-channel EEG data through a multi-head attention Transformer architecture to predict:

- **Which hand component** should rotate (wrist_roll, wrist_pitch, elbow_yaw, elbow_pitch, shoulder_yaw)
- **By how many degrees** (0-180°) the rotation should occur

### Key Capabilities

-  **Real-time EEG Processing**: Handles streaming or batch EEG data in multiple formats
-  **Transformer Architecture**: State-of-the-art attention mechanism for sequence modeling
-  **Interactive 3D Preview**: WebGL-powered visualization with smooth real-time rotation
-  **Multi-Modal Input Support**: CSV, NumPy arrays, or raw JSON data formats
-  **Blender Integration**: Optional photorealistic hand rendering via subprocess automation
-  **Production-Ready**: Comprehensive error handling, input validation, and logging
-  **Extensible Design**: Easy integration with custom models and EEG datasets

##  Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Architecture](#architecture)





##  Quick Start

### Prerequisites
- Python 3.12+
- pip & virtualenv
- (Optional) Blender 4.2+ for rendering

### 5-Minute Setup

```bash
# 1. Clone and navigate
git clone https://github.com/your-username/brainhand.git
cd brainhand_full_fix

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Apply migrations
python manage.py migrate

# 5. Start server
python manage.py runserver

# 6. Open browser
# Visit http://localhost:8000/
```



##  Installation

### Full Installation Guide

#### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/brainhand.git
cd brainhand_full_fix
```

#### Step 2: Set Up Python Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

#### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Requirements include:**
- Django 4.2+ (Web framework)
- PyTorch 2.0+ (Deep learning)
- NumPy (Numerical computing)
- Pillow (Image processing)
- djangorestframework (REST API)

#### Step 4: Database Setup
```bash
python manage.py migrate
python manage.py createsuperuser  # Optional: for admin panel
```

#### Step 5: Run Development Server
```bash
python manage.py runserver 0.0.0.0:8000
```

Visit `http://localhost:8000/` in your browser.

#### Optional: Configure Blender
If you want Blender rendering support:
1. Install Blender 4.2+
2. Update `BLENDER_PATH` in `brainhand_project/settings.py`:
   ```python
   BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
   ```
3. Place your `.blend` file at `MEDIA_ROOT/blender/robotichand.blend`

##  Architecture

### System Design

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACE                        │
│  (Bootstrap 5 + Three.js 3D Visualization + Forms)      │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│              DJANGO REST API                             │
│  (Views, URL Routing, CSRF Protection, Error Handling)  │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│           EEG PREPROCESSING PIPELINE                     │
│  (Normalization, Padding, Transpose Handling)           │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│    TRANSFORMER NEURAL NETWORK (EEGTransformer)          │
│  (Embedding → Positional Encoding → Encoder Layers)    │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│         CLASSIFICATION & REGRESSION HEADS                │
│  (Hand Part Prediction + Degree Regression)             │
└──────────────────┬──────────────────────────────────────┘
                   │
          ┌────────┴─────────┐
          │                  │
    ┌─────▼────┐      ┌──────▼─────┐
    │ Mock Mode│      │Real Model  │
    │(Default) │      │(Optional)  │
    └──────────┘      └────────────┘
```

### Neural Network: EEGTransformer

#### Input Specification
| Parameter | Value |
|-----------|-------|
| EEG Channels | 62 |
| Sequence Length | 1,000 samples |
| Sample Rate | ~250 Hz (4 seconds) |
| Output Classes | 5 (hand parts) |
| Output Degrees | 0-180° (continuous) |

#### Model Architecture

```
Input: (batch_size, 1000, 62)
    ↓
[Embedding Layer]  → Projects channels to 128-dim space
    ↓
[Positional Encoding]  → Sinusoidal positional embeddings
    ↓
[Transformer Encoder]  → 4 layers, 8 attention heads
    ├─ Layer 1: Self-Attention (8 heads) → Feed-Forward
    ├─ Layer 2: Self-Attention (8 heads) → Feed-Forward
    ├─ Layer 3: Self-Attention (8 heads) → Feed-Forward
    └─ Layer 4: Self-Attention (8 heads) → Feed-Forward
    ↓
[Global Average Pooling]  → (batch_size, 128)
    ↓
    ├─→ [Classification Head] → (batch_size, 5)
    │    (ReLU + Dropout + Linear)
    │
    └─→ [Regression Head] → (batch_size, 1)
         (ReLU + Dropout + Linear)
    ↓
Output: Class logits + Degree prediction
```

#### Architecture Specifications
- **Embedding Dimension**: 128
- **Number of Attention Heads**: 8
- **Number of Encoder Layers**: 4
- **Feed-Forward Hidden Dim**: 512 (128 × 4)
- **Activation Function**: GELU
- **Dropout**: 0.1
- **Total Parameters**: ~850K

#### Why Transformer?

1. **Parallel Processing**: All time steps processed simultaneously (efficient)
2. **Long-Range Dependencies**: Self-attention captures patterns across entire sequence
3. **Multi-Scale Representation**: Multiple attention heads capture different frequency bands
4. **Scalability**: Easy to extend with more layers/heads for larger datasets
5. **Transfer Learning**: Pre-trained models available (DistilBERT, etc.)

### Frontend Architecture

#### Three.js 3D Scene
- **Camera**: 120° FOV, positioned at (0, 2.5, 8.0)
- **Lighting**: Directional light + ambient light with shadow mapping
- **Objects**: 
  - Ground plane (10×10 with shadow support)
  - Hand model (0.8×0.3×1.2 blue cube fallback, or custom GLB)
- **Rotation System**: Axis-mapped rotation with real-time updates
- **Responsive**: Auto-resizes on window resize events

#### User Interface
- **Framework**: Bootstrap 5
- **Components**: 
  - File upload form
  - JSON textarea input
  - Prediction result display
  - 3D canvas
  - Test rotation button
- **Responsive Design**: Works on mobile and desktop

### Backend Architecture

#### Django Structure
```
brainhand_project/
├── settings.py      # Configuration & model paths
├── urls.py          # Main URL dispatcher
└── wsgi.py          # Production server entry

brainhand/
├── model_loader.py  # EEGTransformer & mock models
├── views.py         # API endpoints (predict_ajax, render_robotic_hand)
├── urls.py          # App URL routing
├── forms.py         # EEG upload form validation
├── templates/
│   └── brainhand/
│       └── index.html  # Main UI template
└── static/
    └── brainhand/
        └── models/    # 3D model assets
```

##  Usage Guide

### 1. Running the Application

#### Development Mode
```bash
python manage.py runserver
# Runs on http://localhost:8000
```

#### Production Mode (with Gunicorn)
```bash
pip install gunicorn
gunicorn brainhand_project.wsgi:application --bind 0.0.0.0:8000
```

### 2. Using the Web Interface

#### Option A: Upload EEG Data

**CSV Format:**
- Rows = EEG channels (62 rows)
- Columns = Time samples (≥1000 columns)
- Values = Voltage readings (microvolts)

Example:
```csv
-5.2,3.1,2.8,...
-3.4,1.2,0.9,...
...
```

**NumPy Format (.npy):**
- Shape: (62, samples) or (samples, 62)
- Auto-transposed if needed
- Supports any sample length

**JSON Format:**
```json
[
  [ch1_sample1, ch1_sample2, ..., ch1_sample1000],
  [ch2_sample1, ch2_sample2, ..., ch2_sample1000],
  ...
  [ch62_sample1, ch62_sample2, ..., ch62_sample1000]
]
```

#### Option B: Test Rotation
Click "Test Rotation" button to:
1. Generate random hand part (0-4)
2. Generate random rotation (-45° to +45°)
3. Animate 3D hand rotation
4. Display test info in results panel

### 3. API Integration

#### Direct API Call
#### JavaScript Fetch

### 4. Programmatic Usage (Python)

##  Model Details
### Demo Mode: DeterministicMock

Used by default for demonstration without requiring training data.


### Production Mode: EEGTransformer


**Advantages:**
- Learns complex patterns from data
- Multi-scale feature extraction via attention
- End-to-end differentiable architecture
- Faster inference than traditional methods


