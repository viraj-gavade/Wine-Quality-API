# Wine Quality API 🍷

A modular FastAPI backend that serves PyTorch neural network models for wine analysis. Train, evaluate, and deploy models for both **classification** (wine type: red/white) and **regression** (quality score prediction) tasks through clean REST API endpoints.

> **Note:** This is a learning project focused on MLOps-style architecture, modular design, and deployment patterns rather than achieving state-of-the-art model accuracy.

---

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Usage Examples](#-usage-examples)
- [Configuration](#-configuration)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- **🎯 Dual Task Support**: Train models for classification (wine type) or regression (quality prediction)
- **🔧 Configurable Training**: Dynamic hyperparameter configuration via API requests
- **💾 Model Persistence**: Automatic model saving/loading with organized file management
- **📊 Evaluation Pipeline**: Comprehensive testing endpoints with metrics reporting
- **🚀 Real-time Inference**: Fast prediction API for production use
- **📝 Structured Logging**: Detailed logs for debugging and monitoring
- **✅ Request Validation**: Pydantic schemas ensure data integrity
- **📚 Auto Documentation**: Interactive API docs via Swagger UI

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | FastAPI |
| **ML Library** | PyTorch |
| **Validation** | Pydantic |
| **Server** | Uvicorn |
| **Language** | Python 3.x |

---

## 📁 Project Structure

```
wine-quality-api/
├── app.py                      # FastAPI application entrypoint
├── requirements.txt            # Python dependencies
│
├── controllers/                # Business logic layer
│   └── model_controllers.py    # Train/test/predict implementations
│
├── routes/                     # API route definitions
│   └── model_routes.py         # Model endpoint routes
│
├── Schemas/                    # Request/response models
│   ├── train_schema.py         # Training request schema
│   ├── test_schema.py          # Testing request schema
│   └── predict_schema.py       # Prediction request schema
│
├── src/                        # Core ML components
│   ├── models.py               # Neural network architectures
│   ├── training_pipeline.py    # Training orchestration
│   ├── testing_pipeline.py     # Evaluation logic
│   └── processing.py           # Data preprocessing
│
├── utils/                      # Utility functions
│   ├── data_loader.py          # Dataset loading
│   ├── preprocessing.py        # Feature engineering
│   ├── save_load.py            # Model serialization
│   ├── params.py               # Parameter definitions
│   └── logger.py               # Logging configuration
│
├── Models/                     # Model artifacts
│   └── trained/                # Saved .pth model files
│
├── data/                       # Datasets
│   └── processed_winequality.csv
│
├── notebooks/                  # Jupyter experiments
│   └── exploration.ipynb
│
└── logs/                       # Application logs
    └── training.log
```

### Component Overview

| Directory | Purpose |
|-----------|---------|
| `controllers/` | Glue layer between routes and ML pipelines; handles model selection and response formatting |
| `routes/` | FastAPI router registration and endpoint definitions |
| `Schemas/` | Pydantic models for request validation and response serialization |
| `src/` | Core ML functionality: model architecture, training loops, evaluation |
| `utils/` | Shared utilities: data loading, preprocessing, I/O operations, configuration |
| `Models/trained/` | Persistent storage for trained model weights (.pth files) |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster training

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/wine-quality-api.git
   cd wine-quality-api
   ```

2. **Create virtual environment**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   
   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure dataset path**
   
   Edit `controllers/model_controllers.py` and update the `file_path` variable:
   ```python
   file_path = r"path/to/your/data/processed_winequality.csv"
   ```

5. **Start the server**
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Access the API**
   - **Swagger UI**: http://127.0.0.1:8000/docs
   - **ReDoc**: http://127.0.0.1:8000/redoc
   - **API Base**: http://127.0.0.1:8000/model

---

## 📡 API Reference

All endpoints are mounted under the `/model` prefix.

### 1. Train Model

**Endpoint:** `POST /model/train/`

Train a new model with custom hyperparameters.

**Request Body:**
```json
{
  "model_type": "classification",
  "epochs": 50,
  "learning_rate": 0.001,
  "optimizer": "Adam",
  "loss_function": "CrossEntropyLoss",
  "output_feature": "type",
  "model_file_name": "wine_class_model.pth"
}
```

**Parameters:**

| Field | Type | Options | Description |
|-------|------|---------|-------------|
| `model_type` | string | `classification`, `regression` | Task type |
| `epochs` | integer | 1-1000 | Training iterations |
| `learning_rate` | float | 0.0001-0.1 | Optimizer learning rate |
| `optimizer` | string | `Adam`, `SGD`, `RMSprop` | Optimization algorithm |
| `loss_function` | string | `CrossEntropyLoss`, `MSELoss`, `L1Loss` | Loss function |
| `output_feature` | string | `type`, `quality` | Target column |
| `model_file_name` | string | Any valid filename | Save location |

**Response:**
```json
{
  "status": "Training completed successfully",
  "model_type": "classification",
  "optimizer": "Adam",
  "loss_function": "CrossEntropyLoss",
  "epochs": 50,
  "learning_rate": 0.001,
  "model_path": "Models/trained/classification_20250102_143022.pth"
}
```

---

### 2. Test Model

**Endpoint:** `POST /model/test/`

Evaluate a saved model on the test dataset.

**Request Body:**
```json
{
  "model_type": "regression",
  "epochs": 50,
  "learning_rate": 0.001,
  "optimizer": "Adam",
  "loss_function": "MSELoss",
  "output_feature": "quality",
  "model_file_name": "wine_regression_model.pth"
}
```

**Response:**
```json
{
  "status": "Model evaluation completed successfully",
  "model_type": "regression",
  "model_file_name": "wine_regression_model.pth",
  "results": {
    "loss": 0.234,
    "r2_score": 0.82,
    "mae": 0.45
  }
}
```

---

### 3. Predict

**Endpoint:** `POST /model/predict`

Run inference on new data using a trained model.

**Request Body:**
```json
{
  "model_type": "classification",
  "model_file_name": "wine_classification_model.pth",
  "input_data": [
    [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0.0]
  ],
  "device": "cpu"
}
```

**Input Features (in order):**
1. Fixed acidity
2. Volatile acidity
3. Citric acid
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol
12. (Reserved for encoding)

**Response:**
```json
{
  "status": "Prediction successful",
  "model_type": "classification",
  "predictions": [1]
}
```

**Prediction Outputs:**
- **Classification**: `0` (red wine) or `1` (white wine)
- **Regression**: Float value between 0-10 (quality score)

---

## 💡 Usage Examples

### Training a Classification Model

**PowerShell:**
```powershell
$body = @{
    model_type = "classification"
    epochs = 100
    learning_rate = 0.001
    optimizer = "Adam"
    loss_function = "CrossEntropyLoss"
    output_feature = "type"
    model_file_name = "wine_classifier_v1.pth"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/model/train/" -Method POST -Body $body -ContentType "application/json"
```

**cURL:**
```bash
curl -X POST "http://127.0.0.1:8000/model/train/" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "classification",
    "epochs": 100,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "loss_function": "CrossEntropyLoss",
    "output_feature": "type",
    "model_file_name": "wine_classifier_v1.pth"
  }'
```

**Python:**
```python
import requests

payload = {
    "model_type": "classification",
    "epochs": 100,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "loss_function": "CrossEntropyLoss",
    "output_feature": "type",
    "model_file_name": "wine_classifier_v1.pth"
}

response = requests.post(
    "http://127.0.0.1:8000/model/train/",
    json=payload
)
print(response.json())
```

---

### Making Predictions

**Python:**
```python
import requests

# Sample wine features
wine_sample = [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0.0]

payload = {
    "model_type": "classification",
    "model_file_name": "wine_classifier_v1.pth",
    "input_data": [wine_sample],
    "device": "cpu"
}

response = requests.post(
    "http://127.0.0.1:8000/model/predict",
    json=payload
)

result = response.json()
wine_type = "White Wine" if result["predictions"][0] == 1 else "Red Wine"
print(f"Prediction: {wine_type}")
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
RELOAD=true

# Dataset Path
DATA_PATH=data/processed_winequality.csv

# Model Storage
MODEL_DIR=Models/trained/

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/training.log

# PyTorch
CUDA_VISIBLE_DEVICES=0
```

### Hyperparameter Tuning

Recommended starting points:

**Classification:**
- Epochs: 50-150
- Learning Rate: 0.001-0.01
- Optimizer: Adam
- Loss: CrossEntropyLoss

**Regression:**
- Epochs: 100-200
- Learning Rate: 0.0001-0.001
- Optimizer: Adam
- Loss: MSELoss or L1Loss

---

## 🔧 Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
# Format with black
black .

# Sort imports
isort .

# Lint with flake8
flake8 .
```

### Logging

Logs are written to:
- **File**: `logs/training.log`
- **Console**: stdout (during development with `--reload`)

Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

---

## 🤝 Contributing

Contributions are welcome! This is a learning project, so feel free to experiment and suggest improvements.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Viraj Gavade**

A learning project exploring modern ML deployment patterns with FastAPI and PyTorch. Built to demonstrate modular pipeline architecture, clean API design, and MLOps best practices.

---

## 🙏 Acknowledgments

- FastAPI framework for excellent developer experience
- PyTorch team for the powerful ML library
- UCI Machine Learning Repository for the wine quality dataset

---

## 📞 Support

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/wine-quality-api/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/wine-quality-api/discussions)

---

**⭐ If you find this project helpful, please consider giving it a star!**