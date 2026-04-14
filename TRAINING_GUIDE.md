# 🚀 Quick Start: Train Your DR Detection Models

This guide will get you training the Diabetic Retinopathy detection models immediately.

## ✅ Prerequisites Checklist

- [x] Python 3.11+ installed
- [x] Git installed
- [x] 8GB+ RAM available
- [x] GPU recommended (CUDA-capable if using TensorFlow GPU)

## Step 1: Setup Environment

### 1.1 Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 1.2 Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- TensorFlow 2.16.1 (for CNN)
- Scikit-learn 1.5.0 (for Random Forest)
- XGBoost 2.0.3 (alternative model)
- OpenCV & Pillow (image processing)
- Jupyter (notebooks)
- FastAPI & Uvicorn (API server)
- Plus all dependencies

## Step 2: Prepare Your Data

### 2.1 Directory Structure

Your data should be organized as:

```
data/raw/
├── clinical_data.csv          # Biomarker data (one row per patient)
├── image_labels.csv           # Image metadata and labels
└── images/                    # Folder with fundus photographs
    ├── patient_001.jpg
    ├── patient_002.jpg
    └── ...
```

### 2.2 File Formats

**`clinical_data.csv`** - Has rows with columns:
```csv
patient_id,age,bmi,hba1c,blood_pressure_systolic,blood_pressure_diastolic,cholesterol_total,cholesterol_hdl,cholesterol_ldl,triglycerides,diabetes_duration_years,smoking_status,family_history_dr
P001,55,27.5,7.8,140,90,220,50,150,180,8,1,1
P002,62,31.2,9.1,155,95,250,40,170,200,12,0,0
...
```

**`image_labels.csv`** - Maps images to DR grades:
```csv
filename,label,patient_id
patient_001.jpg,0,P001
patient_002.jpg,2,P002
...
```

**DR Grade Labels:**
- 0 = No DR
- 1 = Mild NPDR
- 2 = Moderate NPDR
- 3 = Severe NPDR
- 4 = Proliferative DR

### 2.3 Generate Dummy Data (For Testing)

If you don't have real data yet:

```bash
python create_dummy_data.py
```

This creates sample data in `data/raw/` for testing the pipeline.

## Step 3: Configure (Optional)

### 3.1 View/Edit `.env`

The `.env` file has sensible defaults:
```bash
cat .env
```

To customize, see the environment variables:
```env
LOG_LEVEL=INFO                          # DEBUG, INFO, WARNING, ERROR
TIER_URGENT_THRESHOLD=0.75              # Risk score threshold for urgent
TIER_MODERATE_THRESHOLD=0.45            # Risk score threshold for moderate
GRADCAM_ENABLED=true                    # Enable Grad-CAM explainability
```

## Step 4: Run Tests (Verify Setup)

Before training, verify everything works:

```bash
# Run all tests
pytest tests/ -v

# Specific test file
pytest tests/test_api.py -v
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=api --cov=src
```

Expected output: All tests should pass (or skip gracefully if models aren't loaded yet).

## Step 5: Train Models

### 5.1 Train All Models (Recommended)

```bash
# This trains both biomarker RF and CNN models in sequence
python -m src.pipeline.train
```

**What happens:**
1. Loads clinical data from `data/raw/clinical_data.csv`
2. Loads fundus images from `data/raw/images/`
3. Preprocesses data (scaling, CLAHE, augmentation)
4. Trains Random Forest on biomarkers → `saved_models/biomarker_model.pkl`
5. Trains CNN on images → `saved_models/cnn_weights.weights.h5`
6. Saves parameters to `saved_models/training_summary.json`

**Typical runtime**: 5-30 minutes (depends on dataset size and GPU)

### 5.2 Train Biomarker Model Only

```bash
python -c "from src.pipeline.train import train_biomarker_pipeline; train_biomarker_pipeline()"
```

### 5.3 Train CNN Model Only

```bash
python -c "from src.pipeline.train import train_cnn_pipeline; train_cnn_pipeline()"
```

### 5.4 Monitor Training in Jupyter

Open a notebook to interactively train and visualize:

```bash
jupyter notebook

# Then open notebooks/03_cnn_prototyping.ipynb
```

## Step 6: Test Models Locally

### 6.1 Run Evaluation

After training, evaluate on test set:

```bash
python -m src.pipeline.evaluate
```

Outputs metrics like:
- Accuracy
- F1-Score
- ROC-AUC
- Confusion matrices
- Per-class performance

### 6.2 Start API Server

Once models are trained:

```bash
# Development mode (auto-reload)
uvicorn api.main:app --reload --port 8000

# Production mode (2 workers)
uvicorn api.main:app --workers 2 --host 0.0.0.0 --port 8000
```

Server will start at `http://localhost:8000`

### 6.3 Test Predictions

In another terminal:

```bash
# Biomarker-only prediction
curl -X POST http://localhost:8000/predict/biomarker \
  -H "Content-Type: application/json" \
  -d '{
    "biomarkers": {
      "age": 55, "bmi": 27.5, "hba1c": 7.8,
      "blood_pressure_systolic": 140, "blood_pressure_diastolic": 90,
      "cholesterol_total": 220, "cholesterol_hdl": 50,
      "cholesterol_ldl": 150, "triglycerides": 180,
      "diabetes_duration_years": 8, "smoking_status": 1,
      "family_history_dr": 1
    }
  }'

# Unified prediction (image + biomarker)
curl -X POST http://localhost:8000/predict/unified \
  -F "file=@data/raw/images/sample.jpg" \
  -F "age=55" \
  -F "bmi=27.5" \
  -F "hba1c=7.8" \
  -F "blood_pressure_systolic=140" \
  -F "blood_pressure_diastolic=90" \
  -F "cholesterol_total=220" \
  -F "cholesterol_hdl=50" \
  -F "cholesterol_ldl=150" \
  -F "triglycerides=180" \
  -F "diabetes_duration_years=8" \
  -F "smoking_status=1" \
  -F "family_history_dr=1"
```

## Step 7: Run with Docker (Optional)

### 7.1 Build & Run

```bash
# Build Docker images
docker-compose build

# Start API + Frontend
docker-compose up
```

Access:
- API: `http://localhost:8000`
- Frontend: `http://localhost:3000`
- API Docs: `http://localhost:8000/docs`

### 7.2 Docker Commands

```bash
# View logs
docker-compose logs -f api

# Run specific service
docker-compose up api

# Stop all
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Troubleshooting

### 🔴 "No module named 'tensorflow'"
```bash
pip install -r requirements.txt --force-reinstall
```

### 🔴 "CUDA not found" (if using GPU)
```bash
# Install GPU-specific TensorFlow
pip install tensorflow[and-cuda]==2.16.1
```

### 🔴 "Out of Memory" during training
- Reduce batch size in `src/config/settings.yaml`
- Use fewer training samples
- Train on another machine with more RAM

### 🔴 "No models found" when running API
Make sure to run training first:
```bash
python -m src.pipeline.train
```

### 🔴 Tests fail with "Health check failed"
This is OK if models aren't trained yet. Test endpoints will mock models.

## Next Steps

After training completes:

1. ✅ **Evaluate Models** → `python -m src.pipeline.evaluate`
2. ✅ **Run API** → `uvicorn api.main:app --reload`
3. ✅ **Test Predictions** → Use curl or Postman
4. ✅ **Deploy** → Use `docker-compose up` for production
5. ✅ **Monitor** → Check logs: `docker-compose logs -f`

## Project Structure

```
├── data/                      # Data files (not in git)
│   └── raw/
│       ├── clinical_data.csv
│       ├── image_labels.csv
│       └── images/
├── notebooks/                 # Jupyter for analytics
├── src/                       # Main code
│   ├── config/settings.yaml
│   ├── data_prep/            # Preprocessing
│   ├── models/               # ML models
│   ├── pipeline/             # train.py, evaluate.py
│   └── explainability/       # Grad-CAM
├── api/                      # FastAPI endpoints
├── tests/                    # Unit tests
├── saved_models/             # Trained weights
├── docker-compose.yml
├── Dockerfile
├── requirements.txt          # ✅ Pinned packages
├── .env                      # ✅ Configuration
└── FIXES_APPLIED.md          # Changes made
```

## Key Improvements in This Setup

✅ **Reproducible** - All versions pinned, same results every time  
✅ **Configurable** - Use `.env` for different environments  
✅ **Tested** - 20+ tests catch regressions  
✅ **Secure** - Non-root Docker, no hardcoded secrets  
✅ **Efficient** - Multi-worker, better health checks  
✅ **Documented** - Steps, troubleshooting, architecture  

## Need Help?

- Check [FIXES_APPLIED.md](FIXES_APPLIED.md) for all improvements
- Review [README.md](README.md) for architecture details
- Run tests: `pytest tests/ -v`
- Check logs: `tail -f logs/*.log` (if applicable)

---

**Status**: All systems ready! 🚀 Start training now!

```bash
python -m src.pipeline.train
```
