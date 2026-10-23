# TrafficSense AI

TrafficSense AI is a college project that classifies traffic sign images with two fine-tuned EfficientNet-B3 models. It includes a Streamlit interface for uploading an image or using a camera, plus prediction history and model statistics.

## Models

- **GTSRB** — 43 German traffic sign classes. Validation accuracy: 83.42%.
- **Indian Traffic Sign Dataset** — 59 classes. Validation accuracy: 76.81%.

The model checkpoints are stored with Git LFS. The label maps and training summaries are included in this repository.

## Run locally

Python 3.10 or newer is recommended.

```bash
git lfs install
git clone https://github.com/Shubh-Pi/Trafficsense-ai.git
cd Trafficsense-ai
python -m venv .venv
```

Activate the virtual environment, then install dependencies and start the app:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate`. On macOS or Linux, use `source .venv/bin/activate`.

## Project files

- `app.py` — Streamlit application entry point.
- `desktop_ui.py`, `mobile_ui.py` — desktop and mobile layouts.
- `shared.py` — model loading, image preprocessing, predictions, and shared UI logic.
- `*_labels.json` — class labels for each model.
- `*_training_summary.json`, `training_summary.json` — training metrics.
- `*_traffic_sign_model.pth` — trained model checkpoints.

## Limitations

This is an educational demonstration. Its predictions can be wrong and must not be used for driving, navigation, or any safety-critical decision.
