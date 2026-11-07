# Vertebral Column Condition — End-to-End MLOps

Predict a patient's orthopedic condition — **Normal**, **Type_H** (disk hernia)
or **Type_S** (spondylolisthesis) — from six biomechanics measurements, built as
a complete, reproducible MLOps project (310 patients, multiclass classification).

This repository started as an exploratory KNN notebook (`Healthcare_KNN.ipynb`)
and has been rebuilt into a modular, config-driven, testable, servable pipeline.
The data ships as one CSV per class (`Part1 - Normal/Type_H/Type_S.csv`); the
ingestion stage stacks them row-wise.

## Architecture

```
   data/raw/Part1 - {Normal,Type_H,Type_S}.csv   (one file per class)
                     │  stacked row-wise on ingest
   ┌─────────────────▼──────────────────────────────────────────┐
   │                   Training pipeline (DVC DAG)                │
   │  1. Data Ingestion      stack the 3 class files              │
   │  2. Data Validation     schema.yaml contract → status.json   │
   │  3. Data Transformation label-encode target, impute + scale  │
   │  4. Model Trainer       KNN in an sklearn Pipeline           │
   │  5. Model Evaluation    macro metrics + OvR ROC-AUC + MLflow │
   └─────────────────┬──────────────────────────────────────────┘
                     │  model.joblib + label_encoder.joblib
                     ▼
            FastAPI serving  (/health, /predict, /train)  → Docker
```

Driven by three files: `config/config.yaml` (paths), `params.yaml`
(hyperparameters), `schema.yaml` (data contract, incl. the valid classes).

## Project layout

```
src/vertebral/
  components/     ingestion · validation · transformation · trainer · evaluation
  pipeline/       one runnable stage per component + prediction pipeline
  config/         ConfigurationManager (YAML → typed configs)
  entity/         frozen dataclass configs
  utils/          yaml/json/joblib helpers
  logger.py       rotating-file + stdout logging
  exception.py    error wrapper with file/line context
app/              FastAPI serving layer (schemas + endpoints)
tests/            unit (validation) + end-to-end (pipeline + API) tests
dvc.yaml          reproducible pipeline DAG
Dockerfile        trains at build, serves with uvicorn
.github/workflows CI: lint → train → test → build image
```

## Quickstart

```bash
pip install -r requirements.txt     # install (editable, with dev extras)
python main.py                      # run the full training pipeline
uvicorn app.main:app --reload       # serve the model (see /docs)
```

### Reproduce with DVC

```bash
dvc repro          # re-runs only the stages whose deps/params changed
dvc metrics show   # prints the evaluation metrics
```

### Experiment tracking (MLflow)

```bash
mlflow ui          # browse runs at http://127.0.0.1:5000
```

### Docker

```bash
docker build -t vertebral-column .
docker run -p 8000:8000 vertebral-column
```

### Tests

```bash
pytest -q
```

## Example prediction request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "P_incidence": 63.03, "P_tilt": 22.55, "L_angle": 39.61,
    "S_slope": 40.48, "P_radius": 98.67, "S_Degree": -0.25
  }'
```

```json
{
  "prediction": "Type_H",
  "probabilities": { "Normal": 0.20, "Type_H": 0.67, "Type_S": 0.13 }
}
```

## Notes

- **Multiclass:** the target is label-encoded in transformation and the encoder
  is saved as its own artifact so the serving layer can decode integer
  predictions back to class names.
- **KNN is distance-based**, so feature scaling (`StandardScaler` in the
  preprocessor) is essential and part of the saved serving pipeline.
- Metrics are macro-averaged across the three classes; ROC-AUC is one-vs-rest.
- Generated artifacts (`artifacts/`, `logs/`, `mlruns/`) are gitignored.
