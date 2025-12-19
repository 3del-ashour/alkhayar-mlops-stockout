# Stockout Prediction MLOps Level 2

Python-only MLOps project predicting whether an item at a branch will fall below SafetyStockLevel in the next N days (default 7). Uses CSV inputs only, Prefect orchestration, MLflow tracking/registry, LightGBM ensemble, feature hashing, feature crosses, FastAPI serving, Dockerized tooling, and GitLab CI.

## Supervised Task & Problem Reframing
- Final deployed task: **classification** — label `1` when projected stock in next N days will fall below SafetyStockLevel.
- Problem reframing: A regression alternative (predict future QuantitySold) was considered and bucketized into risk classes, but classification was chosen to directly support alerting and simpler threshold-based actionability.

## Data Inputs (CSV)
Schemas (all CSV):
- stock_movement.csv — MovementID, Date, FromBranchID, FromBranchName, ToBranchID, ToBranchName, ItemCode, ItemName, QuantityMoved
- stock_current.csv — BranchID, BranchName, ItemCode, ItemName, CurrentQuantity, ReservedQuantity, SafetyStockLevel, LastUpdatedAt
- sales_transactions.csv — Date, BranchID, BranchName, InvoiceNumber, ItemCode, ItemName, QuantitySold
- item_master.csv — ItemCode, ArabicName, EnglishName, BrandOnBox
- employees.csv — EmployeeID, FullName, Department, BranchID
- branches.csv — BranchID, BranchName, OpeningDate

Date parsing is robust; ingestion supports chunking. Real data is ignored; sample CSVs live in `data/sample` for smoke tests.

## Key Design Choices
- **High-cardinality:** Feature hashing for ItemCode, BranchID, FromBranchID, ToBranchID (movement-derived).
- **Feature crosses:** ItemCode×BranchID and ItemCode×Month (hashed).
- **Temporal movement aggregates:** Net inflow/outflow per BranchID×ItemCode from stock movements.
- **Model:** LightGBM (gradient boosting ensemble). Imbalance handling via random over/under sampling when minority < 20%.
- **Checkpoints:** LightGBM checkpoint saved every 50 iterations to `artifacts/checkpoints/`.
- **Tracking & registry:** MLflow logging + Model Registry; automatic Staging→Production promotion when F1 ≥ 0.7.
- **Orchestration:** Prefect 2.x flow `run_pipeline` (ingest → validate → feature build → train → evaluate → register/promote).
- **Serving:** FastAPI `/predict` returns class, probability, and model version; pulls latest Production model from MLflow.
- **Monitoring/CME:** Rolling evaluation script with PSI/KL drift checks and fallback to previous Production model or rule-based baseline (CurrentQuantity < SafetyStockLevel).
- **CI/CD:** GitLab CI stages: lint (ruff/black), unit tests, component tests, acceptance (sample end-to-end).

## Repository Layout
```
README.md
roles.md
business_outline.md
requirements.txt
.gitlab-ci.yml
/data/sample/                # sample CSVs only
/src/
  config.py
  utils/
  features/
  models/
  pipeline/
  monitoring/
/orchestration/
/serving/
/tests/
/docker/
/ci/scripts/
```

## Quickstart (Docker-first)
Prereqs: Docker + docker-compose.

1) Build & run MLflow + API:
```bash
cd docker
docker-compose up --build
```
This starts MLflow at `http://localhost:5000` and FastAPI at `http://localhost:8000`.

2) Run pipeline locally (uses sample data):
```bash
python -m src.pipeline.orchestrate
```
Artifacts, checkpoints, and MLflow runs will appear under `mlruns/`.

3) Prefect deployment (optional for scheduling):
```bash
python orchestration/prefect_deployment.py
prefect deployment run 'stockout-pipeline/stockout-deployment'
```

4) Serving demo (requires a Production model):
```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
        "BranchID": "B1",
        "ItemCode": "ITM1",
        "Date": "2024-04-01",
        "CurrentQuantity": 120,
        "ReservedQuantity": 10,
        "SafetyStockLevel": 50
      }'
```
Response includes `prediction`, `probability`, and `model_version`.

5) Monitoring & Fallback demo:
- Run CME script (example):
```bash
python - <<'PY'
import pandas as pd
from src.monitoring.cme import run_cme
from pathlib import Path
base = Path('data/sample')
sales = pd.read_csv(base/'sales_transactions.csv')
stock = pd.read_csv(base/'stock_current.csv')
reference = stock['SafetyStockLevel'].apply(lambda x: 0)
print(run_cme(sales, stock, reference))
PY
```
- If drift/perf thresholds are breached, CME triggers rollback to previous Production model or falls back to rule-based baseline (CurrentQuantity < SafetyStockLevel). Events are logged to MLflow.

## CI/CD
GitLab pipeline stages:
- **lint:** ruff + black check (`ci/scripts/run_lint.sh`)
- **unit:** pytest on `tests/unit`
- **component:** pytest on `tests/component`
- **acceptance:** end-to-end sample run (`tests/acceptance`)

## Local Development (non-Docker)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python -m src.pipeline.orchestrate
uvicorn serving.app:app --reload
```

## Constraints Recap
- CSV-only inputs; robust date parsing and chunked ingestion.
- Python-only stack (no Node/PHP/etc.).
- Mandatory patterns implemented: feature hashing, feature crosses, ensemble model, MLflow registry, Prefect orchestration, FastAPI serving, CME + drift + fallback, Docker + GitLab CI.
