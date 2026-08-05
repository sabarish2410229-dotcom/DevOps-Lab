# Boston Housing — DVC Pipeline

Converts the `boston.ipynb` notebook (Boston Housing price regression)
into a 5-stage DVC pipeline: ingestion -> preprocessing -> feature
engineering -> model building -> evaluation.

## 0. Get the data
Place `HousingData.csv` (the same file the notebook read from
`D:\sem 5\devops\archive\HousingData.csv`) at:
```
data/external/HousingData.csv
```
(Path is configurable in `params.yaml` under `data_ingestion.raw_data_path`.)

## 1. Install dependencies
```bash
pip install -r requirements.txt
```

## 2. Initialize Git + DVC
```bash
git init
dvc init
```

## 3. Run the pipeline
```bash
dvc repro
```

## 4. View the DAG
```bash
dvc dag
```

## 5. View metrics
```bash
dvc metrics show
```

## 6. Commit to Git
```bash
git add .
git commit -m "Initial DVC pipeline"
```

---

**Re-run after changing code/params:**
```bash
dvc repro
```

**Force re-run everything:**
```bash
dvc repro -f
```

## Pipeline stages
| Stage | Script | Input | Output |
|---|---|---|---|
| data_ingestion | `src/data_ingestion.py` | `data/external/HousingData.csv` | `data/raw/data.csv` |
| data_preprocessing | `src/data_preprocessing.py` | `data/raw/data.csv` | `data/processed/data.csv` |
| feature_engineering | `src/feature_engineering.py` | `data/processed/data.csv` | `data/features/{train,test}.csv`, `scaler.pkl` |
| model_building | `src/model_building.py` | `data/features/train.csv` | `model.pkl` (RandomForestRegressor) |
| model_evaluation | `src/model_evaluation.py` | `model.pkl`, `data/features/test.csv` | `metrics.json` (MAE, MSE, RMSE, R2) |

Tune hyperparameters in `params.yaml` — DVC will detect changes and
know which stages need to re-run on the next `dvc repro`.
