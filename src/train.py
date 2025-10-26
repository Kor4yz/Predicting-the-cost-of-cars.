import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from .data import load_raw, save_processed
from .features import build_transformer
from .models import ridge_pipeline, xgb_pipeline

ART = Path("artifacts"); ART.mkdir(exist_ok=True)
NUM_COLS = ["year","mileage","engine_power","engine_volume"]
CAT_COLS = ["brand","model","fuel","transmission","body","region"]

def train():
    df = load_raw("cars.csv")
    df = df.dropna(subset=["price"]).copy()
    X = df[NUM_COLS + CAT_COLS]; y = df["price"]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    tr = build_transformer(NUM_COLS, CAT_COLS)
    models = {
        "ridge": ridge_pipeline(tr),
        "xgb": xgb_pipeline(tr),
    }
    results = {}
    best_name, best_pipe, best_rmse = None, None, 1e18
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_valid)
        mae = mean_absolute_error(y_valid, pred)
        rmse = mean_squared_error(y_valid, pred, squared=False)
        r2 = r2_score(y_valid, pred)
        results[name] = {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}
        if rmse < best_rmse: best_rmse, best_name, best_pipe = rmse, name, pipe

    joblib.dump(best_pipe, ART/"model.pkl")
    with open(ART/"metrics.json","w") as f: json.dump(results, f, indent=2)
    print("Saved:", best_name, results[best_name])

def predict(input_json: str, model_path: str):
    import joblib, numpy as np
    data = json.loads(Path(input_json).read_text())
    X = pd.DataFrame([data])
    pipe = joblib.load(model_path)
    pred = float(pipe.predict(X)[0])
    print(json.dumps({"prediction": pred}, ensure_ascii=False))

def report():
    print(Path(ART/"metrics.json").read_text())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--predict", action="store_true")
    p.add_argument("--input", type=str, default="sample.json")
    p.add_argument("--model", type=str, default="artifacts/model.pkl")
    p.add_argument("--report", action="store_true")
    args = p.parse_args()
    if args.train: train()
    elif args.predict: predict(args.input, args.model)
    elif args.report: report()

if __name__ == "__main__":
    main()
