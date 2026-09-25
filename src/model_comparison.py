import os

from src.train_model import train_linear_model
from src.train_random_forest import train_random_forest
from src.train_gradient_boosting import train_gradient_boosting
from src.save_model import save_model

REPORT_PATH = "reports/metrics_report.txt"


def run_comparison():
    results = []

    linear_pipeline, linear_metrics = train_linear_model()
    results.append((linear_pipeline, linear_metrics))

    rf_pipeline, rf_metrics = train_random_forest()
    results.append((rf_pipeline, rf_metrics))

    gb_pipeline, gb_metrics = train_gradient_boosting()
    results.append((gb_pipeline, gb_metrics))

    # Pick the model with the highest R2 on the holdout set
    best_pipeline, best_metrics = max(results, key=lambda item: item[1]["r2"])

    print("--- Model Comparison Summary ---")
    for _, metrics in results:
        print(f"{metrics['model']:<28} MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  R2={metrics['r2']:.3f}")
    print(f"\nBest model: {best_metrics['model']} (R2={best_metrics['r2']:.3f})")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("Model Comparison Report\n")
        f.write("========================\n\n")
        for _, metrics in results:
            f.write(f"{metrics['model']}\n")
            f.write(f"  MAE:  {metrics['mae']:.2f}\n")
            f.write(f"  RMSE: {metrics['rmse']:.2f}\n")
            f.write(f"  R2:   {metrics['r2']:.3f}\n\n")
        f.write(f"Best model: {best_metrics['model']} (R2={best_metrics['r2']:.3f})\n")

    save_model(best_pipeline)

    return best_pipeline, best_metrics


if __name__ == "__main__":
    run_comparison()
