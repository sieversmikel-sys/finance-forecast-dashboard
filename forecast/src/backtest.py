"""
Rollierendes Backtesting: 12 aufeinanderfolgende 1-Monats-Forecasts
Fenster: 36 Monate Training → 1 Monat Forecast → nächster Schritt

Testmonate: Jan 2024 – Dez 2024 (letzte 12 von 48 verfügbaren Monaten)
  Schritt  1: Train Jan 2021 – Dez 2023 → Forecast Jan 2024
  Schritt  2: Train Feb 2021 – Jan 2024 → Forecast Feb 2024
  ...
  Schritt 12: Train Jan 2022 – Nov 2024 → Forecast Dez 2024

Input : forecast/data/raw_data.csv
Output: forecast/data/backtest_results.csv
"""

import warnings
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from prophet import Prophet

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

DATA_DIR    = Path(__file__).parent.parent / "data"
INPUT_CSV   = DATA_DIR / "raw_data.csv"
OUTPUT_CSV  = DATA_DIR / "backtest_results.csv"

TRAIN_MONATE = 36
TEST_MONATE  = 12
ZIELGROESSEN = ["Umsatz_TEUR", "Auftragseingang_TEUR", "Deckungsbeitrag_TEUR"]

PROPHET_CONFIG = {
    "Umsatz_TEUR": dict(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=12.0,
        interval_width=0.80,
    ),
    "Auftragseingang_TEUR": dict(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.10,
        seasonality_prior_scale=15.0,
        interval_width=0.80,
    ),
    "Deckungsbeitrag_TEUR": dict(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=12.0,
        interval_width=0.80,
    ),
}


def error_pct(actual: float, forecast: float) -> float:
    if abs(actual) < 1e-6:
        return np.nan
    return (forecast - actual) / abs(actual) * 100


def backtest_serie(df_linie: pd.DataFrame, zielgroesse: str, linie: str) -> pd.DataFrame:
    use_ae = (zielgroesse == "Umsatz_TEUR")

    serie = (
        df_linie[["Datum", zielgroesse]]
        .rename(columns={"Datum": "ds", zielgroesse: "y"})
        .sort_values("ds")
        .reset_index(drop=True)
    )

    if use_ae:
        ae_hist = (
            df_linie[["Datum", "Auftragseingang_TEUR"]]
            .sort_values("Datum").reset_index(drop=True)
        )
        ae_hist["AE_lag1"] = ae_hist["Auftragseingang_TEUR"].shift(1)
        serie = serie.merge(
            ae_hist[["Datum", "AE_lag1"]].rename(columns={"Datum": "ds"}),
            on="ds", how="left"
        )

    alle_daten = serie["ds"].sort_values().tolist()
    n = len(alle_daten)

    # Testfenster: letzte TEST_MONATE Datenpunkte
    test_start_idx = n - TEST_MONATE  # Index des ersten Testmonats

    rows = []
    for step in range(TEST_MONATE):
        forecast_idx    = test_start_idx + step
        train_end_idx   = forecast_idx          # exklusiv
        train_start_idx = train_end_idx - TRAIN_MONATE

        train    = serie.iloc[train_start_idx:train_end_idx].copy()
        if use_ae:
            train = train.dropna(subset=["AE_lag1"]).reset_index(drop=True)
        test_row = serie.iloc[forecast_idx]

        cfg = PROPHET_CONFIG[zielgroesse]
        m = Prophet(**cfg)
        if use_ae:
            m.add_regressor("AE_lag1")
        m.fit(train, iter=300)

        # 1 Monat voraus
        future = m.make_future_dataframe(periods=1, freq="MS")
        if use_ae:
            future = future.merge(serie[["ds", "AE_lag1"]], on="ds", how="left")
            # Forecast-Monat T: AE_lag1 = tatsächlicher AE von T-1 (letzter Trainingsmonat)
            last_train_ae = float(
                df_linie.loc[df_linie["Datum"] == train["ds"].max(), "Auftragseingang_TEUR"].values[0]
            )
            future.loc[future["ds"] == test_row["ds"], "AE_lag1"] = last_train_ae
            future["AE_lag1"] = future["AE_lag1"].fillna(last_train_ae)

        forecast = m.predict(future)
        pred_row = forecast[forecast["ds"] == test_row["ds"]]

        if pred_row.empty:
            yhat = np.nan
            yhat_lower = np.nan
            yhat_upper = np.nan
        else:
            yhat       = max(pred_row["yhat"].values[0], 0)
            yhat_lower = max(pred_row["yhat_lower"].values[0], 0)
            yhat_upper = max(pred_row["yhat_upper"].values[0], 0)

        actual = test_row["y"]
        err    = error_pct(actual, yhat)

        rows.append({
            "month":         test_row["ds"].strftime("%Y-%m"),
            "Produktlinie":  linie,
            "Zielgroesse":   zielgroesse,
            "actual":        round(actual, 1),
            "forecast":      round(yhat, 1),
            "yhat_lower":    round(yhat_lower, 1),
            "yhat_upper":    round(yhat_upper, 1),
            "error_pct":     round(err, 2),
            "abs_error_pct": round(abs(err), 2),
            "train_start":   train["ds"].min().strftime("%Y-%m"),
            "train_end":     train["ds"].max().strftime("%Y-%m"),
        })

        direction = "↑" if yhat >= actual else "↓"
        print(
            f"      {test_row['ds'].strftime('%Y-%m')}  "
            f"Ist={actual:>7,.0f}  FC={yhat:>7,.0f}  "
            f"err={err:>+6.1f}%  {direction}"
        )

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    line = "=" * 68

    print(f"\n{line}")
    print("BACKTESTING-ZUSAMMENFASSUNG  |  Jan 2024 – Dez 2024")
    print(line)

    # Gesamt-MAPE pro Kombination
    pivot = (
        df.groupby(["Produktlinie", "Zielgroesse"])["abs_error_pct"]
        .mean()
        .rename("Avg_MAPE")
        .reset_index()
    )
    pivot["Avg_MAPE"] = pivot["Avg_MAPE"].round(1)
    print("\nAvg MAPE je Linie + Zielgröße:")
    print(pivot.pivot(index="Produktlinie", columns="Zielgroesse", values="Avg_MAPE").to_string())

    # Nur Umsatz für Monats-Bestenliste
    umsatz = df[df["Zielgroesse"] == "Umsatz_TEUR"].copy()
    gesamt_mape = umsatz["abs_error_pct"].mean()
    bester  = umsatz.loc[umsatz["abs_error_pct"].idxmin()]
    schlech = umsatz.loc[umsatz["abs_error_pct"].idxmax()]

    print(f"\n--- Umsatz_TEUR (alle Linien, n={len(umsatz)}) ---")
    print(f"  Avg MAPE gesamt    : {gesamt_mape:.1f}%")
    print(
        f"  Bester Monat       : {bester['month']}  "
        f"{bester['Produktlinie']}  |MAPE|={bester['abs_error_pct']:.1f}%  "
        f"(Ist={bester['actual']:,.0f} / FC={bester['forecast']:,.0f})"
    )
    print(
        f"  Schlechtester Monat: {schlech['month']}  "
        f"{schlech['Produktlinie']}  |MAPE|={schlech['abs_error_pct']:.1f}%  "
        f"(Ist={schlech['actual']:,.0f} / FC={schlech['forecast']:,.0f})"
    )

    # Monatliche Avg-MAPE über alle Linien (Umsatz)
    monthly = (
        umsatz.groupby("month")["abs_error_pct"]
        .mean()
        .round(1)
        .reset_index()
        .rename(columns={"abs_error_pct": "Avg_MAPE_Pct"})
    )
    print("\nMonatliche Avg-MAPE Umsatz (alle Linien):")
    for _, r in monthly.iterrows():
        bar = "█" * int(r["Avg_MAPE_Pct"] / 2)
        print(f"  {r['month']}  {r['Avg_MAPE_Pct']:>5.1f}%  {bar}")

    print(f"\n{line}")


def main():
    print(f"Lese Daten: {INPUT_CSV}")
    df_raw = pd.read_csv(INPUT_CSV, sep=";", decimal=",", parse_dates=["Datum"])

    produktlinien = sorted(df_raw["Produktlinie"].unique())
    alle = []

    for linie in produktlinien:
        df_linie = df_raw[df_raw["Produktlinie"] == linie].copy()
        for zg in ZIELGROESSEN:
            print(f"\n  [{linie}]  {zg}")
            result = backtest_serie(df_linie, zg, linie)
            alle.append(result)

    df_out = pd.concat(alle, ignore_index=True)
    df_out.sort_values(["Produktlinie", "Zielgroesse", "month"], inplace=True)
    df_out.reset_index(drop=True, inplace=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False, sep=";", decimal=",", encoding="utf-8-sig")

    print_summary(df_out)
    print(f"\nOutput: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
