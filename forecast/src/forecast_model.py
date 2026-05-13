"""
Forecast-Modell: Rollierendes 12-Monats-Forecast mit Facebook Prophet
Trainingsgrundlage: jeweils letzte 36 Monate (rollierendes Fenster)
Input : forecast/data/raw_data.csv
Output: forecast/data/forecast.csv
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

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
DATA_DIR   = Path(__file__).parent.parent / "data"
INPUT_CSV  = DATA_DIR / "raw_data.csv"
OUTPUT_CSV = DATA_DIR / "forecast.csv"

ZIELGROESSEN    = ["Umsatz_TEUR", "Auftragseingang_TEUR", "Deckungsbeitrag_TEUR"]
TRAIN_MONATE    = 36
FORECAST_MONATE = 12

# Prophet-Parameter je Zielgröße (leicht unterschiedlich tuned)
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
        changepoint_prior_scale=0.10,   # AE volatiler → lockerer
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


# ---------------------------------------------------------------------------
# MAPE-Berechnung
# ---------------------------------------------------------------------------

def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = y_true.abs() > 1e-6
    return (((y_true[mask] - y_pred[mask]).abs() / y_true[mask].abs()) * 100).mean()


# ---------------------------------------------------------------------------
# Einzel-Forecast für eine Produktlinie + Zielgröße
# ---------------------------------------------------------------------------

def forecast_serie(
    df_linie: pd.DataFrame,
    zielgroesse: str,
    linie: str,
) -> pd.DataFrame:
    """
    Trainiert Prophet auf den letzten TRAIN_MONATE Datenpunkten,
    erstellt FORECAST_MONATE-Vorschau, berechnet MAPE via Cross-Validation
    auf dem Trainingsset (Hold-out: letzte 6 Monate des Trainingsfensters).
    """
    # Prophet erwartet 'ds' (Datum) und 'y' (Zielwert)
    serie = (
        df_linie[["Datum", zielgroesse]]
        .rename(columns={"Datum": "ds", zielgroesse: "y"})
        .sort_values("ds")
        .reset_index(drop=True)
    )

    # Rollierendes Fenster: letzte 36 Monate für Training
    train = serie.tail(TRAIN_MONATE).reset_index(drop=True)

    # Hold-out MAPE: Trainiere auf ersten 30 von 36, validiere auf letzten 6
    holdout_split = len(train) - 6
    train_cv  = train.iloc[:holdout_split]
    holdout   = train.iloc[holdout_split:]

    cfg = PROPHET_CONFIG[zielgroesse]
    m_cv = Prophet(**cfg)
    m_cv.fit(train_cv, iter=500)
    future_cv = m_cv.make_future_dataframe(periods=6, freq="MS")
    pred_cv   = m_cv.predict(future_cv)
    pred_holdout = pred_cv[pred_cv["ds"].isin(holdout["ds"])]["yhat"].values
    mape_val  = mape(holdout["y"].reset_index(drop=True),
                     pd.Series(pred_holdout))

    # Vollmodell auf allen 36 Monaten
    m = Prophet(**cfg)
    m.fit(train, iter=500)

    future    = m.make_future_dataframe(periods=FORECAST_MONATE, freq="MS")
    forecast  = m.predict(future)

    # Letztes bekanntes Datum im Training
    letzte_is_date = train["ds"].max()

    # Ergebnistabelle zusammenbauen
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result.columns = ["Datum", "yhat", "yhat_lower", "yhat_upper"]
    result["Produktlinie"] = linie
    result["Zielgroesse"]  = zielgroesse

    # IST-Werte einfügen wo vorhanden
    result = result.merge(
        serie.rename(columns={"ds": "Datum", "y": "ist_wert"}),
        on="Datum", how="left"
    )

    result["ist_forecast"] = result["Datum"].apply(
        lambda d: "Forecast" if d > letzte_is_date else "Ist"
    )
    result["MAPE_Pct"]    = round(mape_val, 2)
    result["Train_Ende"]  = letzte_is_date.strftime("%Y-%m")

    # Negative Forecasts auf 0 cappen (Umsatz kann nicht negativ sein)
    for col in ["yhat", "yhat_lower"]:
        result[col] = result[col].clip(lower=0)

    return result


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def main():
    print(f"Lese Daten: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, sep=";", decimal=",", parse_dates=["Datum"])

    produktlinien = sorted(df["Produktlinie"].unique())
    alle_ergebnisse = []

    for linie in produktlinien:
        df_linie = df[df["Produktlinie"] == linie].copy()

        for zg in ZIELGROESSEN:
            print(f"  → Forecast: {linie} | {zg} … ", end="", flush=True)
            result = forecast_serie(df_linie, zg, linie)
            alle_ergebnisse.append(result)
            mape_val = result["MAPE_Pct"].iloc[0]
            n_fc = (result["ist_forecast"] == "Forecast").sum()
            print(f"MAPE={mape_val:.1f}%  |  {n_fc} Forecast-Monate")

    df_out = (
        pd.concat(alle_ergebnisse, ignore_index=True)
        .sort_values(["Produktlinie", "Zielgroesse", "Datum"])
        .reset_index(drop=True)
    )

    # Spaltenreihenfolge
    cols = [
        "Datum", "Jahr", "Monat",
        "Produktlinie", "Zielgroesse",
        "ist_forecast",
        "ist_wert",
        "yhat", "yhat_lower", "yhat_upper",
        "MAPE_Pct", "Train_Ende",
    ]
    df_out["Jahr"]  = df_out["Datum"].dt.year
    df_out["Monat"] = df_out["Datum"].dt.month
    df_out = df_out[cols]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False, sep=";", decimal=",", encoding="utf-8-sig")

    # Zusammenfassung
    print("\n" + "=" * 60)
    print("MAPE-Übersicht (Hold-out letzte 6 Monate des Trainingsfensters)")
    print("=" * 60)
    summary = (
        df_out[df_out["ist_forecast"] == "Forecast"]
        .drop_duplicates(["Produktlinie", "Zielgroesse"])
        [["Produktlinie", "Zielgroesse", "MAPE_Pct", "Train_Ende"]]
        .sort_values(["Produktlinie", "Zielgroesse"])
    )
    print(summary.to_string(index=False))

    fc_rows = df_out[df_out["ist_forecast"] == "Forecast"]
    print(f"\nForecast-Zeitraum: "
          f"{fc_rows['Datum'].min().strftime('%Y-%m')} – "
          f"{fc_rows['Datum'].max().strftime('%Y-%m')}")

    ges_yhat = (
        fc_rows[fc_rows["Zielgroesse"] == "Umsatz_TEUR"]
        .groupby("Datum")["yhat"].sum()
    )
    print(f"Gesamt-Umsatz-Forecast 12 Monate: {ges_yhat.sum():,.0f} TEUR "
          f"({ges_yhat.sum()/1000:,.1f} Mio. EUR)")
    print(f"\nOutput: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
