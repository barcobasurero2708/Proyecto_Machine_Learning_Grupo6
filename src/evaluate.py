"""Métricas, análisis de errores por segmento y figuras de la entrega final.

Todas las métricas están en puntos porcentuales (pp) de la tarifa base.
No se usa MAPE porque hay objetivos iguales a cero.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

from src.features import AIRPORT_ZONES, quality_flags

BLUE, ORANGE, GREEN, GREY, RED = "#236b8e", "#d98131", "#517e63", "#9aa3ab", "#b34721"
FARE_BINS = [0, 1, 5, 10, 20, 50, np.inf]
DIST_BINS = [-0.01, 1, 2, 5, 10, 20, np.inf]
HOUR_BANDS = {"madrugada (0-5)": range(0, 6), "mañana pico (6-9)": range(6, 10),
              "día (10-15)": range(10, 16), "tarde pico (16-19)": range(16, 20),
              "noche (20-23)": range(20, 24)}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})


def regression_metrics(y_true, y_pred, fare=None) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = np.abs(y_true - y_pred)
    out = {
        "n": int(len(y_true)),
        "MAE_pp": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE_pp": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 6),
        "MedAE_pp": round(float(median_absolute_error(y_true, y_pred)), 4),
        "P90_AE_pp": round(float(np.quantile(err, 0.9)), 4),
    }
    if fare is not None:
        # Diagnóstico de la entrega 1: no sustituye la métrica principal.
        mask = np.asarray(fare) >= 1
        out["RMSE_pp_tarifa_ge_1"] = round(float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))), 4)
    return out


def _bins(series, edges, fmt):
    labels = [fmt(a, b) for a, b in zip(edges[:-1], edges[1:])]
    return pd.cut(series, edges, labels=labels, include_lowest=True)


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    seg = pd.DataFrame(index=df.index)
    seg["tarifa base (USD)"] = _bins(df["fare_amount"], FARE_BINS,
                                     lambda a, b: f"{a:g}+" if np.isinf(b) else f"{a:g}-{b:g}")
    seg["distancia (mi)"] = _bins(df["trip_distance"], DIST_BINS,
                                  lambda a, b: f"{max(a, 0):g}+" if np.isinf(b) else f"{max(a, 0):g}-{b:g}")
    band = pd.Series(index=df.index, dtype=object)
    for name, hours in HOUR_BANDS.items():
        band[df["pickup_hour"].isin(list(hours))] = name
    seg["franja horaria"] = pd.Categorical(band, categories=list(HOUR_BANDS), ordered=True)
    seg["fin de semana"] = np.where(df["pickup_weekday"] >= 5, "sí", "no")
    seg["aeropuerto"] = np.where(df["PULocationID"].isin(AIRPORT_ZONES), "desde aeropuerto",
                                 np.where(df["DOLocationID"].isin(AIRPORT_ZONES), "hacia aeropuerto",
                                          "sin aeropuerto"))
    seg["VendorID"] = df["VendorID"].astype(str)
    seg["RatecodeID faltante"] = np.where(df["RatecodeID"].isna(), "sí", "no")
    seg["registro sospechoso"] = np.where(quality_flags(df).any(axis=1), "sí", "no")
    # Segmento definido por la etiqueta: solo para diagnóstico, nunca como variable.
    seg["propina real"] = np.where(df["tip_percentage"].eq(0), "cero", "positiva")
    return seg


def segment_errors(df: pd.DataFrame, y_pred, baseline_pred=None) -> pd.DataFrame:
    """MAE, sesgo medio (predicho - real) y peso de cada segmento."""
    y = df["tip_percentage"].to_numpy()
    seg = add_segments(df)
    work = pd.DataFrame({"y": y, "abs": np.abs(y - y_pred), "bias": y_pred - y}, index=df.index)
    if baseline_pred is not None:
        work["abs_base"] = np.abs(y - baseline_pred)
    tables = []
    for column in seg.columns:
        g = work.groupby(seg[column], observed=True)
        t = pd.DataFrame({"n": g.size(), "pct_viajes": (100 * g.size() / len(work)).round(2),
                          "mediana_real_pp": g["y"].median().round(2),
                          "MAE_pp": g["abs"].mean().round(3), "MedAE_pp": g["abs"].median().round(3),
                          "sesgo_medio_pp": g["bias"].mean().round(3)})
        if baseline_pred is not None:
            t["MAE_baseline_pp"] = g["abs_base"].mean().round(3)
            t["mejora_pct"] = (100 * (1 - t["MAE_pp"] / t["MAE_baseline_pp"])).round(1)
        t.insert(0, "segmento", column)
        t.index.name = "valor"
        tables.append(t.reset_index())
    return pd.concat(tables, ignore_index=True)


def zone_errors(df: pd.DataFrame, y_pred, min_n: int = 5000) -> pd.DataFrame:
    y = df["tip_percentage"].to_numpy()
    work = pd.DataFrame({"zona_inicio": df["PULocationID"].to_numpy(), "abs": np.abs(y - y_pred),
                         "bias": y_pred - y, "y": y, "cero": y == 0})
    t = work.groupby("zona_inicio").agg(n=("abs", "size"), MAE_pp=("abs", "mean"),
                                         sesgo_medio_pp=("bias", "mean"), mediana_real_pp=("y", "median"),
                                         pct_propina_cero=("cero", "mean"))
    t = t[t.n >= min_n].copy()
    t["pct_propina_cero"] *= 100
    return t.round(3).sort_values("MAE_pp", ascending=False).reset_index()


# --------------------------------------------------------------------------- figuras
def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_monthly_target(drift: pd.DataFrame, samples: dict, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = [BLUE, ORANGE, GREEN]
    for (month, y), color in zip(samples.items(), colors):
        axes[0].hist(np.clip(y, 0, 60), bins=np.arange(0, 61, 1), histtype="step", density=True,
                     lw=1.5, color=color, label=month)
    axes[0].set(xlabel="Propina / tarifa base (%); valores > 60 acumulados en 60", ylabel="Densidad",
                title="Distribución del objetivo por mes")
    axes[0].legend(frameon=False)
    x = np.arange(len(drift))
    axes[1].bar(x - .2, drift["mediana"], .4, color=BLUE, label="Mediana (%)")
    axes[1].bar(x + .2, drift["pct_cero"], .4, color=ORANGE, label="Viajes con propina 0 (%)")
    axes[1].set(xticks=x, xticklabels=drift["mes"], title="Mediana y proporción de ceros")
    for i, (m, z) in enumerate(zip(drift["mediana"], drift["pct_cero"])):
        axes[1].text(i - .2, m + .4, f"{m:.1f}", ha="center", fontsize=9)
        axes[1].text(i + .2, z + .4, f"{z:.1f}", ha="center", fontsize=9)
    axes[1].set_ylim(0, max(drift["mediana"].max(), drift["pct_cero"].max()) * 1.35)
    axes[1].legend(frameon=False, loc="upper center", ncol=2)
    fig.suptitle("9. Deriva temporal del objetivo | poblaciones completas ene-mar", fontsize=13)
    _save(fig, path)


def plot_search(search: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    families = list(dict.fromkeys(search["modelo"]))
    for i, fam in enumerate(families):
        t = search[search.modelo == fam]
        jitter = np.linspace(-.15, .15, len(t)) if len(t) > 1 else [0]
        ax.scatter(i + np.asarray(jitter), t["MAE_pp"], s=45, color=GREY, zorder=2)
        best = t.loc[t["MAE_pp"].idxmin()]
        ax.scatter(i, best["MAE_pp"], s=140, marker="*", color=RED, zorder=3)
        ax.text(i + .2, best["MAE_pp"], f"{best['MAE_pp']:.3f}", va="center", fontsize=9)
    ax.set(xticks=range(len(families)), xticklabels=families, ylabel="MAE (pp) en febrero",
           title="10. Búsqueda de hiperparámetros | entrena enero, valida febrero\n(estrella = configuración elegida por familia)")
    _save(fig, path)


def plot_model_comparison(results: dict, path: Path, title: str, n_label: str):
    names = list(results)
    mae = [results[n]["MAE_pp"] for n in names]
    rmse = [results[n]["RMSE_pp_tarifa_ge_1"] for n in names]
    colors = [GREY if n.startswith("baseline") else BLUE for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.5 * len(names) + 2))
    for ax, values, label in [(axes[0], mae, "MAE (pp) · métrica principal"),
                              (axes[1], rmse, "RMSE (pp) · tarifa ≥ USD 1")]:
        ax.barh(names, values, color=colors)
        for i, v in enumerate(values):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.set_xlim(0, max(values) * 1.2)
    axes[1].set_yticklabels([])
    fig.suptitle(f"{title}\n{n_label}", fontsize=13)
    _save(fig, path)


def plot_pred_vs_real(y_true, y_pred, path: Path, title: str, max_points=300_000, seed=42):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) > max_points:
        idx = rng.choice(len(y_true), max_points, replace=False)
        y_true, y_pred = y_true[idx], y_pred[idx]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    hb = ax.hexbin(y_true, y_pred, gridsize=60, bins="log", cmap="Blues", extent=(0, 60, 0, 60), mincnt=1)
    ax.plot([0, 60], [0, 60], color=RED, lw=1, label="predicción perfecta")
    ax.set(xlim=(0, 60), ylim=(0, 60), xlabel="Propina real (% de la tarifa)",
           ylabel="Propina predicha (% de la tarifa)", title=title)
    ax.legend(loc="upper left", frameon=False)
    fig.colorbar(hb, ax=ax, label="Viajes por celda (log)")
    ax.text(.98, .02, "Vista 0-60%; las métricas usan todos los viajes", transform=ax.transAxes,
            ha="right", fontsize=8)
    _save(fig, path)


def plot_importance(imp: pd.DataFrame, path: Path, title: str):
    imp = imp.sort_values("importancia_media").tail(15)
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(imp) + 1.5))
    ax.barh(imp["feature"], imp["importancia_media"], xerr=imp["desv"], color=BLUE)
    ax.set(xlabel="Aumento del MAE al permutar la variable (pp)", title=title)
    _save(fig, path)


def plot_partial_dependence(pdp: dict, path: Path):
    fig, axes = plt.subplots(1, len(pdp), figsize=(5 * len(pdp), 4))
    for ax, (name, (grid, values)) in zip(np.atleast_1d(axes), pdp.items()):
        ax.plot(grid, values, color=BLUE, lw=2)
        ax.set(xlabel=name, ylabel="Predicción media (%)")
    fig.suptitle("14. Dependencia parcial del modelo final | muestra de validación", fontsize=13)
    _save(fig, path)


def plot_segment(seg: pd.DataFrame, segment: str, path: Path, title: str):
    t = seg[seg["segmento"] == segment]
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - .2, t["MAE_baseline_pp"], .4, color=GREY, label="Baseline (regresión lineal)")
    ax.bar(x + .2, t["MAE_pp"], .4, color=BLUE, label="Modelo final")
    for i, (b, m) in enumerate(zip(t["MAE_baseline_pp"], t["MAE_pp"])):
        ax.text(i + .2, m * 1.08, f"{m:,.1f}", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x, t["valor"], rotation=15)
    ax.set(ylabel="MAE (pp, escala log)", title=title)
    ax2 = ax.twinx()
    ax2.plot(x, t["pct_viajes"], "o--", color=ORANGE)
    ax2.set_ylabel("% de viajes", color=ORANGE)
    ax2.spines["right"].set_visible(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    _save(fig, path)


def plot_error_by_hour(df: pd.DataFrame, y_pred, path: Path):
    y = df["tip_percentage"].to_numpy()
    t = pd.DataFrame({"h": df["pickup_hour"].to_numpy(), "abs": np.abs(y - y_pred),
                      "bias": y_pred - y}).groupby("h")[["abs", "bias"]].median()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t.index, t["abs"], "o-", color=BLUE, label="Mediana del error absoluto")
    ax.plot(t.index, t["bias"], "s--", color=RED, label="Mediana de (predicho − real)")
    ax.axhline(0, color="black", lw=.6)
    ax.set(xticks=range(24), xlabel="Hora de inicio", ylabel="pp",
           title="16. Error del modelo final por hora | validación (marzo)")
    ax.legend(frameon=False)
    _save(fig, path)


def plot_zero_tip(seg: pd.DataFrame, path: Path):
    t = seg[seg["segmento"] == "propina real"].set_index("valor")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(t))
    ax.bar(x - .2, t["MAE_baseline_pp"], .4, color=GREY, label="Baseline lineal")
    ax.bar(x + .2, t["MAE_pp"], .4, color=BLUE, label="Modelo final")
    for i, (b, m) in enumerate(zip(t["MAE_baseline_pp"], t["MAE_pp"])):
        ax.text(i - .2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + .2, m, f"{m:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, [f"{v}\n({p:.1f}% de viajes)" for v, p in zip(t.index, t["pct_viajes"])])
    ax.set(ylabel="MAE (pp)", title="17. Error según la propina real fue cero o positiva\nvalidación (marzo)")
    ax.legend(frameon=False)
    _save(fig, path)
