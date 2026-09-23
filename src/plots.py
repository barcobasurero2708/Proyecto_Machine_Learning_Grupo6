"""Figuras con población, periodo y límites de visualización explícitos."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
import numpy as np

BLUE = "#236b8e"
ORANGE = "#d98131"


def create_figures(result, root):
    path = Path(root) / "reports" / "figures"
    path.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    saved = []

    def save(fig, name):
        fig.tight_layout()
        destination = path / name
        fig.savefig(destination, dpi=155, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(destination)

    summary = result["resumen_mensual"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(3)
    ax.bar(x - .19, summary.originales / 1e6, .38, label="Registros originales", color=BLUE)
    ax.bar(x + .19, summary.elegibles / 1e6, .38, label="Viajes elegibles", color=ORANGE)
    ax.set(xticks=x, xticklabels=["Enero", "Febrero", "Marzo"], ylabel="Millones de viajes",
           title="1. Volumen antes y después de los filtros | 2026")
    ax.legend(frameon=False)
    save(fig, "01_volumen.png")

    miss = result["faltantes"].groupby("variable")[["faltantes", "filas"]].sum()
    miss["pct"] = 100 * miss.faltantes / miss.filas
    miss = miss.sort_values("pct")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(miss.index, miss.pct, color=BLUE)
    ax.set(xlabel="Valores faltantes (%)", title="2. Faltantes en los registros originales | enero-marzo")
    for i, value in enumerate(miss.pct):
        ax.text(value + .3, i, f"{value:.2f}%", va="center", fontsize=8)
    ax.set_xlim(0, max(miss.pct.max() * 1.2, 1))
    save(fig, "02_faltantes.png")

    y = np.concatenate(list(result["targets"].values()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    core = y[y <= 100]
    axes[0].hist(core, bins=np.arange(0, 102, 2), color=BLUE)
    axes[0].set(xlabel="Propina / tarifa base (%)", ylabel="Viajes",
                title=f"3a. Vista de 0 a 100%\n{100 * len(core) / len(y):.2f}% de los viajes elegibles")
    axes[1].hist(np.log10(1 + y), bins=90, color=ORANGE)
    axes[1].set_yscale("log")
    axes[1].set(xlabel="log10(1 + porcentaje de propina)", ylabel="Viajes (escala log)",
                title="3b. Distribución completa\nSe conservan todos los extremos")
    save(fig, "03_distribucion_objetivo.png")

    payment = result["pagos"].groupby("payment_type")[["n", "sin_propina_registrada"]].sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = {0: "0: Flex Fare", 1: "1: Tarjeta", 2: "2: Efectivo", 3: "3: Sin cargo", 4: "4: Disputa"}
    bars = ax.bar([labels.get(int(k), str(k)) for k in payment.index],
                 100 * payment.sin_propina_registrada / payment.n, color=BLUE)
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set(ylim=(0, 113), ylabel="tip_amount = 0 (%)",
           title="4. Ausencia de propina registrada según pago | enero-marzo")
    save(fig, "04_pago_propina_cero.png")

    sample = result["muestra"]
    fig, ax = plt.subplots(figsize=(9, 5))
    values = [sample.loc[sample.mes.eq(m), "tip_percentage"] for m in result["targets"]]
    ax.boxplot(values, tick_labels=["Enero", "Febrero", "Marzo"], showfliers=False,
               patch_artist=True, boxprops={"facecolor": "#a6cfdf"}, medianprops={"color": "#b34721"})
    ax.set(ylabel="Propina / tarifa base (%)", title="5. Variación mensual | muestra aleatoria del 5%")
    ax.text(.02, .02, "Se ocultan los puntos atípicos en esta figura; no se eliminan de las métricas.",
            transform=ax.transAxes, fontsize=9)
    save(fig, "05_boxplot_meses.png")

    hour = sample.groupby("pickup_hour").tip_percentage.agg(["median", "count"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(hour.index, hour["median"], marker="o", color=BLUE)
    axes[0].set(ylabel="Mediana de propina (%)", title="6. Hora de inicio y propina | muestra del 5%, enero-marzo")
    axes[1].bar(hour.index, hour["count"], color=ORANGE)
    axes[1].set(xlabel="Hora de inicio (hora local registrada)", ylabel="Viajes de la muestra", xticks=range(24))
    save(fig, "06_hora_propina.png")

    view = sample.loc[sample.fare_amount.between(1, 100) & sample.tip_percentage.between(0, 100)]
    fig, ax = plt.subplots(figsize=(10, 6))
    plot = ax.hexbin(view.fare_amount, view.tip_percentage, gridsize=65, bins="log", mincnt=1, cmap="Blues")
    fig.colorbar(plot, ax=ax, label="Viajes por celda (escala log)")
    ax.set(xlabel="Tarifa base (USD)", ylabel="Propina / tarifa base (%)",
           title="7. Relación entre tarifa y porcentaje | muestra del 5%\nVista: tarifa de 1 a 100 USD y propina de 0 a 100%")
    save(fig, "07_tarifa_propina.png")

    if "metricas" in result:
        m = result["metricas"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, metric in zip(axes, ["MAE_pp", "RMSE_pp"]):
            bars = ax.bar(m.modelo, m[metric], color=[BLUE, ORANGE, "#517e63"])
            ax.bar_label(bars, fmt="%.2f", padding=4)
            ax.set(ylabel="Puntos porcentuales", title=metric.replace("_pp", ""))
            ax.set_ylim(0, m[metric].max() * 1.18)
        fig.suptitle("8. Baselines | evaluación en todos los viajes elegibles de marzo", fontsize=13)
        save(fig, "08_baselines.png")
    return saved
