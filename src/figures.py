"""Publication-format figures from checked model/error/rank exports."""

import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modeling import ROOT, write_json

FAMILIES = ["linear", "additive", "cart", "boost"]
NAMES = {
    "linear": "Linear ridge",
    "additive": "Additive splines",
    "cart": "Small CART",
    "boost": "Boosted trees",
    "overall_mean": "Overall mean",
    "hour_workday_mean": "Hour × workday mean",
}
COLORS = {
    "linear": "#526981",
    "additive": "#9B5A3C",
    "cart": "#6A7180",
    "boost": "#006D77",
}
WEATHER = {
    "temp": ("Temperature", "#006D77"),
    "atemp": ("Feels-like temp.", "#49769C"),
    "humidity": ("Humidity", "#A47525"),
    "weather": ("Weather category", "#875C96"),
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelcolor": "#243345",
        "text.color": "#243345",
        "xtick.color": "#435368",
        "ytick.color": "#435368",
        "axes.edgecolor": "#CBD5E1",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E5EAF0", linewidth=0.7)
    ax.tick_params(length=0)


def save(fig, name):
    folder = ROOT / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=170)
    fig.savefig(
        folder / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(fig)


def main():
    results = ROOT / "results"
    metrics = pd.read_csv(results / "test_metrics.csv", float_precision="round_trip")
    stability = pd.read_csv(results / "stability.csv", float_precision="round_trip")
    importance = pd.read_csv(results / "importance.csv", float_precision="round_trip")
    full = metrics[metrics.variant == "full"].sort_values("rmse")
    resamples = stability[stability.variant != "full"]
    values = {
        "source_period": "2011training /2012test",
        "test_rows": 8734,
        "evaluation_rows": 1500,
        "resamples_per_family": 12,
        "full_models": full[["model", "family", "mse", "rmse", "mae"]].to_dict(
            "records"
        ),
        "resamples": {},
        "temperature_importance": {},
    }
    for family in FAMILIES:
        part = resamples[resamples.family == family]
        reference = stability[
            (stability.family == family) & (stability.variant == "full")
        ].iloc[0]
        values["resamples"][family] = {
            "reference_weather_leader": reference.top_weather_group,
            "weather_leader_counts": part.top_weather_group.value_counts().to_dict(),
            "headline_changes": int(
                sum(part.top_weather_group != reference.top_weather_group)
            ),
            "temperature_order_reversals": int(part.temperature_order_reversed.sum()),
            "top3_jaccard": part.top3_jaccard.tolist(),
            "spearman": part.spearman.tolist(),
            "joint_temperature_first": int(
                sum(part.joint_temperature_weather_rank == 1)
            ),
        }
        values["temperature_importance"][family] = {}
        for group in ["temp", "atemp", "joint_temperature"]:
            part = importance[
                (importance.family == family) & (importance.group == group)
            ]
            values["temperature_importance"][family][group] = {
                "full": float(part[part.variant == "full"].importance.iloc[0]),
                "resampled": part[part.variant != "full"].importance.tolist(),
            }
    values["input_sha256"] = {
        n: hashlib.sha256((results / n).read_bytes()).hexdigest()
        for n in ["test_metrics.csv", "stability.csv", "importance.csv"]
    }
    write_json(results / "figure_values.json", values)

    fig = plt.figure(figsize=(9.2, 8.4))
    fig.text(
        0.05,
        0.955,
        "Better predictions did not settle the driver headline",
        fontsize=19,
        weight="bold",
    )
    fig.text(
        0.05,
        0.915,
        "Historical bike demand · observed weather · 2011 fit / 2012 test",
        fontsize=12,
        color="#64748B",
    )
    ax = fig.add_axes([0.25, 0.53, 0.67, 0.32])
    y = np.arange(len(full))
    names = [
        NAMES.get(row.family, NAMES.get(row.model, row.model))
        for row in full.itertuples()
    ]
    colors = [COLORS.get(row.family, "#BBC6D1") for row in full.itertuples()]
    ax.barh(y, full.rmse, color=colors, height=0.65)
    for i, value in enumerate(full.rmse):
        ax.text(value + 3, i, f"{value:.1f}", va="center", fontsize=11)
    ax.set(
        yticks=y,
        yticklabels=names,
        xlim=(0, 255),
        xlabel="Test RMSE (rentals per hour; lower is better)",
    )
    ax.invert_yaxis()
    tidy(ax)
    ax.set_title(
        "Prediction error on all 8,734 test hours",
        loc="left",
        fontsize=13,
        pad=12,
        weight="bold",
    )
    ax = fig.add_axes([0.25, 0.15, 0.67, 0.24])
    left = np.zeros(4)
    for group, (label, color) in WEATHER.items():
        counts = np.array(
            [
                values["resamples"][f]["weather_leader_counts"].get(group, 0)
                for f in FAMILIES
            ]
        )
        ax.barh(np.arange(4), counts, left=left, color=color, height=0.65, label=label)
        for i, count in enumerate(counts):
            if count:
                ax.text(
                    left[i] + count / 2,
                    i,
                    str(count),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=11,
                    weight="bold",
                )
        left += counts
    ax.set(
        yticks=np.arange(4),
        yticklabels=[NAMES[f] for f in FAMILIES],
        xlim=(0, 12),
        xticks=[0, 3, 6, 9, 12],
        xlabel="Highest-ranked weather group in 12 training-week resamples",
    )
    ax.invert_yaxis()
    tidy(ax)
    ax.set_title(
        "The reported first-place weather driver changes",
        loc="left",
        fontsize=13,
        pad=12,
        weight="bold",
    )
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper left",
        bbox_to_anchor=(0.05, 0.09),
        ncol=4,
        frameon=False,
        fontsize=10,
        columnspacing=1.1,
    )
    fig.text(
        0.05,
        0.025,
        "Source: UCI Bike Sharing (CC BY 4.0). Retrospective model reliance, not causal weather effects or a deployable forecast.",
        fontsize=9.2,
    )
    save(fig, "decision-summary")

    fig = plt.figure(figsize=(9.2, 5.8))
    fig.text(
        0.05,
        0.94,
        "A stable shortlist can conceal an unstable headline",
        fontsize=19,
        weight="bold",
    )
    fig.text(
        0.05,
        0.885,
        "Each dot is a complete 2011-week resample; the full fit supplies the reference ranking.",
        fontsize=11.5,
        color="#64748B",
    )
    ax = fig.add_axes([0.23, 0.25, 0.39, 0.51])
    for i, family in enumerate(FAMILIES):
        scores = values["resamples"][family]["top3_jaccard"]
        ax.scatter(
            scores,
            i + np.linspace(-0.13, 0.13, 12),
            s=37,
            color=COLORS[family],
            alpha=0.7,
            edgecolors="white",
            linewidths=0.4,
        )
        ax.scatter(
            [np.median(scores)],
            [i],
            s=105,
            marker="D",
            color=COLORS[family],
            edgecolor="#243345",
            linewidth=0.9,
        )
    ax.set(
        yticks=range(4),
        yticklabels=[NAMES[f] for f in FAMILIES],
        xlim=(-0.03, 1.05),
        xticks=[0, 0.25, 0.5, 0.75, 1],
        xlabel="Top-three Jaccard overlap",
    )
    ax.invert_yaxis()
    tidy(ax)
    ax.set_title("Shortlist membership", loc="left", fontsize=13, weight="bold", pad=14)
    ax = fig.add_axes([0.74, 0.25, 0.22, 0.51])
    changes = [values["resamples"][f]["headline_changes"] for f in FAMILIES]
    ax.barh(range(4), changes, color=[COLORS[f] for f in FAMILIES], height=0.53)
    for i, count in enumerate(changes):
        ax.text(count + 0.2, i, f"{count}/12", va="center", fontsize=11)
    ax.set(
        yticks=range(4),
        yticklabels=[],
        xlim=(0, 12),
        xticks=[0, 6, 12],
        xlabel="Changed leader",
    )
    ax.invert_yaxis()
    tidy(ax)
    ax.set_title(
        "Weather leader\nchanged", loc="left", fontsize=11.5, weight="bold", pad=14
    )
    fig.text(
        0.05,
        0.115,
        "Additive splines keep all three names in every resample.\nThe first-place weather driver still changes in 6 of 12.",
        fontsize=11.5,
        weight="bold",
    )
    fig.text(
        0.05,
        0.047,
        "Diamonds: medians. Conditional training-history sensitivity; neither confidence intervals nor independent deployment trials.",
        fontsize=9.2,
    )
    fig.text(
        0.05,
        0.018,
        "Source: UCI 2011–2012 · 1,500 fixed test hours · five common within-month/hour/workday permutations.",
        fontsize=9.2,
    )
    save(fig, "ranking-sensitivity")

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.8))
    fig.subplots_adjust(
        left=0.10, right=0.97, bottom=0.16, top=0.80, wspace=0.25, hspace=0.42
    )
    fig.text(
        0.05,
        0.95,
        "Temperature reliance depends on what is moved together",
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.05,
        0.902,
        "Joint shuffles preserve the observed temperature/feels-like pair, but not every weather relationship.",
        fontsize=11.2,
        color="#64748B",
    )
    groups = ["temp", "atemp", "joint_temperature"]
    for family, ax in zip(FAMILIES, axes.flat):
        part = values["temperature_importance"][family]
        full_values = [part[group]["full"] for group in groups]
        ax.bar(
            range(3),
            full_values,
            color=["#006D77", "#49769C", "#875C96"],
            alpha=0.25,
            width=0.62,
        )
        for j, group in enumerate(groups):
            data = part[group]["resampled"]
            ax.scatter(
                j + np.linspace(-0.14, 0.14, 12),
                data,
                s=16,
                color="#526981",
                alpha=0.65,
            )
            ax.scatter(
                [j], [part[group]["full"]], marker="D", s=43, color="#243345", zorder=4
            )
            ax.annotate(
                f"full {part[group]['full']:.1f}",
                (j, max([part[group]["full"], *data])),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=9.5,
            )
        ax.axhline(0, color="#64748B", lw=0.8)
        ax.set_xticks(
            range(3), ["Temperature", "Feels-like", "Joint pair"], fontsize=10.5
        )
        ax.set_title(NAMES[family], loc="left", fontsize=13, weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)
        ax.set_ylabel("Increase in MSE (rentals²)", fontsize=10.5)
        ax.margins(y=0.24)
    fig.text(
        0.05,
        0.088,
        "Bars/diamonds: full 2011 fit. Small dots: 12 week-resampled fits, each averaging five common permutations.",
        fontsize=9.5,
    )
    fig.text(
        0.05,
        0.047,
        "Model reliance is not a causal effect. The joint value need not equal the sum of separate importance values.",
        fontsize=9.5,
    )
    fig.text(
        0.05,
        0.018,
        "Source: UCI Bike Sharing · 2012 observed weather · 1,500 fixed test hours · no production or forecast claim.",
        fontsize=9.2,
    )
    save(fig, "temperature-grouping")


if __name__ == "__main__":
    main()
