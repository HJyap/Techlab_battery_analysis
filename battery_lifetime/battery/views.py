from django.shortcuts import render
from battery.models import BatteryMeasurement
import pandas as pd
import plotly.express as px
import plotly.io as pio


def home(request):
    return render(request, "battery/home.html")


def _parse_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def dashboard(request):
    qs = BatteryMeasurement.objects.values(
        "battery_id",
        "cycle",
        "capacity",
        "temperature",
        "ica_peak1_v",
        "ica_peak1_val",
        "ica_peak2_v",
        "ica_peak2_val",
        "ica_area_abs",
    )

    df = pd.DataFrame.from_records(qs)

    if df.empty:
        return render(request, "battery/dashboard.html", {
            "error": "No data available!"
        })

    numeric_cols = [
        "cycle",
        "capacity",
        "temperature",
        "ica_peak1_v",
        "ica_peak1_val",
        "ica_peak2_v",
        "ica_peak2_val",
        "ica_area_abs",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["battery_id"] = df["battery_id"].astype(str)
    df = df.dropna(subset=["battery_id", "cycle", "capacity", "temperature"])
    df = df.sort_values(["battery_id", "cycle"]).reset_index(drop=True)

    battery_options = sorted(
        df["battery_id"].unique().tolist(),
        key=lambda x: int(x) if str(x).isdigit() else x
    )

    cycle_min_default = int(df["cycle"].min())
    cycle_max_default = int(df["cycle"].max())

    temp_min_default = round(float(df["temperature"].quantile(0.10)), 2)
    temp_max_default = round(float(df["temperature"].quantile(0.90)), 2)

    selected_battery = request.GET.get("battery", "all")
    score_mode = request.GET.get("score_mode", "last_capacity")

    cycle_min = int(_parse_float(request.GET.get("cycle_min"), cycle_min_default))
    cycle_max = int(_parse_float(request.GET.get("cycle_max"), cycle_max_default))
    temp_min = _parse_float(request.GET.get("temp_min"), temp_min_default)
    temp_max = _parse_float(request.GET.get("temp_max"), temp_max_default)

    cycle_min = max(cycle_min, int(df["cycle"].min()))
    cycle_max = min(cycle_max, int(df["cycle"].max()))
    temp_min = max(temp_min, float(df["temperature"].min()))
    temp_max = min(temp_max, float(df["temperature"].max()))

    if cycle_min > cycle_max:
        cycle_min, cycle_max = cycle_max, cycle_min

    if temp_min > temp_max:
        temp_min, temp_max = temp_max, temp_min

    filtered_df = df[
        (df["cycle"] >= cycle_min) &
        (df["cycle"] <= cycle_max) &
        (df["temperature"] >= temp_min) &
        (df["temperature"] <= temp_max)
    ].copy()

    if selected_battery != "all":
        filtered_df = filtered_df[filtered_df["battery_id"] == selected_battery]

    if filtered_df.empty:
        return render(request, "battery/dashboard.html", {
            "error": "No data found for the selected filters.",
            "battery_options": battery_options,
            "selected_battery": selected_battery,
            "score_mode": score_mode,
            "cycle_min": cycle_min,
            "cycle_max": cycle_max,
            "temp_min": round(temp_min, 2),
            "temp_max": round(temp_max, 2),
            "total_batteries": 0,
            "total_records": 0,
        })

    filtered_df = filtered_df.sort_values(["battery_id", "cycle"])

    summary = (
        filtered_df
        .groupby("battery_id", as_index=False)
        .agg(
            record_count=("battery_id", "size"),
            cycle_min=("cycle", "min"),
            cycle_max=("cycle", "max"),
            capacity_first=("capacity", "first"),
            capacity_last=("capacity", "last"),
            capacity_mean=("capacity", "mean"),
            capacity_min=("capacity", "min"),
            capacity_max=("capacity", "max"),
            capacity_std=("capacity", "std"),
            temperature_mean=("temperature", "mean"),
            temperature_max=("temperature", "max"),
            ica_area_mean=("ica_area_abs", "mean"),
        )
    )

    summary["capacity_std"] = summary["capacity_std"].fillna(0)
    summary["retention_pct"] = (
        summary["capacity_last"] / summary["capacity_first"] * 100.0
    )
    summary["capacity_drop"] = (
        summary["capacity_first"] - summary["capacity_last"]
    )
    
    # Degration Rate
    cycle_span = (summary["cycle_max"] - summary["cycle_min"]).replace(0, 1)
    summary["deg_rate"] = summary["capacity_drop"] / cycle_span
    summary["deg_rank"] = summary["deg_rate"].rank(method="min", ascending=True).astype(int)

    score_config = {
        "last_capacity": {
            "column": "capacity_last",
            "label": "Last Capacity",
            "unit": "mAh",
            "ascending": False,
            "description": "Best long-term battery in current filter range",
        },
        "mean_capacity": {
            "column": "capacity_mean",
            "label": "Mean Capacity",
            "unit": "mAh",
            "ascending": False,
            "description": "Best average performance in current filter range",
        },
        "retention_pct": {
            "column": "retention_pct",
            "label": "Capacity Retention",
            "unit": "%",
            "ascending": False,
            "description": "Best ageing stability in current filter range",
        },
        "deg_rate":{
            "column": "deg_rate",
            "label": "Degradation Rate",
            "unit": "mAh/cycle",
            "ascending": True,
            "description": "Slowest degradation in current filter range",
        }
    }

    if score_mode not in score_config:
        score_mode = "last_capacity"

    score_info = score_config[score_mode]
    score_col = score_info["column"]

    summary = summary.sort_values(
        by=[score_col, "capacity_last", "capacity_mean"],
        ascending=[score_info["ascending"], False, False]
    ).reset_index(drop=True)

    best_battery = summary.iloc[0]["battery_id"]
    best_score = float(summary.iloc[0][score_col])

    dark_bg = "#1f2937"
    darker_bg = "#111827"
    text_color = "#f9fafb"
    grid_color = "#374151"
    colors = px.colors.qualitative.Set3

    # Ranking Chart
    fig_bar = px.bar(
        summary,
        x="battery_id",
        y=score_col,
        color="battery_id",
        title=f"Battery Ranking by {score_info['label']}",
        text=score_col,
        color_discrete_sequence=colors,
    )
    text_format = "%{text:.3f}" if score_col == "deg_rate" else "%{text:.2f}"
    fig_bar.update_traces(texttemplate=text_format, textposition="outside")
    fig_bar.update_layout(
        plot_bgcolor=darker_bg,
        paper_bgcolor=dark_bg,
        font_color=text_color,
        title_font_color=text_color,
        xaxis=dict(
            title="Battery ID",
            gridcolor=grid_color,
            linecolor=grid_color,
            showgrid=True,
            tickfont=dict(color=text_color),
        ),
        yaxis=dict(
            title=f"{score_info['label']} ({score_info['unit']})",
            gridcolor=grid_color,
            linecolor=grid_color,
            showgrid=True,
            tickfont=dict(color=text_color),
        ),
        showlegend=False,
    )
    graph_bar = pio.to_html(fig_bar, full_html=False)

    fig_deg = px.bar(
        summary.sort_values("deg_rate"), 
        x="battery_id", 
        y="deg_rate", 
        color="battery_id", 
        title="Mean Capacity Degradation Rate by Battery", 
        text="deg_rate", 
        color_discrete_sequence=colors,
    )
    fig_deg.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_deg.update_layout(
        plot_bgcolor=darker_bg,
        paper_bgcolor=dark_bg,
        font_color=text_color,
        title_font_color=text_color,
        xaxis=dict(
            title="Battery ID",
            gridcolor=grid_color,
            linecolor=grid_color,
            showgrid=True,
            tickfont=dict(color=text_color),
        ),
        yaxis=dict(
            title="Degradation Rate (mAh/cycle)",
            gridcolor=grid_color,
            linecolor=grid_color,
            showgrid=True,
            tickfont=dict(color=text_color),
        ),
        showlegend=False,
    )
    graph_deg = pio.to_html(fig_deg, full_html=False)
    
    # Capacity vs Cycle
    fig_line = px.line(
        filtered_df,
        x="cycle",
        y="capacity",
        color="battery_id",
        markers=True,
        hover_data=["temperature", "ica_area_abs"],
        title="Capacity Degradation over Cycle",
        color_discrete_sequence=colors,
    )
    fig_line.update_layout(
        plot_bgcolor=darker_bg,
        paper_bgcolor=dark_bg,
        font_color=text_color,
        title_font_color=text_color,
        xaxis=dict(
            title="Cycle",
            gridcolor=grid_color,
            linecolor=grid_color,
            showgrid=True,
            tickfont=dict(color=text_color),
        ),
        yaxis=dict(
            title="Capacity (mAh)",
            gridcolor=grid_color,
            linecolor=grid_color,
            showgrid=True,
            tickfont=dict(color=text_color),
        ),
        legend_title_text="Battery ID",
    )
    graph_line = pio.to_html(fig_line, full_html=False)

    # Heatmap
    pivot = (
        filtered_df
        .pivot_table(
            index="battery_id",
            columns="cycle",
            values="capacity",
            aggfunc="mean"
        )
        .sort_index()
    )

    graph_heatmap = None
    if not pivot.empty:
        fig_heatmap = px.imshow(
            pivot,
            aspect="auto",
            labels={
                "x": "Cycle",
                "y": "Battery ID",
                "color": "Capacity (mAh)"
            },
            title="Capacity Heatmap by Battery and Cycle",
        )
        fig_heatmap.update_layout(
            plot_bgcolor=darker_bg,
            paper_bgcolor=dark_bg,
            font_color=text_color,
            title_font_color=text_color,
            xaxis=dict(
                gridcolor=grid_color,
                linecolor=grid_color,
                tickfont=dict(color=text_color),
            ),
            yaxis=dict(
                gridcolor=grid_color,
                linecolor=grid_color,
                tickfont=dict(color=text_color),
            ),
        )
        graph_heatmap = pio.to_html(fig_heatmap, full_html=False)

    # Summary table for template
    summary_table = summary.copy()
    summary_table["capacity_first"] = summary_table["capacity_first"].round(2)
    summary_table["capacity_last"] = summary_table["capacity_last"].round(2)
    summary_table["capacity_mean"] = summary_table["capacity_mean"].round(2)
    summary_table["retention_pct"] = summary_table["retention_pct"].round(2)
    summary_table["temperature_mean"] = summary_table["temperature_mean"].round(2)
    summary_table["deg_rate"] = summary_table["deg_rate"].round(4)
    
    context = {
        "graph_bar": graph_bar,
        "graph_line": graph_line,
        "graph_heatmap": graph_heatmap,
        "graph_deg": graph_deg,

        "best_rank": int(summary.iloc[0]["deg_rank"]),
        "best_battery": best_battery,
        "best_score": round(best_score, 4) if score_col == "deg_rate" else round(best_score, 2),
        "score_label": score_info["label"],
        "score_unit": score_info["unit"],
        "score_description": score_info["description"],

        "battery_options": battery_options,
        "selected_battery": selected_battery,
        "score_mode": score_mode,
        "cycle_min": cycle_min,
        "cycle_max": cycle_max,
        "temp_min": round(temp_min, 2),
        "temp_max": round(temp_max, 2),

        "total_batteries": int(summary["battery_id"].nunique()),
        "total_records": len(filtered_df),

        "summary_rows": summary_table.to_dict(orient="records"),
    }

    return render(request, "battery/dashboard.html", context)