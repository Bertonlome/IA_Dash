"""
Generic Interdependence Analysis Dashboard
© Benjamin R. Berton 2025 Polytechnique Montreal

Parametric for any Human-Autonomy Team configuration.
Auto-detects team structure from CSV or allows manual definition.
"""
import dash
from dash import html, dcc, dash_table, Input, Output, State, callback_context
import plotly.graph_objects as go
import pandas as pd
import os
import base64
import io
import textwrap

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=["assets/styles.css"])
server = app.server

# ─── Constants ─────────────────────────────────────────────────────────────────
COLOR_OPTIONS = ["red", "yellow", "green", "orange"]
VALID_COLORS = {"red", "yellow", "green", "orange"}


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def wrap_text(text, max_width=30):
    if not isinstance(text, str):
        return ""
    return "<br>".join(textwrap.wrap(text, width=max_width))


# ═══════════════════════════════════════════════════════════════════════════════
# TEAM CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_team_config(df):
    """
    Auto-detect team configuration from a DataFrame.

    Identifies color-valued columns as agent role columns, then groups them
    into team alternatives using a state machine that detects transitions
    between performer (*) and supporter (no *) column groups.

    Returns a config dict or None if detection fails.
    """
    # 1. Find columns whose values are mostly valid colors
    color_columns = []
    for col in df.columns:
        vals = df[col].dropna().astype(str).str.strip().str.lower()
        if len(vals) > 0 and vals.isin(VALID_COLORS).sum() / len(vals) > 0.3:
            color_columns.append(col)

    if not color_columns:
        return None

    # 2. Parse team alternatives with a state machine
    #    Rule: consecutive performer columns (*) form one group, then consecutive
    #    supporter columns form another. When we see a performer after supporters,
    #    a new alternative begins.
    alternatives = []
    current_alt = {"performers": [], "supporters": []}
    state = "start"

    for col in color_columns:
        is_performer = col.endswith("*")
        if is_performer:
            if state == "in_supporters":
                # Transition supporter→performer = new alternative
                alternatives.append(current_alt)
                current_alt = {"performers": [], "supporters": []}
            current_alt["performers"].append(col)
            state = "in_performers"
        else:
            current_alt["supporters"].append(col)
            state = "in_supporters"

    if current_alt["performers"] or current_alt["supporters"]:
        alternatives.append(current_alt)

    # 3. Extract unique agent base names (preserving first-seen order)
    agent_names = []
    seen = set()
    for alt in alternatives:
        for p in alt["performers"]:
            base = p.rstrip("*")
            if base not in seen:
                agent_names.append(base)
                seen.add(base)
        for s in alt["supporters"]:
            if s not in seen:
                agent_names.append(s)
                seen.add(s)

    agents = []
    for name in agent_names:
        atype = "human" if name.lower() in ["human", "pilot", "operator", "crew"] else "autonomous"
        agents.append({"name": name, "type": atype})

    # 4. Detect the task description column
    task_column = None
    for candidate in ["Task Object", "Task"]:
        if candidate in df.columns:
            task_column = candidate
            break
    if task_column is None:
        color_set = set(color_columns)
        skip = {"Row", "Procedure"}
        for col in df.columns:
            if col not in skip and col not in color_set:
                if df[col].dropna().dtype == object:
                    task_column = col
                    break

    # 5. Identify metadata columns (everything not Row/Procedure/task/agent)
    structural = {"Row", "Procedure"}
    if task_column:
        structural.add(task_column)
    color_set = set(color_columns)
    metadata = [c for c in df.columns if c not in structural and c not in color_set]

    config = {
        "agents": agents,
        "alternatives": [
            {"name": f"Team Alternative {i+1}", **alt}
            for i, alt in enumerate(alternatives)
        ],
        "task_column": task_column or "Task",
        "color_columns": color_columns,
        "metadata_columns": metadata,
        "all_columns": list(df.columns),
    }
    return config


def build_config_from_manual(column_str, task_col="Task"):
    """
    Build team config from a manual column specification string.

    Example input: "Human*, TARS, TARS*, Human"
    This will be parsed with the same state machine as CSV detection.
    """
    cols = [c.strip() for c in column_str.split(",") if c.strip()]
    if not cols:
        return None

    # Parse alternatives
    alternatives = []
    current_alt = {"performers": [], "supporters": []}
    state = "start"

    for col in cols:
        is_perf = col.endswith("*")
        if is_perf:
            if state == "in_supporters":
                alternatives.append(current_alt)
                current_alt = {"performers": [], "supporters": []}
            current_alt["performers"].append(col)
            state = "in_performers"
        else:
            current_alt["supporters"].append(col)
            state = "in_supporters"

    if current_alt["performers"] or current_alt["supporters"]:
        alternatives.append(current_alt)

    # Extract agents
    agent_names = []
    seen = set()
    for alt in alternatives:
        for p in alt["performers"]:
            base = p.rstrip("*")
            if base not in seen:
                agent_names.append(base)
                seen.add(base)
        for s in alt["supporters"]:
            if s not in seen:
                agent_names.append(s)
                seen.add(s)

    agents = []
    for name in agent_names:
        atype = "human" if name.lower() in ["human", "pilot", "operator", "crew"] else "autonomous"
        agents.append({"name": name, "type": atype})

    all_columns = (
        ["Row", "Procedure", task_col]
        + cols
        + ["Observability", "Predictability", "Directability"]
    )

    config = {
        "agents": agents,
        "alternatives": [
            {"name": f"Team Alternative {i+1}", **alt}
            for i, alt in enumerate(alternatives)
        ],
        "task_column": task_col,
        "color_columns": cols,
        "metadata_columns": ["Observability", "Predictability", "Directability"],
        "all_columns": all_columns,
    }
    return config


# ─── Config helper accessors ──────────────────────────────────────────────────

def get_agent_columns(config):
    """Ordered list of all agent (color) columns."""
    return config.get("color_columns", [])


def get_performer_columns(config):
    """All performer columns (ending with *) across all alternatives."""
    out = []
    for alt in config.get("alternatives", []):
        out.extend(alt["performers"])
    return out


def get_supporter_columns(config):
    """All supporter columns (no *) across all alternatives."""
    out = []
    for alt in config.get("alternatives", []):
        out.extend(alt["supporters"])
    return out


def get_chosen_performer(row, config, strategy, category_overrides=None):
    """
    Determine the chosen performer column for a task row based on the strategy.

    Strategies:
    - human_baseline / human_full_support: prefer human-type performers
    - agent_whenever_possible / agent_whenever_possible_full_support: prefer autonomous performers
    - most_reliable: best color (green > yellow), human preferred in ties

    Returns column name (e.g. "Human*") or None.
    """
    performer_cols = get_performer_columns(config)
    agent_types = {a["name"]: a["type"] for a in config["agents"]}
    COLOR_PRIORITY = {"green": 1, "yellow": 2, "orange": 3}

    # Gather available performers (non-red)
    available = {}
    for pc in performer_cols:
        val = str(row.get(pc, "") or "").strip().lower()
        if val in VALID_COLORS and val != "red":
            available[pc] = val

    if not available:
        return None

    # Category override check
    if category_overrides:
        cat = str(row.get("Category", "") or "").strip()
        if cat in category_overrides:
            override_type = category_overrides[cat].lower()  # "human" or "autonomous"
            preferred = {
                pc: c for pc, c in available.items()
                if agent_types.get(pc.rstrip("*"), "").lower() == override_type
            }
            if preferred:
                return min(preferred, key=lambda pc: COLOR_PRIORITY.get(preferred[pc], 999))
            return min(available, key=lambda pc: COLOR_PRIORITY.get(available[pc], 999))

    if strategy in ("human_baseline", "human_full_support"):
        human_perfs = {
            pc: c for pc, c in available.items()
            if agent_types.get(pc.rstrip("*"), "").lower() == "human"
        }
        if human_perfs:
            return min(human_perfs, key=lambda pc: COLOR_PRIORITY.get(human_perfs[pc], 999))
        return None

    elif strategy in ("agent_whenever_possible", "agent_whenever_possible_full_support"):
        auto_perfs = {
            pc: c for pc, c in available.items()
            if agent_types.get(pc.rstrip("*"), "").lower() == "autonomous"
        }
        if auto_perfs:
            return min(auto_perfs, key=lambda pc: COLOR_PRIORITY.get(auto_perfs[pc], 999))
        human_perfs = {
            pc: c for pc, c in available.items()
            if agent_types.get(pc.rstrip("*"), "").lower() == "human"
        }
        if human_perfs:
            return min(human_perfs, key=lambda pc: COLOR_PRIORITY.get(human_perfs[pc], 999))
        return None

    elif strategy == "most_reliable":
        def sort_key(pc):
            cprio = COLOR_PRIORITY.get(available[pc], 999)
            tprio = 0 if agent_types.get(pc.rstrip("*"), "").lower() == "human" else 1
            return (cprio, tprio)
        return min(available, key=sort_key)

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def build_table_columns(config):
    """Build DataTable column definitions from config, preserving CSV order."""
    agent_set = set(get_agent_columns(config))
    columns = []
    for col in config.get("all_columns", []):
        d = {"name": col, "id": col}
        if col == "Row":
            d["editable"] = False
        elif col in agent_set:
            d["editable"] = True
            d["presentation"] = "dropdown"
        else:
            d["editable"] = True
        columns.append(d)
    return columns


def build_dropdowns(config):
    return {
        col: {"options": [{"label": c.capitalize(), "value": c} for c in COLOR_OPTIONS]}
        for col in get_agent_columns(config)
    }


def build_borders(config):
    """Thick borders to visually separate team alternatives."""
    borders = []
    for alt in config.get("alternatives", []):
        all_in_alt = alt["performers"] + alt["supporters"]
        if all_in_alt:
            borders.append(
                {"if": {"column_id": all_in_alt[0]}, "borderLeft": "3px solid black"}
            )
            borders.append(
                {"if": {"column_id": all_in_alt[-1]}, "borderRight": "3px solid black"}
            )
    return borders


def style_table(df, config):
    """Conditional styles: color cells match their value."""
    agent_cols = get_agent_columns(config)
    styles = []
    for i, row in df.iterrows():
        for col in agent_cols:
            if col in df.columns:
                val = row[col]
                if isinstance(val, str) and val.strip().lower() in VALID_COLORS:
                    color = val.strip().lower()
                    styles.append({
                        "if": {"row_index": i, "column_id": col},
                        "backgroundColor": color,
                        "color": color,
                        "textAlign": "center",
                        "fontWeight": "bold",
                    })
    return styles


def build_data_table(df, config):
    """Create a new DataTable component from a DataFrame and config."""
    return dash_table.DataTable(
        id="responsibility-table",
        columns=build_table_columns(config),
        data=df.to_dict("records"),
        editable=True,
        row_deletable=True,
        dropdown=build_dropdowns(config),
        style_data_conditional=style_table(df, config),
        style_cell={"textAlign": "left", "padding": "5px", "whiteSpace": "normal"},
        style_cell_conditional=build_borders(config),
        style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
        style_table={"overflowX": "auto", "border": "1px solid lightgrey"},
    )


def create_empty_df(config):
    """Create an empty DataFrame with one blank row for a new team."""
    all_cols = config.get("all_columns", ["Row", "Procedure", "Task"])
    row = {c: "" for c in all_cols}
    row["Row"] = 1
    return pd.DataFrame([row])


def ensure_row_column(df):
    """Make sure the Row column exists and is properly numbered."""
    df = df.copy()
    df["Row"] = range(1, len(df) + 1)
    return df


def config_summary_html(config):
    """Render team config as HTML for display."""
    if not config:
        return html.P("No configuration loaded.")

    agents_str = ", ".join(
        [f"{a['name']} ({a['type']})" for a in config["agents"]]
    )
    alt_items = []
    for alt in config["alternatives"]:
        perfs = ", ".join(alt["performers"])
        sups = ", ".join(alt["supporters"]) if alt["supporters"] else "None"
        alt_items.append(
            html.Li(f"{alt['name']}: Performers [{perfs}] — Supporters [{sups}]")
        )

    return html.Div([
        html.P([html.B("Agents: "), agents_str]),
        html.P([html.B("Task column: "), config["task_column"]]),
        html.P([html.B("Columns: "), ", ".join(config.get("all_columns", []))]),
        html.Ul(alt_items),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

def build_workflow_figure(df, config, procedure=None, highlight_track=None,
                         category_overrides=None, view_mode="full"):
    """
    Build a parametric workflow graph for any team configuration.

    - Dots for each agent column with a valid color
    - Dashed arrows for supporter→performer within an alternative
    - Solid arrows for performer transitions between consecutive tasks
    - Parametric highlighting with multiple strategies
    - Optional performers-only view
    """
    if config is None or df.empty:
        return go.Figure()

    agent_cols = get_agent_columns(config)
    performer_cols = set(get_performer_columns(config))
    task_col = config.get("task_column", "Task")

    # Performers-only view: filter to performer columns only
    if view_mode == "performers":
        agent_cols = [c for c in agent_cols if c in performer_cols]

    # Highlight tracking
    highlight_set = set()
    should_hl_support = highlight_track in (
        "human_full_support", "agent_whenever_possible_full_support", "most_reliable",
    )

    if procedure is not None and "Procedure" in df.columns:
        df = df[df["Procedure"] == procedure].copy()

    if df.empty:
        return go.Figure()

    df = df.reset_index(drop=True)
    df["task_idx"] = df.index
    tasks = df[task_col].tolist() if task_col in df.columns else [f"Task {i}" for i in range(len(df))]
    height = max(400, 100 + len(tasks) * 80)

    agent_pos = {agent: i for i, agent in enumerate(agent_cols)}

    dots = []
    dashed_arrows = []
    solid_arrows = []

    # ── Per-task processing ───────────────────────────────────────────────
    for _, row in df.iterrows():
        task_idx = row["task_idx"]

        # Place dots
        for col in agent_cols:
            if col in df.columns:
                val = str(row.get(col, "") or "").strip().lower()
                if val in VALID_COLORS:
                    dots.append({"task": task_idx, "agent": col, "color": val})

        # Dashed arrows: supporter → performer within each alternative
        for alt in config["alternatives"]:
            active_perfs = [
                pc for pc in alt["performers"]
                if pc in df.columns
                and str(row.get(pc, "") or "").strip().lower() in VALID_COLORS
                and str(row.get(pc, "") or "").strip().lower() != "red"
            ]
            for sc in alt["supporters"]:
                if sc in df.columns:
                    sval = str(row.get(sc, "") or "").strip().lower()
                    if sval in VALID_COLORS and sval != "red":
                        for pc in active_perfs:
                            dashed_arrows.append({
                                "start_agent": sc,
                                "end_agent": pc,
                                "task": task_idx,
                            })

        # Build highlight track entry for current task
        if highlight_track and highlight_track != "none":
            chosen = get_chosen_performer(row, config, highlight_track, category_overrides)
            if chosen and chosen in agent_pos:
                highlight_set.add((task_idx, chosen))

    # ── Solid arrows between consecutive tasks ────────────────────────────
    for i in range(1, len(df)):
        prev_perfs = list({
            d["agent"] for d in dots
            if d["task"] == i - 1 and d["agent"] in performer_cols and d["color"] != "red"
        })
        curr_perfs = list({
            d["agent"] for d in dots
            if d["task"] == i and d["agent"] in performer_cols and d["color"] != "red"
        })
        for pa in prev_perfs:
            for ca in curr_perfs:
                solid_arrows.append({
                    "start_task": i - 1, "start_agent": pa,
                    "end_task": i, "end_agent": ca,
                })

    # ── Render figure ─────────────────────────────────────────────────────
    fig = go.Figure()

    for dot in dots:
        row_data = df.iloc[dot["task"]]
        hover_parts = [
            f"<b>Task:</b> {wrap_text(str(row_data.get(task_col, '')))}",
            f"<b>Agent:</b> {dot['agent']}",
        ]
        for meta in ["Observability", "Predictability", "Directability"]:
            if meta in df.columns:
                hover_parts.append(
                    f"<b>{meta}:</b><br>{wrap_text(str(row_data.get(meta, '')))}"
                )
        fig.add_trace(go.Scatter(
            x=[agent_pos[dot["agent"]]],
            y=[dot["task"]],
            mode="markers",
            marker=dict(size=20, color=dot["color"], symbol="circle"),
            showlegend=False,
            hoverinfo="text",
            hovertext="<br><br>".join(hover_parts),
        ))

    # Group dashed arrows by task for vertical offset
    dashed_by_task = {}
    for arrow in dashed_arrows:
        if arrow["start_agent"] in agent_pos and arrow["end_agent"] in agent_pos:
            dashed_by_task.setdefault(arrow["task"], []).append(arrow)

    for task, arrows in dashed_by_task.items():
        n = len(arrows)
        offsets = [0] * n if n == 1 else [
            -0.08 + 0.16 * i / (n - 1) for i in range(n)
        ]
        for arrow, offset in zip(arrows, offsets):
            is_hl = should_hl_support and (task, arrow["end_agent"]) in highlight_set
            fig.add_shape(
                type="line",
                x0=agent_pos[arrow["start_agent"]], y0=task + offset,
                x1=agent_pos[arrow["end_agent"]], y1=task + offset,
                line=dict(
                    color="crimson" if is_hl else "black",
                    width=4 if is_hl else 2,
                    dash="dot",
                ),
            )

    for arrow in solid_arrows:
        is_hl = bool(highlight_set) and (
            (arrow["start_task"], arrow["start_agent"]) in highlight_set
            and (arrow["end_task"], arrow["end_agent"]) in highlight_set
        )
        fig.add_annotation(
            x=agent_pos[arrow["end_agent"]], y=arrow["end_task"],
            ax=agent_pos[arrow["start_agent"]], ay=arrow["start_task"],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1,
            arrowwidth=4 if is_hl else 2,
            arrowcolor="crimson" if is_hl else "black",
            opacity=0.9,
        )

    # Layout
    title_suffix = f" — {procedure}" if procedure else " (All Procedures)"
    if procedure:
        y_labels = [wrap_text(str(t)) for t in tasks]
    else:
        procs = df["Procedure"].tolist() if "Procedure" in df.columns else [""] * len(tasks)
        y_labels = [wrap_text(f"{p} | {t}") for p, t in zip(procs, tasks)]

    fig.update_layout(
        title=f"Workflow Graph{title_suffix}",
        xaxis=dict(
            tickvals=list(agent_pos.values()),
            ticktext=list(agent_pos.keys()),
            title="Agent",
            showgrid=True, gridcolor="lightgrey",
        ),
        yaxis=dict(
            tickvals=list(range(len(tasks))),
            ticktext=y_labels,
            title="Task",
            autorange="reversed",
            showgrid=True, gridcolor="lightgrey",
        ),
        height=height,
        margin=dict(l=250, r=50, t=50, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# BAR CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def build_capacity_bar_chart(df, config):
    """Bar chart showing performer/supporter capacities per agent."""
    if config is None or df.empty:
        return go.Figure()

    performer_cols = get_performer_columns(config)
    supporter_cols = get_supporter_columns(config)
    color_shades = {
        "green": ["seagreen", "limegreen", "mediumspringgreen", "darkgreen", "forestgreen", "springgreen"],
        "yellow": ["gold", "khaki", "lemonchiffon", "goldenrod", "palegoldenrod", "darkkhaki"],
        "orange": ["darkorange", "orange", "coral", "tomato", "sandybrown", "peru"],
    }

    fig = go.Figure()

    for grade in ["green", "yellow", "orange"]:
        shades = color_shades[grade]
        # Performers
        for i, col in enumerate(performer_cols):
            base_name = col.rstrip("*")
            count = sum(
                1 for _, row in df.iterrows()
                if str(row.get(col, "") or "").strip().lower() == grade
            )
            fig.add_trace(go.Bar(
                name=base_name,
                x=[f"Performer {grade.capitalize()}"],
                y=[count],
                marker_color=shades[i % len(shades)],
                showlegend=False,
                text=[base_name], textposition="outside", textangle=0,
            ))
        # Supporters
        for i, col in enumerate(supporter_cols):
            count = sum(
                1 for _, row in df.iterrows()
                if str(row.get(col, "") or "").strip().lower() == grade
            )
            fig.add_trace(go.Bar(
                name=col,
                x=[f"Supporter {grade.capitalize()}"],
                y=[count],
                marker_color=shades[i % len(shades)],
                showlegend=False,
                text=[col], textposition="outside", textangle=0,
            ))

    fig.update_layout(
        title="Performer and Supporter Capacities",
        xaxis_title="Role and Capacity",
        yaxis_title="Number of Tasks",
        barmode="group", bargap=0.15, bargroupgap=0.1,
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
    )
    return fig


def build_allocation_bar_chart(df, config):
    """Horizontal bar showing allocation type distribution."""
    if config is None or df.empty:
        return go.Figure()

    performer_cols = get_performer_columns(config)
    supporter_cols = get_supporter_columns(config)

    single = 0
    multiple = 0
    interdependent = 0

    for _, row in df.iterrows():
        perfs = [
            c for c in performer_cols
            if c in df.columns
            and str(row.get(c, "") or "").strip().lower() in VALID_COLORS
            and str(row.get(c, "") or "").strip().lower() != "red"
        ]
        sups = [
            c for c in supporter_cols
            if c in df.columns
            and str(row.get(c, "") or "").strip().lower() in VALID_COLORS
            and str(row.get(c, "") or "").strip().lower() != "red"
        ]
        if len(sups) > 0:
            interdependent += 1
        elif len(perfs) == 1:
            single += 1
        elif len(perfs) > 1:
            multiple += 1

    total = max(len(df), 1)
    fig = go.Figure()

    for label, count, color in [
        ("Single Allocation Independent", single, "lightcoral"),
        ("Multiple Allocation Independent", multiple, "lightskyblue"),
        ("Interdependent (Support Available)", interdependent, "lightgreen"),
    ]:
        pct = f"{count / total * 100:.1f}%" if total > 0 else "0%"
        fig.add_trace(go.Bar(
            name=label,
            y=["Task Allocation Types"],
            x=[count],
            orientation="h",
            marker=dict(color=color),
            text=[f"{label}: {count} ({pct})" if count > 0 else ""],
            textposition="inside",
            textfont=dict(color="black", size=12),
            hovertemplate=f"{label}<br>Count: %{{x}}<br>Percentage: {pct}<extra></extra>",
        ))

    fig.update_layout(
        title="Task Type Distribution",
        xaxis_title="Number of Tasks",
        barmode="stack", height=200,
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50),
    )
    return fig


def build_autonomy_bar_chart(df, config):
    """Horizontal bar showing agent autonomy (task continuity)."""
    if config is None or df.empty:
        return go.Figure()

    performer_cols = get_performer_columns(config)
    agent_autonomy = {c: {"autonomous": 0, "non_autonomous": 0} for c in performer_cols}

    prev_performers = []
    for idx, row in df.iterrows():
        current_performers = []
        for col in performer_cols:
            if col in df.columns:
                val = str(row.get(col, "") or "").strip().lower()
                if val in VALID_COLORS and val != "red":
                    current_performers.append(col)
                    if val == "orange":
                        agent_autonomy[col]["non_autonomous"] += 1
                    elif idx == 0 or col not in prev_performers:
                        agent_autonomy[col]["non_autonomous"] += 1
                    else:
                        agent_autonomy[col]["autonomous"] += 1
        prev_performers = current_performers

    fig = go.Figure()
    for col in performer_cols:
        auto = agent_autonomy[col]["autonomous"]
        non_auto = agent_autonomy[col]["non_autonomous"]
        total = auto + non_auto
        if total == 0:
            continue
        fig.add_trace(go.Bar(
            name=f"{col} Non-Auto", y=[col], x=[non_auto], orientation="h",
            marker=dict(color="lightcoral"),
            text=[f"Non-Auto: {non_auto} ({non_auto/total*100:.1f}%)" if non_auto > 0 else ""],
            textposition="inside", textfont=dict(color="black", size=11),
            showlegend=False,
        ))
        fig.add_trace(go.Bar(
            name=f"{col} Auto", y=[col], x=[auto], orientation="h",
            marker=dict(color="lightgreen"),
            text=[f"Auto: {auto} ({auto/total*100:.1f}%)" if auto > 0 else ""],
            textposition="inside", textfont=dict(color="black", size=11),
            showlegend=False,
        ))

    fig.update_layout(
        title="Agent Autonomy: Task Continuity",
        xaxis_title="Number of Tasks", yaxis_title="Agent",
        barmode="stack", height=200 + 50 * len(performer_cols),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=100, r=50, t=50, b=50),
    )
    return fig


def build_most_reliable_bar_chart(df, config):
    """Performer/supporter capacities along the most reliable path."""
    if config is None or df.empty:
        return go.Figure()

    perf_green, perf_yellow, perf_orange = 0, 0, 0
    sup_green, sup_yellow, sup_orange = 0, 0, 0

    for _, row in df.iterrows():
        chosen = get_chosen_performer(row, config, "most_reliable")
        if not chosen:
            continue
        val = str(row.get(chosen, "") or "").strip().lower()
        if val == "green":
            perf_green += 1
        elif val == "yellow":
            perf_yellow += 1
        elif val == "orange":
            perf_orange += 1

        # Find active supporter for the chosen performer's alternative
        for alt in config["alternatives"]:
            if chosen in alt["performers"]:
                for sc in alt["supporters"]:
                    if sc in df.columns:
                        sval = str(row.get(sc, "") or "").strip().lower()
                        if sval == "green":
                            sup_green += 1
                        elif sval == "yellow":
                            sup_yellow += 1
                        elif sval == "orange":
                            sup_orange += 1

    fig = go.Figure()
    for label, count, color in [
        ("Performer Green", perf_green, "seagreen"),
        ("Performer Yellow", perf_yellow, "gold"),
        ("Performer Orange", perf_orange, "darkorange"),
        ("Supporter Green", sup_green, "limegreen"),
        ("Supporter Yellow", sup_yellow, "khaki"),
        ("Supporter Orange", sup_orange, "sandybrown"),
    ]:
        fig.add_trace(go.Bar(name=label, x=[label], y=[count], marker_color=color, showlegend=False))

    fig.update_layout(
        title="Most Reliable Path: Performer and Supporter Capacities",
        xaxis_title="Role and Capacity", yaxis_title="Number of Tasks",
        barmode="group", bargap=0.15,
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
    )
    return fig


def build_human_baseline_bar_chart(df, config):
    """Human-only performer capacities (no support, no autonomous agents)."""
    if config is None or df.empty:
        return go.Figure()

    agent_types = {a["name"]: a["type"] for a in config["agents"]}
    performer_cols = get_performer_columns(config)
    human_perfs = [
        pc for pc in performer_cols
        if agent_types.get(pc.rstrip("*"), "").lower() == "human"
    ]
    if not human_perfs:
        return go.Figure()

    perf_green, perf_yellow, perf_orange = 0, 0, 0
    for _, row in df.iterrows():
        for pc in human_perfs:
            if pc in df.columns:
                val = str(row.get(pc, "") or "").strip().lower()
                if val == "green":
                    perf_green += 1
                elif val == "yellow":
                    perf_yellow += 1
                elif val == "orange":
                    perf_orange += 1

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Green", x=["Green"], y=[perf_green], marker_color="seagreen", showlegend=False))
    fig.add_trace(go.Bar(name="Yellow", x=["Yellow"], y=[perf_yellow], marker_color="gold", showlegend=False))
    fig.add_trace(go.Bar(name="Orange", x=["Orange"], y=[perf_orange], marker_color="darkorange", showlegend=False))

    fig.update_layout(
        title="Human-Only Baseline: Human Performer Capacities",
        xaxis_title="Capacity Level", yaxis_title="Number of Tasks",
        barmode="group", bargap=0.15,
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
    )
    return fig


# ─── Automation Proportion ────────────────────────────────────────────────────

def compute_automation_proportion_data(df, config, highlight_track, category_overrides=None):
    """
    Compute automation proportion P using Liu & Kaber (2025) method.

    For each category k: p_k = (1/T_k) * sum(w(t))
    Where w(t) = 1.0 if autonomous performer chosen,
                  0.5 if human performer chosen with autonomous support (full_support modes),
                  0.0 if human performer chosen alone.

    P = (1/K) * sum(p_k)

    Returns (P, category_scores_dict) or (None, {}) if not applicable.
    """
    if config is None or df.empty or "Category" not in df.columns:
        return None, {}

    if not highlight_track or highlight_track == "none":
        return None, {}

    agent_types = {a["name"]: a["type"] for a in config["agents"]}
    categories = sorted(df["Category"].dropna().unique())

    if not categories:
        return None, {}

    K = len(categories)
    category_scores = {}
    is_full_support = highlight_track in (
        "human_full_support", "agent_whenever_possible_full_support", "most_reliable",
    )

    for cat in categories:
        cat_df = df[df["Category"] == cat]
        Tk = len(cat_df)
        if Tk == 0:
            continue

        weights = []
        for _, row in cat_df.iterrows():
            chosen = get_chosen_performer(row, config, highlight_track, category_overrides)

            if chosen is None:
                weights.append(0.0)
                continue

            chosen_type = agent_types.get(chosen.rstrip("*"), "autonomous")

            if chosen_type == "autonomous":
                weights.append(1.0)
            elif chosen_type == "human" and is_full_support:
                # Check if autonomous support is available
                has_auto_support = False
                for alt in config["alternatives"]:
                    if chosen in alt["performers"]:
                        for sc in alt["supporters"]:
                            sc_type = agent_types.get(sc.rstrip("*"), sc)
                            if sc_type != "human" and sc in df.columns:
                                sval = str(row.get(sc, "") or "").strip().lower()
                                if sval in VALID_COLORS and sval != "red":
                                    has_auto_support = True
                weights.append(0.5 if has_auto_support else 0.0)
            else:
                weights.append(0.0)

        pk = sum(weights) / Tk
        category_scores[cat] = pk

    P = sum(category_scores.values()) / K if K > 0 else 0.0
    return P, category_scores


# ═══════════════════════════════════════════════════════════════════════════════
# APP LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

SHOW = {}
HIDE = {"display": "none"}

app.layout = html.Div([
    # ── Stores ──
    dcc.Store(id="team-config-store", data=None),
    dcc.Store(id="category-overrides-store", data={}),

    # ── Header ──
    html.Div(
        "© Benjamin R. Berton 2025 Polytechnique Montreal",
        style={"textAlign": "right", "color": "#888", "fontSize": "14px", "marginBottom": "10px"},
    ),
    html.H1("Interdependence Analysis Dashboard", style={"textAlign": "center"}),
    html.P(
        "Generic tool for any Human-Autonomy Team",
        style={"textAlign": "center", "color": "#666", "marginBottom": "30px"},
    ),

    # ══════════════════════════════════════════════════════════════════════
    # SETUP SECTION
    # ══════════════════════════════════════════════════════════════════════
    html.Div(id="setup-section", children=[
        html.Div([
            html.H3("Team Setup", style={"marginBottom": "20px"}),

            # ── Option 1: CSV Upload ──
            html.Div([
                html.H4("Option 1: Load from CSV"),
                html.P(
                    "Upload a CSV and the team structure will be auto-detected from "
                    "columns containing color values (red/yellow/green/orange). "
                    "Columns ending with * are treated as performer roles.",
                    style={"color": "#666", "fontSize": "14px"},
                ),
                dcc.Upload(
                    id="setup-upload",
                    children=html.Div([
                        "📂 Drag and Drop or ",
                        html.A("Select a CSV File", style={"color": "dodgerblue", "cursor": "pointer"}),
                    ]),
                    style={
                        "width": "100%", "height": "60px", "lineHeight": "60px",
                        "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                        "borderColor": "#ccc", "textAlign": "center", "margin": "10px 0",
                    },
                    multiple=False,
                ),
                html.Div(id="upload-status", style={"marginTop": "10px", "fontStyle": "italic"}),
            ], style={"marginBottom": "30px", "padding": "20px", "backgroundColor": "#f9f9f9", "borderRadius": "8px"}),

            # ── Option 2: Manual Setup ──
            html.Div([
                html.H4("Option 2: Define Team Manually"),
                html.P(
                    "Enter the agent role columns in order. Use * for performer columns. "
                    "Group them as: Alt1-performers, Alt1-supporters, Alt2-performers, Alt2-supporters, ...",
                    style={"color": "#666", "fontSize": "14px"},
                ),
                html.Div([
                    html.Label("Agent columns (comma-separated):", style={"fontWeight": "bold"}),
                    dcc.Input(
                        id="manual-columns-input",
                        value="Human*, Robot, Robot*, Human",
                        style={"width": "100%", "marginBottom": "10px", "padding": "8px"},
                    ),
                ]),
                html.Div([
                    html.Label("Task column name:", style={"fontWeight": "bold", "marginRight": "10px"}),
                    dcc.Input(
                        id="manual-task-col-input",
                        value="Task",
                        style={"width": "200px", "padding": "8px"},
                    ),
                ], style={"marginBottom": "15px"}),
                # Preset buttons
                html.Div([
                    html.Label("Presets: ", style={"fontWeight": "bold", "marginRight": "10px"}),
                    html.Button("Human + Robot", id="preset-2agent", n_clicks=0,
                                style={"marginRight": "10px"}),
                    html.Button("Human + UGV + UAV", id="preset-3agent", n_clicks=0,
                                style={"marginRight": "10px"}),
                    html.Button("Human + TARS", id="preset-tars", n_clicks=0),
                ], style={"marginBottom": "15px"}),
                html.Button(
                    "🚀 Create Team", id="create-team-button", n_clicks=0,
                    style={
                        "marginTop": "10px", "padding": "10px 30px", "fontSize": "16px",
                        "backgroundColor": "#4CAF50", "color": "white", "border": "none",
                        "borderRadius": "5px", "cursor": "pointer",
                    },
                ),
            ], style={"padding": "20px", "backgroundColor": "#f0f8ff", "borderRadius": "8px"}),
        ], style={"maxWidth": "800px", "margin": "0 auto", "padding": "20px"}),
    ]),

    # ══════════════════════════════════════════════════════════════════════
    # CONFIG SUMMARY (shown after setup)
    # ══════════════════════════════════════════════════════════════════════
    html.Div(id="config-summary", style=HIDE, children=[
        html.Div([
            html.H4("Current Team Configuration", style={"display": "inline-block"}),
            html.Button("🔄 Change Team", id="reset-config-button", n_clicks=0,
                        style={"marginLeft": "20px", "cursor": "pointer"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div(id="config-details"),
        html.Hr(),
    ]),

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS SECTION
    # ══════════════════════════════════════════════════════════════════════
    html.Div(id="analysis-section", style=HIDE, children=[
        html.H2("Interdependence Analysis Table", style={"textAlign": "center"}),
        html.Div(id="table-wrapper"),

        # Action buttons
        html.Div([
            html.Div([
                html.Div([
                    dcc.Upload(
                        id="upload-data",
                        children=html.Button("📂 Load Table", id="load-button", n_clicks=0),
                        multiple=False,
                        style={"display": "inline-block", "marginRight": "10px"},
                    ),
                    html.Button("➕ Add Row", id="add-row-button", n_clicks=0),
                    html.Button("⬇️ Copy Cell Down", id="copy-down-button", n_clicks=0),
                ], style={"display": "flex", "gap": "10px"}),
                html.Div([
                    html.Button("💾 Save Table", id="save-button", n_clicks=0),
                    dcc.Download(id="download-csv"),
                ], style={"marginLeft": "auto"}),
            ], style={"display": "flex", "width": "100%"}),
            html.Div(id="save-confirmation", style={"marginTop": "10px", "fontStyle": "italic"}),
        ]),

        # ── Workflow Graph ──
        html.H2("Interdependence Workflow Graph", style={"textAlign": "center", "marginTop": "40px"}),
        html.Div([
            dcc.Dropdown(
                id="procedure-dropdown",
                options=[],
                value=None,
                placeholder="Select a procedure to filter the graph...",
                clearable=True,
                style={"width": "50%", "margin": "0 auto"},
            )
        ]),
        # ── View selector ──
        dcc.RadioItems(
            id="view-selector",
            options=[
                {"label": "Full View (All Agents)", "value": "full"},
                {"label": "Performers Only", "value": "performers"},
            ],
            value="full",
            labelStyle={"display": "inline-block", "margin-right": "20px"},
            style={"textAlign": "center", "marginTop": "10px"},
        ),
        dcc.RadioItems(
            id="highlight-selector",
            options=[
                {"label": "No highlight", "value": "none"},
                {"label": "Human-performer-only no support", "value": "human_baseline"},
                {"label": "Human-performer-only full support", "value": "human_full_support"},
                {"label": "Agent-performer-whenever-possible no support", "value": "agent_whenever_possible"},
                {"label": "Agent-performer-whenever-possible full support", "value": "agent_whenever_possible_full_support"},
                {"label": "Most reliable path", "value": "most_reliable"},
            ],
            value="none",
            labelStyle={"display": "inline-block", "margin-right": "20px"},
            style={"textAlign": "center", "marginTop": "10px"},
        ),

        # ── Automation Proportion Summary ──
        html.Div(id="automation-proportion-box", children=[
            html.Div([
                html.Span("Automation Proportion: ",
                           style={"fontSize": "16px", "fontWeight": "bold"}),
                html.Span(id="ap-summary-value", children="--",
                           style={"fontSize": "20px", "fontWeight": "bold", "color": "#2196F3"}),
            ], style={
                "textAlign": "center", "padding": "10px 20px",
                "border": "2px solid #ccc", "borderRadius": "8px",
                "display": "inline-block", "margin": "10px auto",
            }),
            html.Div(id="ap-detail", style={"textAlign": "center", "fontSize": "13px", "color": "#666"}),
        ], style={"textAlign": "center", "marginTop": "10px", "display": "none"}),
        html.Div(id="automation-proportion-results"),
        dcc.Graph(id="interdependence-graph", config={"displayModeBar": False}),

        # Team alternative labels (dynamic)
        html.Div(id="alt-labels"),

        # ── Bar Charts ──
        html.Div([
            dcc.Graph(id="capacity-bar-chart"),
            dcc.Graph(id="most-reliable-bar-chart"),
            dcc.Graph(id="human-baseline-bar-chart"),
            dcc.Graph(id="allocation-type-bar-chart"),
            dcc.Graph(id="agent-autonomy-bar-chart"),
        ]),

        # ── Category Overrides (shown only when Category column exists) ──
        html.Div(id="category-overrides-section", style={"display": "none"}, children=[
            html.H3("Category Overrides", style={"textAlign": "center", "marginTop": "30px"}),
            html.P(
                "Click a bar to force a specific agent type for that category. "
                "Click again to reset to default strategy.",
                style={"textAlign": "center", "color": "#666", "fontSize": "14px"},
            ),
            html.Div(id="category-overrides-container"),
        ]),
    ]),

    # Footer
    html.Footer(
        "© Benjamin R. Berton 2025 Polytechnique Montreal",
        style={
            "textAlign": "center", "marginTop": "40px", "padding": "10px 0",
            "color": "#888", "fontSize": "14px",
        },
    ),
], style={
    "fontFamily": "'Roboto', 'Helvetica', 'Arial', sans-serif",
    "maxWidth": "1400px", "margin": "0 auto", "padding": "20px",
})


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Preset buttons ────────────────────────────────────────────────────────────
@app.callback(
    Output("manual-columns-input", "value"),
    Output("manual-task-col-input", "value"),
    Input("preset-2agent", "n_clicks"),
    Input("preset-3agent", "n_clicks"),
    Input("preset-tars", "n_clicks"),
    prevent_initial_call=True,
)
def apply_preset(n2, n3, nt):
    ctx = callback_context
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    if btn == "preset-2agent":
        return "Human*, Robot, Robot*, Human", "Task"
    elif btn == "preset-3agent":
        return "Human*, UGV, UAV, UGV*, UAV*, Human", "Task"
    elif btn == "preset-tars":
        return "Human*, TARS, TARS*, Human", "Task Object"
    return dash.no_update, dash.no_update


# ── Setup / Config callback ──────────────────────────────────────────────────
@app.callback(
    Output("team-config-store", "data"),
    Output("table-wrapper", "children"),
    Output("analysis-section", "style"),
    Output("setup-section", "style"),
    Output("config-summary", "style"),
    Output("config-details", "children"),
    Output("procedure-dropdown", "options"),
    Output("upload-status", "children"),
    Input("setup-upload", "contents"),
    Input("create-team-button", "n_clicks"),
    Input("reset-config-button", "n_clicks"),
    State("setup-upload", "filename"),
    State("manual-columns-input", "value"),
    State("manual-task-col-input", "value"),
    prevent_initial_call=True,
)
def handle_setup(upload_contents, create_clicks, reset_clicks,
                 upload_filename, manual_columns, manual_task_col):
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    no = dash.no_update

    if triggered == "reset-config-button":
        return None, None, HIDE, SHOW, HIDE, None, [], ""

    if triggered == "setup-upload" and upload_contents:
        try:
            content_type, content_string = upload_contents.split(",")
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        except Exception as e:
            return no, no, no, no, no, no, no, f"⚠️ Error reading file: {e}"

        config = detect_team_config(df)
        if config is None:
            return no, no, no, no, no, no, no, "⚠️ Could not detect team structure. No color columns found."

        # Ensure Row column
        if "Row" not in df.columns:
            df.insert(0, "Row", range(1, len(df) + 1))
            config["all_columns"] = ["Row"] + [c for c in config["all_columns"] if c != "Row"]

        # Procedure dropdown options
        proc_options = []
        if "Procedure" in df.columns:
            proc_options = [{"label": p, "value": p} for p in df["Procedure"].dropna().unique()]

        table = build_data_table(df, config)
        summary = config_summary_html(config)
        return config, table, SHOW, HIDE, SHOW, summary, proc_options, f"✅ Loaded {upload_filename}"

    if triggered == "create-team-button":
        config = build_config_from_manual(manual_columns or "", manual_task_col or "Task")
        if config is None:
            return no, no, no, no, no, no, no, no
        df = create_empty_df(config)
        table = build_data_table(df, config)
        summary = config_summary_html(config)
        return config, table, SHOW, HIDE, SHOW, summary, [], ""

    return no, no, no, no, no, no, no, no


# ── Table operations callback ────────────────────────────────────────────────
@app.callback(
    Output("table-wrapper", "children", allow_duplicate=True),
    Output("save-confirmation", "children"),
    Output("download-csv", "data"),
    Output("procedure-dropdown", "options", allow_duplicate=True),
    Output("team-config-store", "data", allow_duplicate=True),
    Output("analysis-section", "style", allow_duplicate=True),
    Output("setup-section", "style", allow_duplicate=True),
    Output("config-summary", "style", allow_duplicate=True),
    Output("config-details", "children", allow_duplicate=True),
    Input("responsibility-table", "data"),
    Input("save-button", "n_clicks"),
    Input("upload-data", "contents"),
    Input("add-row-button", "n_clicks"),
    Input("copy-down-button", "n_clicks"),
    State("responsibility-table", "active_cell"),
    State("upload-data", "filename"),
    State("team-config-store", "data"),
    prevent_initial_call=True,
)
def handle_table(data, save_clicks, upload_contents, add_clicks, copy_clicks,
                 active_cell, upload_filename, config):
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    no = dash.no_update

    save_msg = ""
    download = None

    # ── Save ──
    if triggered == "save-button" and data and config:
        df = pd.DataFrame(data)
        download = dcc.send_data_frame(df.to_csv, "interdependence_analysis.csv", index=False)
        save_msg = "✅ Table downloaded as interdependence_analysis.csv"
        return no, save_msg, download, no, no, no, no, no, no

    # ── Load new CSV (from analysis section) ──
    if triggered == "upload-data" and upload_contents:
        try:
            ct, cs = upload_contents.split(",")
            decoded = base64.b64decode(cs)
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        except Exception as e:
            return no, f"⚠️ Error: {e}", None, no, no, no, no, no, no

        new_config = detect_team_config(df)
        if new_config is None:
            return no, "⚠️ Could not detect team structure.", None, no, no, no, no, no, no

        if "Row" not in df.columns:
            df.insert(0, "Row", range(1, len(df) + 1))
            new_config["all_columns"] = ["Row"] + [c for c in new_config["all_columns"] if c != "Row"]

        proc_options = []
        if "Procedure" in df.columns:
            proc_options = [{"label": p, "value": p} for p in df["Procedure"].dropna().unique()]

        table = build_data_table(df, new_config)
        summary = config_summary_html(new_config)
        return table, f"✅ Loaded {upload_filename}", None, proc_options, new_config, SHOW, HIDE, SHOW, summary

    # ── Add Row ──
    if triggered == "add-row-button" and data and config:
        df = pd.DataFrame(data)
        new_row = {col: "" for col in df.columns}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df = ensure_row_column(df)
        table = build_data_table(df, config)
        return table, "", None, no, no, no, no, no, no

    # ── Copy Down ──
    if triggered == "copy-down-button" and data and config:
        df = pd.DataFrame(data)
        if active_cell and "row" in active_cell and "column_id" in active_cell:
            r = active_cell["row"]
            c = active_cell["column_id"]
            if r is not None and c is not None and r + 1 < len(df):
                df.at[r + 1, c] = df.at[r, c]
        table = build_data_table(df, config)
        return table, "", None, no, no, no, no, no, no

    # ── Table edited ──
    if triggered == "responsibility-table" and data and config:
        df = pd.DataFrame(data)
        df = ensure_row_column(df)
        proc_options = []
        if "Procedure" in df.columns:
            proc_options = [{"label": p, "value": p} for p in df["Procedure"].dropna().unique()]
        table = build_data_table(df, config)
        return table, "", None, proc_options, no, no, no, no, no

    return no, "", None, no, no, no, no, no, no


# ── Graph callback ────────────────────────────────────────────────────────────
@app.callback(
    Output("interdependence-graph", "figure"),
    Output("alt-labels", "children"),
    Input("procedure-dropdown", "value"),
    Input("highlight-selector", "value"),
    Input("view-selector", "value"),
    Input("responsibility-table", "data"),
    Input("category-overrides-store", "data"),
    State("team-config-store", "data"),
)
def update_graph(procedure, highlight_track, view_mode, data, category_overrides, config):
    if not data or not config:
        return go.Figure(), None

    df = pd.DataFrame(data)
    if df.empty:
        return go.Figure(), None

    ht = None if highlight_track == "none" else highlight_track
    fig = build_workflow_figure(
        df, config, procedure=procedure, highlight_track=ht,
        category_overrides=category_overrides or {}, view_mode=view_mode or "full",
    )

    # Team alternative labels
    labels = []
    for alt in config.get("alternatives", []):
        labels.append(html.Div(
            alt["name"],
            style={
                "display": "inline-block", "textAlign": "center",
                "marginTop": "10px", "fontWeight": "bold",
                "flex": "1",
            },
        ))
    alt_label_div = html.Div(labels, style={
        "display": "flex", "width": "100%",
        "marginLeft": "150px", "marginRight": "50px",
    }) if labels else None

    return fig, alt_label_div


# ── Bar chart callback ────────────────────────────────────────────────────────
@app.callback(
    Output("capacity-bar-chart", "figure"),
    Output("most-reliable-bar-chart", "figure"),
    Output("human-baseline-bar-chart", "figure"),
    Output("allocation-type-bar-chart", "figure"),
    Output("agent-autonomy-bar-chart", "figure"),
    Input("procedure-dropdown", "value"),
    Input("responsibility-table", "data"),
    State("team-config-store", "data"),
)
def update_bar_charts(procedure, data, config):
    if not data or not config:
        return go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure()

    df = pd.DataFrame(data)
    if procedure and "Procedure" in df.columns:
        df = df[df["Procedure"] == procedure]

    return (
        build_capacity_bar_chart(df, config),
        build_most_reliable_bar_chart(df, config),
        build_human_baseline_bar_chart(df, config),
        build_allocation_bar_chart(df, config),
        build_autonomy_bar_chart(df, config),
    )


# ── Automation Proportion callback ────────────────────────────────────────────
@app.callback(
    Output("automation-proportion-box", "style"),
    Output("ap-summary-value", "children"),
    Output("ap-summary-value", "style"),
    Output("ap-detail", "children"),
    Output("automation-proportion-results", "children"),
    Input("highlight-selector", "value"),
    Input("category-overrides-store", "data"),
    Input("procedure-dropdown", "value"),
    State("responsibility-table", "data"),
    State("team-config-store", "data"),
)
def compute_automation_proportion(highlight_track, category_overrides, procedure, data, config):
    if not data or not config:
        return {"display": "none"}, "--", {}, "", None

    df = pd.DataFrame(data)
    if procedure and "Procedure" in df.columns:
        df = df[df["Procedure"] == procedure]

    if "Category" not in df.columns or not highlight_track or highlight_track == "none":
        return {"display": "none"}, "--", {}, "", None

    P, cat_scores = compute_automation_proportion_data(
        df, config, highlight_track, category_overrides or {},
    )

    if P is None:
        return {"display": "none"}, "--", {}, "", None

    # Color code: blue if P > 0.5, green if P < 0.5, orange if P ≈ 0.5
    if P > 0.55:
        color = "#2196F3"
    elif P < 0.45:
        color = "#4CAF50"
    else:
        color = "#FF9800"

    val_style = {"fontSize": "20px", "fontWeight": "bold", "color": color}
    box_style = {"textAlign": "center", "marginTop": "10px"}

    # Category detail
    detail_parts = [f"{cat}: {score:.2f}" for cat, score in sorted(cat_scores.items())]
    detail_text = " | ".join(detail_parts) if detail_parts else ""

    # LaTeX formula display
    formula = html.Div([
        html.P([
            html.B("Formula: "),
            f"P = (1/K) × Σ pₖ = (1/{len(cat_scores)}) × "
            f"{sum(cat_scores.values()):.2f} = {P:.3f}",
        ], style={"fontSize": "13px", "color": "#666", "marginTop": "5px"}),
        html.P([
            html.B("Where: "),
            "w(t) = 1.0 (autonomous), 0.5 (human with autonomous support), 0.0 (human alone)",
        ], style={"fontSize": "12px", "color": "#888"}),
    ])

    return box_style, f"{P:.3f}", val_style, detail_text, formula


# ── Category Overrides callbacks ──────────────────────────────────────────────
@app.callback(
    Output("category-overrides-section", "style"),
    Output("category-overrides-container", "children"),
    Input("responsibility-table", "data"),
    State("team-config-store", "data"),
)
def generate_category_overrides(data, config):
    """Generate per-category override UI with mini bar charts."""
    if not data or not config:
        return {"display": "none"}, None

    df = pd.DataFrame(data)
    if "Category" not in df.columns:
        return {"display": "none"}, None

    agent_types = {a["name"]: a["type"] for a in config["agents"]}
    performer_cols = get_performer_columns(config)
    categories = sorted(df["Category"].dropna().unique())

    if not categories:
        return {"display": "none"}, None

    # Group performers by type
    human_perfs = [
        pc for pc in performer_cols
        if agent_types.get(pc.rstrip("*"), "").lower() == "human"
    ]
    auto_perfs = [
        pc for pc in performer_cols
        if agent_types.get(pc.rstrip("*"), "").lower() == "autonomous"
    ]

    if not human_perfs or not auto_perfs:
        return {"display": "none"}, None

    human_label = "Human"
    auto_label = ", ".join([pc.rstrip("*") for pc in auto_perfs])

    children = []
    for cat in categories:
        cat_df = df[df["Category"] == cat]

        human_assignable = sum(
            1 for _, row in cat_df.iterrows()
            for pc in human_perfs
            if pc in df.columns
            and str(row.get(pc, "") or "").strip().lower() in ("green", "yellow", "orange")
        )
        auto_assignable = sum(
            1 for _, row in cat_df.iterrows()
            for pc in auto_perfs
            if pc in df.columns
            and str(row.get(pc, "") or "").strip().lower() in ("green", "yellow", "orange")
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=[auto_label], x=[auto_assignable], orientation="h",
            marker_color="grey", name=auto_label,
            text=[f"{auto_assignable}"], textposition="outside",
        ))
        fig.add_trace(go.Bar(
            y=[human_label], x=[human_assignable], orientation="h",
            marker_color="grey", name=human_label,
            text=[f"{human_assignable}"], textposition="outside",
        ))
        fig.update_layout(
            title=f"{cat} ({len(cat_df)} tasks)",
            height=120, margin=dict(l=100, r=30, t=30, b=10),
            showlegend=False, barmode="group",
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor="white", paper_bgcolor="white",
        )

        children.append(html.Div([
            dcc.Graph(
                id={"type": "category-bar-chart", "category": cat},
                figure=fig, config={"displayModeBar": False},
                style={"height": "120px"},
            ),
            dcc.Store(id={"type": "category-override", "category": cat}, data="default"),
            dcc.Store(id={"type": "category-counts", "category": cat},
                      data={"human": human_assignable, "auto": auto_assignable,
                            "auto_label": auto_label}),
        ], style={
            "display": "inline-block", "width": "250px",
            "verticalAlign": "top", "margin": "5px",
            "border": "1px solid #eee", "borderRadius": "5px", "padding": "5px",
        }))

    return {"display": "block"}, children


@app.callback(
    Output({"type": "category-override", "category": dash.MATCH}, "data"),
    Output({"type": "category-bar-chart", "category": dash.MATCH}, "figure"),
    Input({"type": "category-bar-chart", "category": dash.MATCH}, "clickData"),
    State({"type": "category-override", "category": dash.MATCH}, "data"),
    State({"type": "category-counts", "category": dash.MATCH}, "data"),
    prevent_initial_call=True,
)
def toggle_category_selection(click_data, current_selection, counts):
    """Toggle category override when clicking a bar."""
    if not click_data:
        return dash.no_update, dash.no_update

    clicked_label = click_data["points"][0].get("y", None)
    if clicked_label is None:
        return dash.no_update, dash.no_update

    auto_label = counts.get("auto_label", "Autonomous")

    # Map clicked label to agent type
    if clicked_label == auto_label:
        clicked_type = "autonomous"
    elif clicked_label == "Human":
        clicked_type = "human"
    else:
        return dash.no_update, dash.no_update

    # Toggle: clicking same bar deselects
    new_selection = "default" if current_selection == clicked_type else clicked_type

    # Rebuild figure with updated colors
    human_color = "dodgerblue" if new_selection == "human" else "grey"
    auto_color = "dodgerblue" if new_selection == "autonomous" else "grey"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[auto_label], x=[counts.get("auto", 0)], orientation="h",
        marker_color=auto_color, name=auto_label,
        text=[f"{counts.get('auto', 0)}"], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        y=["Human"], x=[counts.get("human", 0)], orientation="h",
        marker_color=human_color, name="Human",
        text=[f"{counts.get('human', 0)}"], textposition="outside",
    ))
    fig.update_layout(
        height=120, margin=dict(l=100, r=30, t=30, b=10),
        showlegend=False, barmode="group",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor="white", paper_bgcolor="white",
    )

    return new_selection, fig


@app.callback(
    Output("category-overrides-store", "data"),
    Input({"type": "category-override", "category": dash.ALL}, "data"),
    State({"type": "category-override", "category": dash.ALL}, "id"),
    prevent_initial_call=True,
)
def collect_category_overrides(values, ids):
    """Collect all category overrides into a single store."""
    overrides = {}
    for id_dict, value in zip(ids, values):
        if value != "default":
            overrides[id_dict["category"]] = value
    return overrides


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
