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
import time
import pathlib

# Path to the bundled example CSV (works locally and in deployment)
_HERE = pathlib.Path(__file__).parent
_EXAMPLE_CANDIDATES = [_HERE / "V7" / "IA_V7.csv", _HERE / "IA_V7.csv"]
EXAMPLE_CSV = next((p for p in _EXAMPLE_CANDIDATES if p.exists()), None)

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=["assets/styles.css"])
server = app.server

# ─── Design Palette ────────────────────────────────────────────────────────────
# Edit these to re-skin all graphs / inline styles in one place.
BG           = "#f3f0eb"
SURFACE      = "#f7f7f5"
SURFACE      = "#f7f7f5"
INK          = "#1a1a1a"
INK_MUTED    = "#313131"
BORDER       = "#b0ada6"
ACCENT       = "#b42806"      # also used for highlights
DARK_GREY = "#1F1F1F"

# Semantic colours (mapped from the IA red/yellow/green/orange scheme)
PAL_RED      = "#b40606"
PAL_ORANGE   = "#d95e21"
PAL_YELLOW   = "#f5e74e"
PAL_GREEN    = "#1b7f41"
# Map used by style_table() and dot colours
COLOR_MAP = {
    "red":    PAL_RED,
    "orange": PAL_ORANGE,
    "yellow": PAL_YELLOW,
    "green":  PAL_GREEN,
}
# Lighter variants for pie charts / secondary usage
COLOR_MAP_LIGHT = {
    "red":    "#d44040",
    "orange": "#de7642",
    "yellow": "#f5e74e",
    "green":  "#1b7f41",
}
# Shade families for grouped bar charts (capacity charts)
COLOR_SHADES = {
    "green":  [PAL_GREEN, "#1b7f41", "#2a8f52", "#3a9f63", "#4abf74", "#4f8a5c"],
    "yellow": [PAL_YELLOW, "#f5e74e", "#f7f9a8", "#f9fbc4", "#fbe98a", "#c9b43e"],
    "orange": [PAL_ORANGE, "#d88a5c", "#e09c73", "#b45a2e", "#c47648", "#a04e22"],
}

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

    # 4b. Detect the procedure column (hierarchical grouping above tasks)
    procedure_column = None
    for candidate in ["Procedure", "procedure", "Phase", "Group", "Section"]:
        if candidate in df.columns:
            procedure_column = candidate
            break

    # 4c. Detect the category column (used for Automation Proportion)
    category_column = None
    color_set = set(color_columns)
    structural_set = {"Row", procedure_column or "Procedure", task_column or "Task"}
    for candidate in ["Category", "Type", "Classification", "Class", "Stage"]:
        if candidate in df.columns:
            category_column = candidate
            break
    if category_column is None:
        for col in df.columns:
            if col in structural_set or col in color_set:
                continue
            vals = df[col].dropna().astype(str)
            # Heuristic: few unique values relative to row count → categorical
            if 1 < vals.nunique() <= max(10, len(df) * 0.2):
                category_column = col
                break

    # 5. Identify metadata columns (everything not Row/structural/task/agent/category)
    structural = {"Row", procedure_column or "Procedure"}
    if task_column:
        structural.add(task_column)
    if category_column:
        structural.add(category_column)
    color_set = set(color_columns)
    metadata = [c for c in df.columns if c not in structural and c not in color_set]

    config = {
        "agents": agents,
        "alternatives": [
            {"name": f"Team Alternative {i+1}", **alt}
            for i, alt in enumerate(alternatives)
        ],
        "task_column": task_column or "Task",
        "procedure_column": procedure_column or "Procedure",
        "category_column": category_column,
        "color_columns": color_columns,
        "metadata_columns": metadata,
        "all_columns": list(df.columns),
    }
    return config


def build_config_from_manual(column_str, task_col="Task", procedure_col="Procedure", category_col=None):
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

    extra_cols = ["Observability", "Predictability", "Directability"]
    struct_cols = ["Row", procedure_col, task_col]
    if category_col and category_col not in struct_cols:
        struct_cols.insert(3, category_col)  # insert after task_col
    all_columns = struct_cols + cols + extra_cols

    config = {
        "agents": agents,
        "alternatives": [
            {"name": f"Team Alternative {i+1}", **alt}
            for i, alt in enumerate(alternatives)
        ],
        "task_column": task_col,
        "procedure_column": procedure_col,
        "category_column": category_col,
        "color_columns": cols,
        "metadata_columns": extra_cols,
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
        cat_col = config.get("category_column") or "Category"
        cat = str(row.get(cat_col, "") or "").strip()
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
    """Build DataTable column definitions with 3-level merged headers.

    Row 0 (top)  : "Activity Decomposition" | "Capacity Assessment" | "" (metadata)
    Row 1 (mid)  : ""                        | "Team Alternative N"  | ""
    Row 2 (bottom): actual column name
    """
    agent_set = set(get_agent_columns(config))

    # Map every agent column → its alternative label
    col_to_alt: dict[str, str] = {}
    for i, alt in enumerate(config.get("alternatives", []), 1):
        label = alt.get("name") or f"Team Alternative {i}"
        for col in alt.get("performers", []) + alt.get("supporters", []):
            col_to_alt[col] = label

    all_columns = config.get("all_columns", [])
    agent_indices = [i for i, c in enumerate(all_columns) if c in agent_set]
    first_agent_idx = min(agent_indices) if agent_indices else len(all_columns)

    # Columns that belong under the "Teaming Requirements" group header
    TEAMING_COLS = {"Observability", "Predictability", "Directability"}

    columns = []
    for idx, col in enumerate(all_columns):
        if col in agent_set:
            alt_label = col_to_alt.get(col, "Team Alternative ?")
            name = ["Capacity Assessment", alt_label, col]
        elif idx < first_agent_idx:
            name = ["Activity Decomposition", " ", col]
        elif col in TEAMING_COLS:
            name = ["Teaming Requirements", "  ", col]
        else:
            # Other metadata columns after the agent block (unique spacing
            # prevents accidental merging with neighbouring groups)
            name = ["  ", "   ", col]

        d = {"name": name, "id": col}
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
    """Conditional styles: color cells match their value.
    Uses filter_query so styles update live when the user edits a cell.
    """
    agent_cols = get_agent_columns(config)
    styles = []
    for col in agent_cols:
        for color in VALID_COLORS:
            for variant in [color, color.capitalize(), color.upper()]:
                styles.append({
                    "if": {
                        "filter_query": f"{{{col}}} = '{variant}'",
                        "column_id": col,
                    },
                    "backgroundColor": COLOR_MAP.get(color, color),
                    "color": COLOR_MAP.get(color, color),
                    "textAlign": "center",
                    "fontWeight": "bold",
                })
    return styles


def style_procedure_merge(df, config):
    """Visual pseudo-merge for the Procedure column.

    - Continuation rows (same Procedure as the row above): Procedure cell text
      is made transparent so the cell appears empty, mimicking a merged cell.
    - First row of each new Procedure group (after the very first): a top
      separator line is drawn across every column to visually delimit clusters.
    """
    proc_col = config.get("procedure_column", "Procedure")
    if proc_col not in df.columns or df.empty:
        return []

    all_cols = config.get("all_columns", [])
    styles = []
    df_reset = df.reset_index(drop=True)   # ensure 0-based positional index
    prev_proc = None

    for i in range(len(df_reset)):
        proc = str(df_reset.at[i, proc_col]).strip()
        if proc == prev_proc:
            # Continuation row — hide repeated Procedure label
            styles.append({
                "if": {"row_index": i, "column_id": proc_col},
                "color": "transparent",
                "borderTop": "1px solid #e8e8e8",   # keep a very faint line
            })
        else:
            # First row of a new group — draw a visible separator
            if i > 0:
                for col in all_cols:
                    styles.append({
                        "if": {"row_index": i, "column_id": col},
                        "borderTop": "2px solid #555555",
                    })
        prev_proc = proc

    return styles


# Columns always shown when present; everything else is hidden by default.
# Agent columns from the config are also always shown.
DEFAULT_VISIBLE_COLUMNS = {
    "Row", "Procedure", "Class", "Type", "Category", "Task",
    "Object", "Value",
    "Observability", "Predictability", "Directability",
    "TARS Performer Role", "TARS Supporter Role",
}


def build_hidden_columns(config):
    """Return a list of column IDs that should be hidden by default."""
    agent_set = set(get_agent_columns(config))
    visible = DEFAULT_VISIBLE_COLUMNS | agent_set
    return [c for c in config.get("all_columns", []) if c not in visible]


def build_data_table(df, config):
    """Create a new DataTable component from a DataFrame and config."""
    hidden = build_hidden_columns(config)
    return dash_table.DataTable(
        id="responsibility-table",
        columns=build_table_columns(config),
        data=df.to_dict("records"),
        editable=True,
        row_deletable=True,
        hidden_columns=hidden,
        merge_duplicate_headers=True,
        dropdown=build_dropdowns(config),
        style_data_conditional=style_table(df, config) + style_procedure_merge(df, config),
        style_cell={"textAlign": "left", "padding": "5px", "whiteSpace": "normal",
                    "fontFamily": "'Space Grotesk', 'Inter', sans-serif",
                    "backgroundColor": BG, "color": INK, "border": f"1px solid {BORDER}"},
        style_cell_conditional=build_borders(config),
        style_header={"fontWeight": "bold", "textAlign": "center"},
        style_header_conditional=[
            # Row 0 – top group labels
            {
                "if": {"header_index": 0},
                "backgroundColor": INK,
                "color": BG,
                "fontSize": "13px",
                "borderBottom": f"2px solid {BG}",
                "fontFamily": "'Space Grotesk', 'Inter', sans-serif",
                "letterSpacing": "0.04em",
                "textTransform": "uppercase",
            },
            # Row 1 – team alternative labels
            {
                "if": {"header_index": 1},
                "backgroundColor": "#3a3a3a",
                "color": BG,
                "fontSize": "12px",
                "borderBottom": f"2px solid {BG}",
                "fontFamily": "'Space Grotesk', 'Inter', sans-serif",
            },
            # Row 2 – column names (standard)
            {
                "if": {"header_index": 2},
                "backgroundColor": SURFACE,
                "color": INK,
                "fontSize": "12px",
                "fontFamily": "'Space Grotesk', 'Inter', sans-serif",
            },
        ],
        style_table={"overflowX": "auto", "border": f"2px solid {INK}"},
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

    # Build column list with visible ones bolded
    agent_set = set(get_agent_columns(config))
    visible = DEFAULT_VISIBLE_COLUMNS | agent_set
    all_cols = config.get("all_columns", [])
    col_spans = []
    for i, col in enumerate(all_cols):
        label = html.B(col) if col in visible else html.Span(col, style={"color": "var(--ink-muted)"})
        col_spans.append(label)
        if i < len(all_cols) - 1:
            col_spans.append(", ")

    return html.Div([
        html.P([html.B("Agents: "), agents_str]),
        html.P([html.B("Procedure column: "), config.get("procedure_column", "Procedure")]),
        html.P([html.B("Task column: "), config["task_column"]]),
        html.P([html.B("Category column: "), config.get("category_column") or "—  (none detected)"]),
        html.P([html.B("Columns: ")] + col_spans),
        html.Ul(alt_items),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

def build_workflow_figure_base(df, config, procedure=None, view_mode="full"):
    """
    Build the base workflow graph structure (without highlighting).

    Returns (fig, arrow_info) where arrow_info contains indices needed for
    the fast highlighting pass.
    """
    if config is None or df.empty:
        return go.Figure(), None

    agent_cols = get_agent_columns(config)
    performer_cols = set(get_performer_columns(config))
    task_col = config.get("task_column", "Task")

    # Performers-only view: filter to performer columns only
    if view_mode == "performers":
        agent_cols = [c for c in agent_cols if c in performer_cols]

    proc_col = config.get("procedure_column", "Procedure") if config else "Procedure"
    single_procedure = procedure is not None
    if single_procedure and proc_col in df.columns:
        df = df[df[proc_col] == procedure].copy()

    if df.empty:
        return go.Figure()

    df = df.reset_index(drop=True)
    df["task_idx"] = df.index
    tasks = df[task_col].tolist() if task_col in df.columns else [f"Task {i}" for i in range(len(df))]

    # ── Y-coordinate mapping ──────────────────────────────────────────────
    # When showing all procedures, insert a 1-unit gap between groups so
    # procedure-divider lines can be drawn in the extra space.
    # When filtered to a single procedure, y == task_idx (no gaps needed).
    GAP = 1.2          # extra y-units reserved for the divider between groups
    y_pos = []         # y_pos[i] = the plot y-coordinate for task i
    proc_dividers = [] # list of (y_between, proc_label) for separator lines
    procs = df[proc_col].tolist() if proc_col in df.columns else [""] * len(tasks)

    if single_procedure:
        y_pos = list(range(len(tasks)))
    else:
        y = 0.0
        # Seed the first procedure label above the first task
        if procs:
            proc_dividers.append((-0.6, procs[0]))
        for i, task_i in enumerate(tasks):
            if i > 0 and procs[i] != procs[i - 1]:
                # mid-point of the gap between groups
                proc_dividers.append((y + GAP / 2 - 0.5, procs[i]))
                y += GAP
            y_pos.append(y)
            y += 1.0

    agent_pos = {agent: i for i, agent in enumerate(agent_cols)}

    dots = []          # {task, y, agent, color}
    hover_lookup = {}  # (task_idx, col) -> hover text
    dashed_arrows = []

    # ── Per-task processing ───────────────────────────────────────────────
    for i, row in df.iterrows():
        task_idx = row["task_idx"]
        yp = y_pos[task_idx]
        task_label = wrap_text(str(row.get(task_col, "")))

        # Place dots + precompute hover text
        for col in agent_cols:
            if col in df.columns:
                val = str(row.get(col, "") or "").strip().lower()
                if val in VALID_COLORS:
                    dots.append({"task": task_idx, "y": yp, "agent": col, "color": val})
                    hover_parts = [
                        f"<b>Task:</b> {task_label}",
                        f"<b>Agent:</b> {col}",
                    ]
                    for meta in ["Observability", "Predictability", "Directability"]:
                        if meta in df.columns:
                            hover_parts.append(
                                f"<b>{meta}:</b><br>{wrap_text(str(row.get(meta, '')))}"
                            )
                    hover_lookup[(task_idx, col)] = "<br><br>".join(hover_parts)

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
                                "y": yp,
                            })

    # ── Solid arrows between consecutive tasks ────────────────────────────
    performers_by_task = {}
    for d in dots:
        if d["agent"] in performer_cols and d["color"] != "red":
            performers_by_task.setdefault(d["task"], set()).add(d["agent"])

    solid_arrows = []
    for i in range(1, len(df)):
        for pa in performers_by_task.get(i - 1, set()):
            for ca in performers_by_task.get(i, set()):
                solid_arrows.append({
                    "start_task": i - 1, "start_y": y_pos[i - 1], "start_agent": pa,
                    "end_task": i,   "end_y":   y_pos[i],     "end_agent": ca,
                })

    # ── Render figure ─────────────────────────────────────────────────────
    fig = go.Figure()

    # Procedure divider lines + labels (rendered first so dots sit on top)
    x_min = -0.5
    x_max = len(agent_cols) - 0.5

    # Build alt-label lookup and per-alt column groups (only cols present in agent_cols)
    col_to_alt: dict[str, str] = {}
    alt_col_groups: list[tuple[str, list[int]]] = []  # (label, [x indices])
    for i, alt in enumerate(config.get("alternatives", []), 1):
        label = alt.get("name") or f"Alt {i}"
        cols_in_alt = [c for c in alt.get("performers", []) + alt.get("supporters", [])
                       if c in agent_pos]
        for col in cols_in_alt:
            col_to_alt[col] = label
        if cols_in_alt:
            alt_col_groups.append((label, [agent_pos[c] for c in cols_in_alt]))

    # In performers-only view anchor the procedure label to the left edge so it
    # doesn't sit on top of the connector lines that run through the centre.
    if view_mode == "performers":
        label_x = x_min
        label_xanchor = "left"
    else:
        label_x = (x_min + x_max) / 2
        label_xanchor = "center"
    for div_y, proc_label in proc_dividers:
        fig.add_shape(
            type="line",
            x0=x_min, y0=div_y, x1=x_max, y1=div_y,
            xref="x", yref="y",
            line=dict(color=INK_MUTED, width=1.5, dash="dot"),
        )
        fig.add_annotation(
            x=label_x, y=div_y,
            xref="x", yref="y",
            text=f"<b>{proc_label}</b>",
            showarrow=False,
            xanchor=label_xanchor,
            font=dict(size=11, color=INK),
            bgcolor="rgba(221,217,210,0.85)",
            bordercolor=BORDER,
            borderwidth=1,
            borderpad=4,
        )
        # One centred alt label per alternative group, sitting above the divider line
        for alt_label, x_indices in alt_col_groups:
            centre_x = sum(x_indices) / len(x_indices)
            fig.add_annotation(
                x=centre_x, y=div_y - 0.08,
                xref="x", yref="y",
                text=f"<i>{alt_label}</i>",
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=9, color=INK_MUTED),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
            )
        # Per-column agent name annotations just above the divider line
        for col, x_idx in agent_pos.items():
            role = "Performer" if col in performer_cols else "Supporter"
            fig.add_annotation(
                x=x_idx, y=div_y - 0.22,
                xref="x", yref="y",
                text=f"<b>{col}</b><br><span style='font-size:8px'>{role}</span>",
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=10, color=INK),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
            )

    # One scatter trace per agent column with per-point colors — much fewer traces
    for col in agent_cols:
        col_dots = [d for d in dots if d["agent"] == col]
        if not col_dots:
            continue
        fig.add_trace(go.Scatter(
            x=[agent_pos[col]] * len(col_dots),
            y=[d["y"] for d in col_dots],
            mode="markers",
            marker=dict(
                size=20,
                color=[COLOR_MAP.get(d["color"], d["color"]) for d in col_dots],
                symbol="circle",
            ),
            showlegend=False,
            hoverinfo="text",
            hovertext=[hover_lookup.get((d["task"], col), "") for d in col_dots],
        ))

    # Group dashed arrows by task for vertical offset
    dashed_by_task = {}
    for arrow in dashed_arrows:
        if arrow["start_agent"] in agent_pos and arrow["end_agent"] in agent_pos:
            dashed_by_task.setdefault(arrow["task"], []).append(arrow)

    dashed_arrow_info = []
    for task, arrows in dashed_by_task.items():
        n = len(arrows)
        offsets = [0] * n if n == 1 else [
            -0.08 + 0.16 * i / (n - 1) for i in range(n)
        ]
        for arrow, offset in zip(arrows, offsets):
            fig.add_shape(
                type="line",
                x0=agent_pos[arrow["start_agent"]], y0=arrow["y"] + offset,
                x1=agent_pos[arrow["end_agent"]], y1=arrow["y"] + offset,
                line=dict(color=INK, width=2, dash="dot"),
            )
            dashed_arrow_info.append({"task": arrow["task"], "end_agent": arrow["end_agent"]})

    solid_arrow_info = []
    for arrow in solid_arrows:
        fig.add_annotation(
            x=agent_pos[arrow["end_agent"]], y=arrow["end_y"],
            ax=agent_pos[arrow["start_agent"]], ay=arrow["start_y"],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1,
            arrowwidth=2,
            arrowcolor=INK,
            opacity=0.9,
        )
        solid_arrow_info.append({
            "start_task": arrow["start_task"], "start_agent": arrow["start_agent"],
            "end_task": arrow["end_task"], "end_agent": arrow["end_agent"],
        })

    # Layout
    title_suffix = f" — {procedure}" if single_procedure else " (All Procedures)"
    y_labels = [wrap_text(str(t)) for t in tasks]

    # Estimated height: task rows + gap rows
    total_y_span = y_pos[-1] if y_pos else len(tasks)
    height = max(400, 100 + int(total_y_span * 80))

    fig.update_layout(
        title=f"Workflow Graph{title_suffix}",
        xaxis=dict(
            tickvals=list(agent_pos.values()),
            ticktext=list(agent_pos.keys()),
            title="Agent",
            showgrid=True, gridcolor=BORDER,
            range=[x_min, x_max],
        ),
        yaxis=dict(
            tickvals=y_pos,
            ticktext=y_labels,
            title="Task",
            range=[(y_pos[-1] + 0.5) if y_pos else len(tasks), -1.0],
            showgrid=False,
        ),
        height=height,
        margin=dict(l=250, r=50, t=50, b=50),
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font=dict(family="Space Grotesk, Inter, sans-serif", color=INK),
    )

    arrow_info = {
        "n_divider_shapes": len(proc_dividers),
        # 1 proc-label + 1 per alt group + 1 per agent column, all per divider
        "n_divider_annotations": len(proc_dividers) * (1 + len(alt_col_groups) + len(agent_cols)),
        "dashed_arrows": dashed_arrow_info,
        "solid_arrows": solid_arrow_info,
    }
    return fig, arrow_info


def apply_workflow_highlighting(fig_dict, arrow_info, df, config, procedure=None,
                                highlight_track=None, category_overrides=None):
    """Apply highlighting to a cached base workflow figure. Fast: only updates colors/widths."""
    if fig_dict is None:
        return go.Figure()

    fig = go.Figure(fig_dict)

    if arrow_info is None:
        return fig

    if (not highlight_track or highlight_track == "none") and not category_overrides:
        return fig

    if category_overrides is None:
        category_overrides = {}

    proc_col = config.get("procedure_column", "Procedure") if config else "Procedure"

    if procedure is not None and proc_col in df.columns:
        df = df[df[proc_col] == procedure].copy()

    df = df.reset_index(drop=True)
    df["task_idx"] = df.index

    should_hl_support = highlight_track in (
        "human_full_support", "agent_whenever_possible_full_support", "most_reliable",
    )

    # Build highlight set
    highlight_set = set()
    if highlight_track and highlight_track != "none":
        for _, row in df.iterrows():
            chosen = get_chosen_performer(row, config, highlight_track, category_overrides)
            if chosen:
                highlight_set.add((row["task_idx"], chosen))

    # Apply highlighting to dashed arrow shapes
    if fig.layout.shapes:
        shapes = list(fig.layout.shapes)
        offset = arrow_info.get("n_divider_shapes", 0)
        for i, info in enumerate(arrow_info.get("dashed_arrows", [])):
            idx = offset + i
            if idx < len(shapes):
                is_hl = should_hl_support and (info["task"], info["end_agent"]) in highlight_set
                shapes[idx].line.color = ACCENT if is_hl else INK
                shapes[idx].line.width = 4 if is_hl else 2
        fig.layout.shapes = shapes

    # Apply highlighting to solid arrow annotations
    if fig.layout.annotations:
        annotations = list(fig.layout.annotations)
        offset = arrow_info.get("n_divider_annotations", 0)
        for i, info in enumerate(arrow_info.get("solid_arrows", [])):
            idx = offset + i
            if idx < len(annotations):
                is_hl = bool(highlight_set) and \
                        (info["start_task"], info["start_agent"]) in highlight_set and \
                        (info["end_task"], info["end_agent"]) in highlight_set
                annotations[idx].arrowcolor = ACCENT if is_hl else INK
                annotations[idx].arrowwidth = 4 if is_hl else 2
        fig.layout.annotations = annotations

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
    color_shades = COLOR_SHADES

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
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(family="Space Grotesk, Inter, sans-serif", color=INK),
        showlegend=False,
    )
    return fig


def build_allocation_bar_chart(df, config):
    """Pie chart showing task type (allocation) distribution."""
    if config is None or df.empty:
        return go.Figure()

    performer_cols = get_performer_columns(config)
    supporter_cols = get_supporter_columns(config)

    single_independent = 0
    multiple_independent = 0
    single_interdependent = 0
    multiple_interdependent = 0

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
            if len(perfs) == 1:
                single_interdependent += 1
            else:
                multiple_interdependent += 1
        elif len(perfs) == 1:
            single_independent += 1
        elif len(perfs) > 1:
            multiple_independent += 1

    labels = [
        "Single Allocation Independent",
        "Multiple Allocation Independent",
        "Single Allocation Interdependent",
        "Multiple Allocation Interdependent",
    ]
    values = [single_independent, multiple_independent, single_interdependent, multiple_interdependent]

    # High-contrast fills:
    #  1. solid white
    #  2. white + faint grey dots
    #  3. white + bold ink diagonal lines
    #  4. dark grey solid (no pattern needed)
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker=dict(
            colors=["white", "white", "white", DARK_GREY],
            pattern=dict(
                shape=["", ".", "/", ""],
                fgcolor=[INK, BORDER, INK, DARK_GREY],
                size=[6, 6, 7, 6],
                solidity=[1.0, 0.35, 0.75, 1.0],
            ),
            line=dict(color=INK, width=2),
        ),
        textinfo="label+percent",
        textposition="outside",
        hovertemplate="%{label}<br>Count: %{value}<br>%{percent}<extra></extra>",
        hole=0.3,
    ))
    fig.update_layout(
        title="Task Type Distribution",
        height=520,
        paper_bgcolor=BG,
        font=dict(family="Space Grotesk, Inter, sans-serif", color=INK),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_autonomy_bar_chart(df, config):
    """Pie charts showing agent autonomy (task continuity) — one pie per performer."""
    if config is None or df.empty:
        return go.Figure()

    from plotly.subplots import make_subplots

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

    active_cols = [c for c in performer_cols if (agent_autonomy[c]["autonomous"] + agent_autonomy[c]["non_autonomous"]) > 0]
    if not active_cols:
        return go.Figure()

    n = len(active_cols)
    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{"type": "pie"}] * n],
        subplot_titles=[c.rstrip("*") for c in active_cols],
    )

    for i, col in enumerate(active_cols, 1):
        auto = agent_autonomy[col]["autonomous"]
        non_auto = agent_autonomy[col]["non_autonomous"]
        fig.add_trace(go.Pie(
            labels=["Autonomous", "Non-Autonomous"],
            values=[auto, non_auto],
            marker=dict(
                colors=["white", "#555250"],   # solid white / dark grey solid
                pattern=dict(
                    shape=["", ""],
                    fgcolor=[INK, "#555250"],
                    size=[6, 6],
                    solidity=1.0,
                ),
                line=dict(color=INK, width=2),
            ),
            textinfo="label+percent",
            textposition="outside",
            hovertemplate="%{label}<br>Count: %{value}<br>%{percent}<extra></extra>",
            hole=0.3,
            showlegend=(i == 1),
        ), row=1, col=i)

    fig.update_layout(
        title="Agent Autonomy: Task Continuity",
        height=400,
        paper_bgcolor=BG,
        font=dict(family="Space Grotesk, Inter, sans-serif", color=INK),
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
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
        ("Performer Green", perf_green, PAL_GREEN),
        ("Performer Yellow", perf_yellow, PAL_YELLOW),
        ("Performer Orange", perf_orange, PAL_ORANGE),
        ("Supporter Green", sup_green, COLOR_MAP_LIGHT["green"]),
        ("Supporter Yellow", sup_yellow, COLOR_MAP_LIGHT["yellow"]),
        ("Supporter Orange", sup_orange, COLOR_MAP_LIGHT["orange"]),
    ]:
        fig.add_trace(go.Bar(name=label, x=[label], y=[count], marker_color=color, showlegend=False))

    fig.update_layout(
        title="Most Reliable Path: Performer and Supporter Capacities",
        xaxis_title="Role and Capacity", yaxis_title="Number of Tasks",
        barmode="group", bargap=0.15,
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(family="Space Grotesk, Inter, sans-serif", color=INK),
        showlegend=False,
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
    fig.add_trace(go.Bar(name="Green", x=["Green"], y=[perf_green], marker_color=PAL_GREEN, showlegend=False))
    fig.add_trace(go.Bar(name="Yellow", x=["Yellow"], y=[perf_yellow], marker_color=PAL_YELLOW, showlegend=False))
    fig.add_trace(go.Bar(name="Orange", x=["Orange"], y=[perf_orange], marker_color=PAL_ORANGE, showlegend=False))

    fig.update_layout(
        title="Human-Only Baseline: Human Performer Capacities",
        xaxis_title="Capacity Level", yaxis_title="Number of Tasks",
        barmode="group", bargap=0.15,
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(family="Space Grotesk, Inter, sans-serif", color=INK),
        showlegend=False,
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
    cat_col = config.get("category_column") or "Category"
    if config is None or df.empty or cat_col not in df.columns:
        return None, {}

    if not highlight_track or highlight_track == "none":
        return None, {}

    agent_types = {a["name"]: a["type"] for a in config["agents"]}
    categories = sorted(df[cat_col].dropna().unique())

    if not categories:
        return None, {}

    K = len(categories)
    category_scores = {}
    is_full_support = highlight_track in (
        "human_full_support", "agent_whenever_possible_full_support", "most_reliable",
    )

    for cat in categories:
        cat_df = df[df[cat_col] == cat]
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
    dcc.Store(id="base-figure-store", data=None),
    dcc.Store(id="arrow-indices-store", data=None),

    # ═══════════════════════════════════════════════════════════════════════
    # STICKY NAVIGATION HEADER
    # ═══════════════════════════════════════════════════════════════════════
    html.Nav(className="sticky-nav", children=[
        html.A("Interdependence Analysis Dashboard", href="#", className="nav-brand"),
        html.Div(id="nav-links", className="nav-links", style=HIDE, children=[
            html.A("Table",      href="#table-anchor",      className="nav-link"),
            html.A("Workflow",   href="#workflow-anchor",   className="nav-link"),
            html.A("Statistics", href="#statistics-anchor", className="nav-link"),
        ]),
    ]),

    # ═══════════════════════════════════════════════════════════════════════
    # SETUP SECTION  [01]
    # ═══════════════════════════════════════════════════════════════════════
    html.Div(id="setup-section", className="section", children=[
        html.Div("[ 01 ]", className="section-number"),
        html.H2("Team Setup", className="section-title"),
        html.P(
            "Upload a CSV or manually define your Human-Autonomy Team configuration.",
            className="section-subtitle",
        ),

        # ── Option 1: CSV Upload ──
        html.Div(className="card", children=[
            html.Div("Option 1 — Load from CSV", className="card-header"),
            html.P(
                "Upload a CSV and the team structure will be auto-detected from "
                "columns containing color values (red/yellow/green/orange). "
                "Columns ending with * are treated as performer roles.",
                style={"fontSize": "14px"},
            ),
            html.P([
                html.B("Expected column structure: "),
                "Row | Procedure | Task | [optional: Category] | Agent columns… | Metadata columns",
            ], style={"fontSize": "13px", "fontFamily": "var(--font-mono)"}),
            html.P([
                "The Procedure column groups Tasks hierarchically (Procedure → Task). "
                "The optional Category column (e.g. Observe/Orient/Decide/Act from a OODA decomposition) "
                "is used for Automation Proportion computation. Any column with few unique string values "
                "not matching colors will be auto-detected as the Category column.",
            ], style={"fontSize": "13px"}),
            dcc.Upload(
                id="setup-upload",
                children=html.Div([
                    "Drag and Drop or ",
                    html.A("Select a CSV File", style={"color": ACCENT, "cursor": "pointer", "fontWeight": "600"}),
                ]),
                className="upload-zone",
                style={
                    "width": "100%", "lineHeight": "60px",
                    "textAlign": "center", "margin": "10px 0",
                },
                multiple=False,
            ),
            html.Div([
                html.Button(
                    "Load Example (IA_V7.csv)",
                    id="load-example-button",
                    n_clicks=0,
                    style={"marginTop": "8px", "fontSize": "13px"},
                ),
                html.Span(
                    " — load a pre-built example to explore the dashboard",
                    style={"fontSize": "12px", "color": INK_MUTED, "marginLeft": "8px"},
                ),
            ]),
            html.Div(id="upload-status", style={"marginTop": "10px", "fontStyle": "italic"}),
        ]),

        # ── Option 2: Manual Setup ──
        html.Div(className="card-alt", children=[
            html.Div("Option 2 — Define Team Manually", className="card-header"),
            html.P(
                "The first task of the IA requires defining team alternatives, first list all of the agents "
                "in the team, then check for alternatives in agent's roles (e.g. performer vs supporter). "
                "Enter the agent role columns in order. Use * for performer columns. "
                "Group them as: Alt1-performers, Alt1-supporters, Alt2-performers, Alt2-supporters, …",
                style={"fontSize": "14px"},
            ),
            html.Div([
                html.Label("Agent columns (comma-separated):", style={"fontWeight": "bold"}),
                dcc.Input(
                    id="manual-columns-input",
                    value="Human*, Robot, Robot*, Human",
                    style={"width": "100%", "marginBottom": "10px", "padding": "8px"},
                ),
            ]),

            # ── Procedure column ──
            html.Div([
                html.Label("Higher-level activity column name:", style={"fontWeight": "bold", "marginRight": "10px"}),
                dcc.Input(
                    id="manual-procedure-col-input",
                    value="Procedure",
                    style={"width": "200px", "padding": "8px"},
                ),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
            html.P(
                "Define the column name that groups several Tasks and represents one level of hierarchical "
                "decomposition of the joint activity — the analysis uses a two-level hierarchy: "
                "(e.g. Procedure → Task.) This column typically corresponds to high-level mission phases "
                "or activity clusters.",
                style={"fontSize": "13px", "marginTop": "2px", "marginBottom": "14px"},
            ),

            # ── Task column ──
            html.Div([
                html.Label("Lower-level activity column name:", style={"fontWeight": "bold", "marginRight": "10px"}),
                dcc.Input(
                    id="manual-task-col-input",
                    value="Task",
                    style={"width": "200px", "padding": "8px"},
                ),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
            html.P([
                "Represents the terminal nodes of the activity decomposition. We recommend decomposing into "
                "required capacity in terms of information-processing stages: ",
                html.B("Sense → Interpret → Decide → Act"),
            ], style={"fontSize": "13px", "marginTop": "2px", "marginBottom": "14px"}),

            # ── Category column ──
            html.Div([
                html.Label("Category column name (optional):", style={"fontWeight": "bold", "marginRight": "10px"}),
                dcc.Input(
                    id="manual-category-col-input",
                    value="Category",
                    placeholder="e.g. Category, Type, Stage",
                    style={"width": "220px", "padding": "8px"},
                ),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
            html.P(
                "The Category column assigns each task to a named group. It is used as the grouping variable "
                "for Automation Proportion computation: one proportion pₖ is computed per category, then "
                "averaged to produce the overall index P. Leave blank if not applicable.",
                style={"fontSize": "13px", "marginTop": "2px", "marginBottom": "14px"},
            ),

            # Preset buttons
            html.Div([
                html.Label("Presets: ", style={"fontWeight": "bold", "marginRight": "10px"}),
                html.Button("Human + Robot", id="preset-2agent", n_clicks=0,
                            style={"marginRight": "10px"}),
                html.Button("Human + UGV + UAV", id="preset-3agent", n_clicks=0,
                            style={"marginRight": "10px"}),
            ], style={"marginBottom": "15px"}),
            html.Button(
                "Create Team", id="create-team-button", n_clicks=0,
                className="btn-primary",
                style={"marginTop": "10px"},
            ),
        ]),
        # ── References ──
        html.Div(style={"marginTop": "3rem", "borderTop": f"1px solid {BORDER}", "paddingTop": "1.5rem"}, children=[
            html.P([
                "If you want to know more about interdependence analysis for building effective Human-Autonomy Teams, "
                "check these publications and these two videos by Matthew Johnson: ",
                html.A("Video 1", href="https://www.youtube.com/watch?v=BnuTBMWnf6M",
                       target="_blank", style={"color": ACCENT, "fontWeight": "600"}),
                " · ",
                html.A("Video 2", href="https://www.youtube.com/watch?v=M2UgTNPjHyM",
                       target="_blank", style={"color": ACCENT, "fontWeight": "600"}),
                ".",
            ], style={"fontSize": "14px", "marginBottom": "1.2rem", "lineHeight": "1.7"}),
            html.H4("References", style={"textTransform": "uppercase", "letterSpacing": "0.05em",
                                         "fontSize": "13px", "color": INK_MUTED, "marginBottom": "1rem"}),
            html.Ol(style={"fontSize": "13px", "lineHeight": "1.8", "paddingLeft": "1.2rem",
                           "color": INK, "fontFamily": "var(--font-body)"}, children=[
                html.Li("Johnson, M. (2014). Coactive Design: Designing Support for Interdependence in Human-Robot Teamwork."),
                html.Li([
                    "Johnson, M., Bradshaw, J., & Feltovich, P. J. (2017). Tomorrow's Human–Machine Design Tools: From Levels of Automation to Interdependencies. ",
                    html.Em("Journal of Cognitive Engineering and Decision Making"), ", 12, 155534341773646. ",
                    html.A("https://doi.org/10.1177/1555343417736462",
                           href="https://doi.org/10.1177/1555343417736462", target="_blank",
                           style={"color": ACCENT}),
                ]),
                html.Li([
                    "Johnson, M., Bradshaw, J., Feltovich, P. J., Hoffman, R., Jonker, C., Riemsdijk, B., & Sierhuis, M. (2011). Beyond Cooperative Robotics: The Central Role of Interdependence in Coactive Design. ",
                    html.Em("IEEE Intelligent Systems"), ", 26, 81–88. ",
                    html.A("https://doi.org/10.1109/MIS.2011.47",
                           href="https://doi.org/10.1109/MIS.2011.47", target="_blank",
                           style={"color": ACCENT}),
                ]),
                html.Li([
                    "Johnson, M., & Bradshaw, J. M. (2021). How Interdependence Explains the World of Teamwork. In W. F. Lawless, J. Llinas, D. A. Sofge, & R. Mittu (Eds.), ",
                    html.Em("Engineering Artificially Intelligent Systems: A Systems Engineering Approach to Realizing Synergistic Capabilities"),
                    " (pp. 122–146). Springer International Publishing. ",
                    html.A("https://doi.org/10.1007/978-3-030-89385-9_8",
                           href="https://doi.org/10.1007/978-3-030-89385-9_8", target="_blank",
                           style={"color": ACCENT}),
                ]),
                html.Li([
                    "Johnson, M., Bradshaw, J. M., Feltovich, P. J., Jonker, C. M., van Riemsdijk, M. B., & Sierhuis, M. (2014). Coactive design: Designing support for interdependence in joint activity. ",
                    html.Em("J. Hum.-Robot Interact."), ", 3(1), 43–69. ",
                    html.A("https://doi.org/10.5898/JHRI.3.1.Johnson",
                           href="https://doi.org/10.5898/JHRI.3.1.Johnson", target="_blank",
                           style={"color": ACCENT}),
                ]),
                html.Li([
                    "Johnson, M., Vignati, M., & Duran, D. (2018). Understanding Human-Autonomy Teaming through Interdependence Analysis. ",
                    html.A("ihmc",
                           href="https://www.ihmc.us/wp-content/uploads/2019/01/180907-HAT-Interdependence-Analysis.pdf",
                           target="_blank", style={"color": ACCENT}),
                ]),
            ]),
        ]),
    ]),

    # ═══════════════════════════════════════════════════════════════════════
    # CONFIG SUMMARY (shown after setup)
    # ═══════════════════════════════════════════════════════════════════════
    html.Div(id="config-summary", style=HIDE, children=[
        html.Div(className="section", style={"paddingBottom": "1rem"}, children=[
            html.Div("[ 01 ]", className="section-number"),
            html.Div([
                html.H4("Current Team Configuration", style={"display": "inline-block", "margin": "0"}),
                html.Button("Change Team", id="reset-config-button", n_clicks=0,
                            style={"marginLeft": "20px"}),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div(id="config-details", style={"marginTop": "1rem"}),
        ]),
    ]),

    # ═══════════════════════════════════════════════════════════════════════
    # ANALYSIS SECTION
    # ═══════════════════════════════════════════════════════════════════════
    html.Div(id="analysis-section", style=HIDE, children=[

        # ── [02] Table ──
        html.Div(id="table-anchor", className="section", children=[
            html.Div("[ 02 ]", className="section-number"),
            html.H2("Interdependence Analysis Table", className="section-title"),

            html.Div(id="table-wrapper"),

            # Action buttons
            html.Div(style={"marginTop": "1rem"}, children=[
                html.Div([
                    html.Div([
                        dcc.Upload(
                            id="upload-data",
                            children=html.Button("Load Table", id="load-button", n_clicks=0),
                            multiple=False,
                            style={"display": "inline-block", "marginRight": "10px"},
                        ),
                        html.Button("Add Row", id="add-row-button", n_clicks=0),
                        html.Button("Copy Cell Down", id="copy-down-button", n_clicks=0),
                    ], style={"display": "flex", "gap": "10px"}),
                    html.Div([
                        html.Button("Save Table", id="save-button", n_clicks=0),
                        dcc.Download(id="download-csv"),
                    ], style={"marginLeft": "auto"}),
                ], style={"display": "flex", "width": "100%"}),
                html.Div(id="save-confirmation", style={"marginTop": "10px", "fontStyle": "italic"}),
            ]),
        ]),

        # ── [03] Workflow Graph ──
        html.Div(id="workflow-anchor", className="section", children=[
            html.Div("[ 03 ]", className="section-number"),
            html.H2("Workflow Graph", className="section-title"),

            # Procedure dropdown + View selector
            html.Div([
                dcc.Dropdown(
                    id="procedure-dropdown",
                    options=[],
                    value=None,
                    placeholder="Select a procedure to filter the graph…",
                    clearable=True,
                    style={"width": "320px", "marginRight": "30px"},
                ),
                dcc.RadioItems(
                    id="view-selector",
                    options=[
                        {"label": "Full View (All Agents)", "value": "full"},
                        {"label": "Performers Only", "value": "performers"},
                    ],
                    value="full",
                    labelStyle={"display": "inline-block", "marginRight": "20px"},
                    style={"display": "flex", "alignItems": "center"},
                ),
            ], style={"display": "flex", "alignItems": "center", "marginTop": "12px"}),

            # Allocation pattern selector
            dcc.RadioItems(
                id="highlight-selector",
                options=[
                    {"label": "No highlight", "value": "none"},
                    {"label": "Alt 1 performer — independent", "value": "human_baseline"},
                    {"label": "Alt 1 performer — interdependent", "value": "human_full_support"},
                    {"label": "Alt 2 performer — independent", "value": "agent_whenever_possible"},
                    {"label": "Alt 2 performer — interdependent", "value": "agent_whenever_possible_full_support"},
                    {"label": "Path of highest reliability", "value": "most_reliable"},
                ],
                value="none",
                labelStyle={"display": "inline-block", "marginRight": "20px"},
                style={"marginTop": "10px"},
            ),

            # Category Overrides
            html.Div(id="category-overrides-section", style={"display": "none"}, children=[
                html.H3("Category Overrides", style={"marginTop": "30px", "textTransform": "uppercase",
                                                      "letterSpacing": "0.04em"}),
                html.P(
                    "Click a bar to force a specific agent type for that category. "
                    "Click again to reset to default strategy.",
                    style={"fontSize": "14px"},
                ),
                html.Div(id="category-override-warning", style={"fontSize": "13px", "color": ACCENT, "marginTop": "6px"}),
                html.Div(id="category-overrides-container"),
            ]),

            # Automation Proportion Summary
            html.Div(id="automation-proportion-box", children=[
                html.Div([
                    html.Span("Automation Proportion: ",
                               style={"fontSize": "16px", "fontWeight": "bold"}),
                    html.Span(id="ap-summary-value", children="--",
                               style={"fontSize": "20px", "fontWeight": "bold", "color": ACCENT}),
                ], className="ap-box"),
                html.Div(id="ap-detail", style={"fontSize": "13px", "marginTop": "6px"}),
            ], style={"marginTop": "20px", "display": "none"}),
            html.Div(id="automation-proportion-results"),
            dcc.Graph(id="interdependence-graph", config={"displayModeBar": False}),

            # Team alternative labels (dynamic)
            html.Div(id="alt-labels"),
        ]),

        # ── [04] Statistics ──
        html.Div(id="statistics-anchor", className="section", children=[
            html.Div("[ 04 ]", className="section-number"),
            html.H2("Statistics", className="section-title"),
            html.Div([
                dcc.Graph(id="allocation-type-bar-chart",  config={"displayModeBar": False}),
                dcc.Graph(id="agent-autonomy-bar-chart",   config={"displayModeBar": False}),
            ]),
            html.Details([
                html.Summary("Capacity Assessment", style={
                    "fontSize": "1.1rem", "fontWeight": "bold", "cursor": "pointer",
                    "padding": "0.75rem 0", "userSelect": "none",
                    "textTransform": "uppercase", "letterSpacing": "0.04em",
                }),
                dcc.Graph(id="capacity-bar-chart",         config={"displayModeBar": False}),
                dcc.Graph(id="most-reliable-bar-chart",    config={"displayModeBar": False}),
                dcc.Graph(id="human-baseline-bar-chart",   config={"displayModeBar": False}),
            ]),
        ]),
    ]),

    # Footer
    html.Footer(
        "© Benjamin R. Berton 2025 Polytechnique Montreal",
        className="site-footer",
    ),
], style={
    "fontFamily": "var(--font-body, 'Space Grotesk', 'Inter', sans-serif)",
    "backgroundColor": BG,
    "color": INK,
    "margin": "0",
    "padding": "0",
})


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Preset buttons ────────────────────────────────────────────────────────────
@app.callback(
    Output("manual-columns-input", "value"),
    Output("manual-task-col-input", "value"),
    Output("manual-procedure-col-input", "value"),
    Input("preset-2agent", "n_clicks"),
    Input("preset-3agent", "n_clicks"),
    prevent_initial_call=True,
)
def apply_preset(n2, n3):
    ctx = callback_context
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    if btn == "preset-2agent":
        return "Human*, Robot, Robot*, Human", "Task", "Procedure"
    elif btn == "preset-3agent":
        return "Human*, UGV, UAV, UGV*, UAV*, Human", "Task", "Procedure"
    return dash.no_update, dash.no_update, dash.no_update


# ── Nav-links visibility ─────────────────────────────────────────────────────
@app.callback(
    Output("nav-links", "style"),
    Input("team-config-store", "data"),
)
def toggle_nav_links(config):
    return SHOW if config else HIDE


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
    Input("load-example-button", "n_clicks"),
    State("setup-upload", "filename"),
    State("manual-columns-input", "value"),
    State("manual-task-col-input", "value"),
    State("manual-procedure-col-input", "value"),
    State("manual-category-col-input", "value"),
    prevent_initial_call=True,
)
def handle_setup(upload_contents, create_clicks, reset_clicks, example_clicks,
                 upload_filename, manual_columns, manual_task_col,
                 manual_procedure_col, manual_category_col):
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    no = dash.no_update

    if triggered == "reset-config-button":
        return None, None, HIDE, SHOW, HIDE, None, [], ""

    if triggered == "load-example-button":
        if EXAMPLE_CSV is None:
            return no, no, no, no, no, no, no, "⚠️ Example file not found."
        try:
            df = pd.read_csv(EXAMPLE_CSV)
        except Exception as e:
            return no, no, no, no, no, no, no, f"⚠️ Error reading example: {e}"
        config = detect_team_config(df)
        if config is None:
            return no, no, no, no, no, no, no, "⚠️ Could not detect team structure in example."
        if "Row" not in df.columns:
            df.insert(0, "Row", range(1, len(df) + 1))
            config["all_columns"] = ["Row"] + [c for c in config["all_columns"] if c != "Row"]
        proc_col = config.get("procedure_column", "Procedure")
        proc_options = []
        if proc_col in df.columns:
            proc_options = [{"label": p, "value": p} for p in df[proc_col].dropna().unique()]
        table = build_data_table(df, config)
        summary = config_summary_html(config)
        return config, table, SHOW, HIDE, SHOW, summary, proc_options, "✅ Loaded example: IA_V7.csv"

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
        proc_col = config.get("procedure_column", "Procedure")
        proc_options = []
        if proc_col in df.columns:
            proc_options = [{"label": p, "value": p} for p in df[proc_col].dropna().unique()]

        table = build_data_table(df, config)
        summary = config_summary_html(config)
        return config, table, SHOW, HIDE, SHOW, summary, proc_options, f"✅ Loaded {upload_filename}"

    if triggered == "create-team-button":
        config = build_config_from_manual(
            manual_columns or "",
            task_col=manual_task_col or "Task",
            procedure_col=manual_procedure_col or "Procedure",
            category_col=manual_category_col.strip() or None if manual_category_col else None,
        )
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

        proc_col = new_config.get("procedure_column", "Procedure")
        proc_options = []
        if proc_col in df.columns:
            proc_options = [{"label": p, "value": p} for p in df[proc_col].dropna().unique()]

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
        proc_col = config.get("procedure_column", "Procedure")
        proc_options = []
        if proc_col in df.columns:
            proc_options = [{"label": p, "value": p} for p in df[proc_col].dropna().unique()]
        table = build_data_table(df, config)
        return table, "", None, proc_options, no, no, no, no, no

    return no, "", None, no, no, no, no, no, no


# ── Graph callbacks (two-stage: base figure + highlighting) ───────────────────
@app.callback(
    Output("base-figure-store", "data"),
    Output("arrow-indices-store", "data"),
    Output("alt-labels", "children"),
    Input("procedure-dropdown", "value"),
    Input("view-selector", "value"),
    Input("responsibility-table", "data"),
    State("team-config-store", "data"),
)
def build_base_figure(procedure, view_mode, data, config):
    """Build the base figure structure (without highlighting).
    Only runs when procedure, view mode, or data changes."""
    if not data or not config:
        return None, None, None

    df = pd.DataFrame(data)
    if df.empty:
        return None, None, None

    fig, arrow_info = build_workflow_figure_base(
        df, config, procedure=procedure, view_mode=view_mode or "full",
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

    return fig.to_dict(), arrow_info, alt_label_div


@app.callback(
    Output("interdependence-graph", "figure"),
    Input("base-figure-store", "data"),
    Input("arrow-indices-store", "data"),
    Input("highlight-selector", "value"),
    Input("category-overrides-store", "data"),
    Input("procedure-dropdown", "value"),
    State("responsibility-table", "data"),
    State("team-config-store", "data"),
)
def apply_highlighting_callback(base_fig_dict, arrow_info, highlight_track,
                                 category_overrides, procedure, data, config):
    """Apply highlighting to the cached base figure. Fast: only updates colors/widths."""
    if not base_fig_dict or not data or not config:
        return go.Figure()

    df = pd.DataFrame(data)
    ht = None if highlight_track == "none" else highlight_track

    return apply_workflow_highlighting(
        base_fig_dict, arrow_info, df, config,
        procedure=procedure, highlight_track=ht,
        category_overrides=category_overrides or {},
    )


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
    proc_col = config.get("procedure_column", "Procedure")
    if procedure and proc_col in df.columns:
        df = df[df[proc_col] == procedure]

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
    proc_col = config.get("procedure_column", "Procedure")
    if procedure and proc_col in df.columns:
        df = df[df[proc_col] == procedure]

    cat_col = config.get("category_column") or "Category"
    if cat_col not in df.columns or not highlight_track or highlight_track == "none":
        return {"display": "none"}, "--", {}, "", None

    P, cat_scores = compute_automation_proportion_data(
        df, config, highlight_track, category_overrides or {},
    )

    if P is None:
        return {"display": "none"}, "--", {}, "", None

    # Color code: accent if P > 0.5, green if P < 0.5, yellow if P ≈ 0.5
    if P > 0.55:
        color = ACCENT
    elif P < 0.45:
        color = PAL_GREEN
    else:
        color = PAL_YELLOW

    val_style = {"fontSize": "20px", "fontWeight": "bold", "color": color}
    box_style = {"textAlign": "center", "marginTop": "10px"}

    # Category detail
    detail_parts = [f"{cat}: {score:.2f}" for cat, score in sorted(cat_scores.items())]
    detail_text = " | ".join(detail_parts) if detail_parts else ""

    # Extract performer names for formula explanation
    alts = config.get("alternatives", [])
    def _perf_name(alt):
        perfs = alt.get("performers", [])
        return perfs[0].rstrip("*") if perfs else "Agent"
    alt1_name = _perf_name(alts[0]) if len(alts) > 0 else "Alt 1 performer"
    alt2_name = _perf_name(alts[1]) if len(alts) > 1 else "Alt 2 performer"

    # LaTeX formula display
    formula = html.Div([
        html.P([
            html.B("Formula: "),
            f"P = (1/K) × Σ pₖ = (1/{len(cat_scores)}) × "
            f"{sum(cat_scores.values()):.2f} = {P:.3f}",
        ], style={"fontSize": "13px", "marginTop": "5px"}),
        html.P([
            html.B("Where: "),
            f"w(t) = 0.0 ({alt1_name} independent), "
            f"0.5 ({alt1_name} interdependent — supported by autonomous), "
            f"0.75 ({alt1_name} as supporter), "
            f"1.0 ({alt2_name} independent)",
        ], style={"fontSize": "12px"}),
    ])

    return box_style, f"{P:.3f}", val_style, detail_text, formula


# ── Dynamic highlight-selector labels ─────────────────────────────────────────
@app.callback(
    Output("highlight-selector", "options"),
    Input("team-config-store", "data"),
)
def update_highlight_options(config):
    def _perf_name(alt):
        perfs = alt.get("performers", [])
        return perfs[0].rstrip("*") if perfs else "Agent"
    if not config or not config.get("alternatives"):
        a1, a2 = "Alt 1 performer", "Alt 2 performer"
    else:
        alts = config["alternatives"]
        a1 = _perf_name(alts[0]) if len(alts) > 0 else "Alt 1 performer"
        a2 = _perf_name(alts[1]) if len(alts) > 1 else "Alt 2 performer"
    return [
        {"label": "No highlight", "value": "none"},
        {"label": f"{a1} — independent", "value": "human_baseline"},
        {"label": f"{a1} — interdependent", "value": "human_full_support"},
        {"label": f"{a2} — independent", "value": "agent_whenever_possible"},
        {"label": f"{a2} — interdependent", "value": "agent_whenever_possible_full_support"},
        {"label": "Path of highest reliability", "value": "most_reliable"},
    ]


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
    cat_col = config.get("category_column") or "Category"
    if cat_col not in df.columns:
        return {"display": "none"}, None

    agent_types = {a["name"]: a["type"] for a in config["agents"]}
    performer_cols = get_performer_columns(config)
    categories = sorted(df[cat_col].dropna().unique())

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
        cat_df = df[df[cat_col] == cat]

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
        max_count = max(auto_assignable, human_assignable, 1)
        _ts = time.time()
        fig.add_trace(go.Bar(
            y=[auto_label], x=[auto_assignable], orientation="h",
            marker_color="#555250", name=auto_label,
            text=[f"{auto_assignable}"], textposition="outside",
            cliponaxis=False, customdata=[_ts],
        ))
        fig.add_trace(go.Bar(
            y=[human_label], x=[human_assignable], orientation="h",
            marker_color="#555250", name=human_label,
            text=[f"{human_assignable}"], textposition="outside",
            cliponaxis=False, customdata=[_ts],
        ))
        fig.update_layout(
            title=f"{cat} ({len(cat_df)} tasks)",
            height=100, margin=dict(l=80, r=40, t=25, b=5),
            showlegend=False, barmode="group",
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                       range=[0, max_count * 1.5]),
            yaxis=dict(showgrid=False),
            plot_bgcolor=BG, paper_bgcolor=BG,
            font=dict(family="Space Grotesk, Inter, sans-serif", color=INK),
        )

        children.append(html.Div([
            dcc.Graph(
                id={"type": "category-bar-chart", "category": cat},
                figure=fig, config={"displayModeBar": False},
                style={"height": "100px"},
            ),
            dcc.Store(id={"type": "category-override", "category": cat}, data="default"),
            dcc.Store(id={"type": "category-counts", "category": cat},
                      data={"human": human_assignable, "auto": auto_assignable,
                            "auto_label": auto_label,
                            "title": f"{cat} ({len(cat_df)} tasks)"}),
        ], style={
            "display": "inline-block", "width": "220px",
            "verticalAlign": "top", "margin": "4px",
            "border": f"1px solid {BORDER}", "padding": "3px",
            "backgroundColor": SURFACE,
        }))

    return {"display": "block"}, children


@app.callback(
    Output({"type": "category-override", "category": dash.MATCH}, "data"),
    Output({"type": "category-bar-chart", "category": dash.MATCH}, "figure"),
    Input({"type": "category-bar-chart", "category": dash.MATCH}, "clickData"),
    State({"type": "category-override", "category": dash.MATCH}, "data"),
    State({"type": "category-counts", "category": dash.MATCH}, "data"),
    State("highlight-selector", "value"),
    prevent_initial_call=True,
)
def toggle_category_selection(click_data, current_selection, counts, highlight_value):
    """Toggle category override when clicking a bar."""
    if not click_data:
        return dash.no_update, dash.no_update

    if not highlight_value or highlight_value == "none":
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
    human_color = ACCENT if new_selection == "human" else "#555250"
    auto_color = ACCENT if new_selection == "autonomous" else "#555250"

    auto_count = counts.get("auto", 0)
    human_count = counts.get("human", 0)
    max_count = max(auto_count, human_count, 1)
    _ts = time.time()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[auto_label], x=[auto_count], orientation="h",
        marker_color=auto_color, name=auto_label,
        text=[f"{auto_count}"], textposition="outside",
        cliponaxis=False, customdata=[_ts],
    ))
    fig.add_trace(go.Bar(
        y=["Human"], x=[human_count], orientation="h",
        marker_color=human_color, name="Human",
        text=[f"{human_count}"], textposition="outside",
        cliponaxis=False, customdata=[_ts],
    ))
    fig.update_layout(
        title=counts.get("title", ""),
        height=100, margin=dict(l=80, r=40, t=25, b=5),
        showlegend=False, barmode="group",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   range=[0, max_count * 1.5]),
        yaxis=dict(showgrid=False),
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(family="Space Grotesk, Inter, sans-serif", color=INK),
    )

    return new_selection, fig


@app.callback(
    Output("category-override-warning", "children"),
    Input("highlight-selector", "value"),
)
def show_category_override_warning(highlight_value):
    """Show a warning when no highlight track is selected."""
    if not highlight_value or highlight_value == "none":
        return (
            "⚠ Select a highlight strategy first. Automation proportion requires a track "
            "to compute against — pick a highlight above, or specify every category manually."
        )
    return ""


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
