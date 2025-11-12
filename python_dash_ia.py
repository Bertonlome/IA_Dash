"""
© Benjamin R. Berton 2025 Polytechnique Montreal
"""
import dash
from dash import html, dcc, dash_table, Input, Output, State, callback_context
import plotly.graph_objects as go
import pandas as pd
import os
import base64
import io

app = dash.Dash(__name__)
DATA_FILE = "table_hat_game.csv"
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)

# Update agent columns
agent_columns = ["Human*", "UGV", "UAV", "UGV*", "UAV*", "Human"]
editable_columns = agent_columns

# Update dropdowns for new agent columns
color_options = ["red", "yellow", "green", "orange"]
dropdowns = {
    col: {
        'options': [{'label': c.capitalize(), 'value': c} for c in color_options]
    }
    for col in agent_columns
}

# Update columns definition
columns = [
    {"name": "Row", "id": "Row", "editable": False},
    {"name": "Procedure", "id": "Procedure", "editable": True},
    {"name": "Task", "id": "Task", "editable": True},
    {"name": "Human*", "id": "Human*", "editable": True, "presentation": "dropdown"},
    {"name": "UGV", "id": "UGV", "editable": True, "presentation": "dropdown"},
    {"name": "UAV", "id": "UAV", "editable": True, "presentation": "dropdown"},
    {"name": "UGV*", "id": "UGV*", "editable": True, "presentation": "dropdown"},
    {"name": "UAV*", "id": "UAV*", "editable": True, "presentation": "dropdown"},
    {"name": "Human", "id": "Human", "editable": True, "presentation": "dropdown"},
    {"name": "Observability", "id": "Observability", "editable": True},
    {"name": "Predictability", "id": "Predictability", "editable": True},
    {"name": "Directability", "id": "Directability", "editable": True},
]

def wrap_text(text, max_width=30):
    import textwrap
    if not isinstance(text, str):
        return ""
    return '<br>'.join(textwrap.wrap(text, width=max_width))


def style_table(df):
    styles = []
    for i, row in df.iterrows():
        for col in editable_columns:
            value = row[col]
            color = value.lower() if isinstance(value, str) else "white"
            styles.append({
                'if': {'row_index': i, 'column_id': col},
                'backgroundColor': color if color != "white" else "#ffffff",
                'color': color,
                'textAlign': 'center'
            })
    return styles
df = df.assign(Row=lambda x: x.index + 1)
editable_columns = ["Human*", "UGV", "UAV", "UGV*", "UAV*", "Human"]
table = dash_table.DataTable(
    id='responsibility-table',
    columns=[
        {"name": col, "id": col, "editable": col in editable_columns}
        for col in df.columns
    ],
    data=df.assign(Row=lambda x: x.index + 1).to_dict("records"),
    editable=True,
    row_deletable=True,
    dropdown=dropdowns,
    style_data_conditional=style_table(df),
    style_cell={'textAlign': 'left', 'padding': '5px', 'whiteSpace': 'normal'},
    style_cell_conditional=[
        {
            'if': {'column_id': 'Human*'},
            'borderLeft': '3px solid black'
        },
        {
            'if': {'column_id': 'Human'},
            'borderRight': '3px solid black'
        },
        {
            'if': {'column_id': 'TARS'},
            'borderRight': '2px solid black'
        }
    ],
    style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0'},
    style_table={'overflowX': 'auto', 'border': '1px solid lightgrey'},
    active_cell={"row": 0, "column_id":"Objective", "column":1}
)

def build_interdependence_figures(df, highlight_track=None):
    agents = ["Human*", "UGV", "UAV", "UGV*", "UAV*", "Human"]
    VALID_COLORS = {"red", "yellow", "green", "orange"}

    figures = {}

    for procedure, proc_df in df.groupby("Procedure"):
        tasks = proc_df["Task"].tolist()
        height = 300 + len(tasks) * 100
        proc_df = proc_df.reset_index(drop=True)
        proc_df["task_idx"] = proc_df.index

        dots = []
        black_to_black_arrows = []
        grey_to_black_arrows = []
        horizontal_bidirectional_arrows = []
        most_reliable_track = []
        all_task_performers = []

        for idx, row in proc_df.iterrows():
            task_idx = row["task_idx"]
            black_points = []

            agent_colors = {
                "Human*": str(row.get("Human*", "") or "").strip().lower(),
                "UGV*": str(row.get("UGV*", "") or "").strip().lower(),
                "UAV*": str(row.get("UAV*", "") or "").strip().lower(),
            }

            # Most reliable path logic (green > yellow > orange)
            chosen_agent = None
            for agent in ["Human*", "UGV*", "UAV*"]:
                if agent_colors[agent] == "green":
                    chosen_agent = agent
                    break
            if not chosen_agent:
                for agent in ["Human*", "UGV*", "UAV*"]:
                    if agent_colors[agent] == "yellow":
                        chosen_agent = agent
                        break
            if not chosen_agent:
                for agent in ["Human*", "UGV*", "UAV*"]:
                    if agent_colors[agent] == "orange":
                        chosen_agent = agent
                        break
            if chosen_agent:
                most_reliable_track.append((task_idx, chosen_agent))

            # Team alternative 1: Human* performer, UGV/UAV supporters
            val_human_star = str(row.get("Human*", "") or "").strip().lower()
            if val_human_star in VALID_COLORS:
                dots.append({"task": task_idx, "agent": "Human*", "color": val_human_star})
                if val_human_star != "red":
                    black_points.append("Human*")
            val_ugv = str(row.get("UGV", "") or "").strip().lower()
            if val_ugv in VALID_COLORS:
                dots.append({"task": task_idx, "agent": "UGV", "color": val_ugv})
                if val_human_star != "red" and val_ugv != "red":
                    grey_to_black_arrows.append({
                        "start_agent": "UGV",
                        "end_agent": "Human*",
                        "task": task_idx
                    })
            val_uav = str(row.get("UAV", "") or "").strip().lower()
            if val_uav in VALID_COLORS:
                dots.append({"task": task_idx, "agent": "UAV", "color": val_uav})
                if val_human_star != "red" and val_uav != "red":
                    grey_to_black_arrows.append({
                        "start_agent": "UAV",
                        "end_agent": "Human*",
                        "task": task_idx
                    })

            # Team alternative 2: UGV* or UAV* performer, Human supporter
            val_ugv_star = str(row.get("UGV*", "") or "").strip().lower()
            val_uav_star = str(row.get("UAV*", "") or "").strip().lower()
            performer_candidates = []
            performer_grade = None

            # Find best grade for UGV* and UAV*
            for grade in ["green", "yellow", "orange"]:
                if val_ugv_star == grade or val_uav_star == grade or val_human_star == grade:
                    performer_grade = grade
                    break

            # Collect all possible performers for this task
            task_performers = []
            for agent, val in [("Human*", val_human_star), ("UGV*", val_ugv_star), ("UAV*", val_uav_star)]:
                if val == performer_grade and val != "red":
                    task_performers.append(agent)
            all_task_performers.append(task_performers)

            if val_ugv_star in VALID_COLORS:
                dots.append({"task": task_idx, "agent": "UGV*", "color": val_ugv_star})
            if val_uav_star in VALID_COLORS:
                dots.append({"task": task_idx, "agent": "UAV*", "color": val_uav_star})

            val_human = str(row.get("Human", "") or "").strip().lower()
            if val_human in VALID_COLORS:
                dots.append({"task": task_idx, "agent": "Human", "color": val_human})
                if val_ugv_star in VALID_COLORS and val_ugv_star != "red":
                    grey_to_black_arrows.append({
                        "start_agent": "Human",
                        "end_agent": "UGV*",
                        "task": task_idx
                    })
                if val_uav_star in VALID_COLORS and val_uav_star != "red":
                    grey_to_black_arrows.append({
                        "start_agent": "Human",
                        "end_agent": "UAV*",
                        "task": task_idx
                    })

            # If both UGV* and UAV* are performers with same grade, draw horizontal bidirectional arrow
            if "UGV*" in task_performers and "UAV*" in task_performers:
                horizontal_bidirectional_arrows.append({
                    "left_agent": "UGV*",
                    "right_agent": "UAV*",
                    "task": task_idx
                })

        # Draw solid arrows for performer transitions (from previous to current)
        for i in range(1, len(proc_df)):
            prev_performers = all_task_performers[i-1]
            curr_performers = all_task_performers[i]

            # If both UGV* and UAV* are top performers in the PREVIOUS task, only use UGV* as source
            if "UGV*" in prev_performers and "UAV*" in prev_performers:
                filtered_prev_performers = [p for p in prev_performers if p != "UAV*"]
            else:
                filtered_prev_performers = prev_performers

            # If both UGV* and UAV* are top performers in the CURRENT task, only draw arrows to UGV*
            if "UGV*" in curr_performers and "UAV*" in curr_performers:
                filtered_curr_performers = [p for p in curr_performers if p != "UAV*"]
            else:
                filtered_curr_performers = curr_performers

            for prev_agent in filtered_prev_performers:
                for curr_agent in filtered_curr_performers:
                    black_to_black_arrows.append({
                        "start_task": i-1,
                        "start_agent": prev_agent,
                        "end_task": i,
                        "end_agent": curr_agent
                    })

        agent_pos = {agent: i for i, agent in enumerate(agents)}
        fig = go.Figure()

        for dot in dots:
            row = proc_df.iloc[dot["task"]]
            hover_text = (
                f"<b>Task:</b> {wrap_text(row['Task'])}<br>"
                f"<b>Agent:</b> {dot['agent']}<br><br>"
                f"<b>Observability:</b><br>{wrap_text(row.get('Observability', ''))}<br><br>"
                f"<b>Predictability:</b><br>{wrap_text(row.get('Predictability', ''))}<br><br>"
                f"<b>Directability:</b><br>{wrap_text(row.get('Directability', ''))}"
            )
            fig.add_trace(go.Scatter(
                x=[agent_pos[dot["agent"]]],
                y=[dot["task"]],
                mode="markers",
                marker=dict(size=20, color=dot["color"], symbol="circle"),
                showlegend=False,
                hoverinfo="text",
                hovertext=hover_text
            ))

        for arrow in grey_to_black_arrows:
            fig.add_shape(
                type="line",
                x0=agent_pos[arrow["start_agent"]],
                y0=arrow["task"],
                x1=agent_pos[arrow["end_agent"]],
                y1=arrow["task"],
                line=dict(
                    color="black",
                    width=2,
                    dash="dot"
                )
            )

        for arrow in black_to_black_arrows:
            is_highlighted = False
            if highlight_track == "most_reliable":
                is_highlighted = (
                    (arrow["start_task"], arrow["start_agent"]) in most_reliable_track and
                    (arrow["end_task"], arrow["end_agent"]) in most_reliable_track
                )
            fig.add_annotation(
                x=agent_pos[arrow["end_agent"]],
                y=arrow["end_task"],
                ax=agent_pos[arrow["start_agent"]],
                ay=arrow["start_task"],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=4 if is_highlighted else 2,
                arrowcolor="crimson" if is_highlighted else "black",
                opacity=0.9
            )

        for arrow in horizontal_bidirectional_arrows:
            y = arrow["task"]
            x_left = agent_pos[arrow["left_agent"]]
            x_right = agent_pos[arrow["right_agent"]]
            
            # Check if most reliable track goes through UGV* for this task
            is_highlighted = False
            if highlight_track == "most_reliable":
                is_highlighted = (y, "UGV*") in most_reliable_track
            
            # Draw left-to-right arrow
            fig.add_annotation(
                x=x_right, y=y, ax=x_left, ay=y,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=4 if is_highlighted else 2,
                arrowcolor="crimson" if is_highlighted else "black",
                opacity=0.9
            )
            # Draw right-to-left arrow
            fig.add_annotation(
                x=x_left, y=y, ax=x_right, ay=y,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=4 if is_highlighted else 2,
                arrowcolor="crimson" if is_highlighted else "black",
                opacity=0.9
            )

        fig.update_layout(
            title=f"Workflow for {procedure}",
            xaxis=dict(
                tickvals=list(agent_pos.values()),
                ticktext=list(agent_pos.keys()),
                title="Agent",
                showgrid=True,
                gridcolor='lightgrey'
            ),
            yaxis=dict(
                tickvals=list(range(len(tasks))),
                ticktext=[wrap_text(task) for task in tasks],
                title="Task",
                autorange="reversed",
                showgrid=True,
                gridcolor='lightgrey'
            ),
            height=height,
            margin=dict(l=200, r=50, t=50, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white',
        )
        figures[procedure] = fig
    return figures

def build_combined_interdependence_figure(df, highlight_track=None):
    agents = ["Human*", "UGV", "UAV", "UGV*", "UAV*", "Human"]
    VALID_COLORS = {"red", "yellow", "green", "orange"}
    tasks = df["Task"].tolist()
    height = 600 + len(tasks) * 100
    dots = []
    black_to_black_arrows = []
    grey_to_black_arrows = []
    horizontal_bidirectional_arrows = []

    df = df.reset_index(drop=True)
    df["task_idx"] = df.index

    most_reliable_track = []

    # Store all possible performers for each task
    all_task_performers = []

    for idx, row in df.iterrows():
        task_idx = row["task_idx"]
        black_points = []

        # Get performer colors for both alternatives
        agent_colors = {
            "Human*": str(row.get("Human*", "") or "").strip().lower(),
            "UGV*": str(row.get("UGV*", "") or "").strip().lower(),
            "UAV*": str(row.get("UAV*", "") or "").strip().lower(),
        }

        # Most reliable path logic (green > yellow > orange)
        chosen_agent = None
        for agent in ["Human*", "UGV*", "UAV*"]:
            if agent_colors[agent] == "green":
                chosen_agent = agent
                break
        if not chosen_agent:
            for agent in ["Human*", "UGV*", "UAV*"]:
                if agent_colors[agent] == "yellow":
                    chosen_agent = agent
                    break
        if not chosen_agent:
            for agent in ["Human*", "UGV*", "UAV*"]:
                if agent_colors[agent] == "orange":
                    chosen_agent = agent
                    break
        if chosen_agent:
            most_reliable_track.append((task_idx, chosen_agent))

        # Team alternative 1: Human* performer, UGV/UAV supporters
        val_human_star = str(row.get("Human*", "") or "").strip().lower()
        if val_human_star in VALID_COLORS:
            dots.append({"task": task_idx, "agent": "Human*", "color": val_human_star})
            black_points.append("Human*")
        val_ugv = str(row.get("UGV", "") or "").strip().lower()
        if val_ugv in VALID_COLORS:
            dots.append({"task": task_idx, "agent": "UGV", "color": val_ugv})
            if val_human_star != "red" and val_ugv != "red":
                grey_to_black_arrows.append({
                    "start_agent": "UGV",
                    "end_agent": "Human*",
                    "task": task_idx
                })
        val_uav = str(row.get("UAV", "") or "").strip().lower()
        if val_uav in VALID_COLORS:
            dots.append({"task": task_idx, "agent": "UAV", "color": val_uav})
            if val_human_star != "red" and val_uav != "red":
                grey_to_black_arrows.append({
                    "start_agent": "UAV",
                    "end_agent": "Human*",
                    "task": task_idx
                })

        # Team alternative 2: UGV* or UAV* performer, Human supporter
        val_ugv_star = str(row.get("UGV*", "") or "").strip().lower()
        val_uav_star = str(row.get("UAV*", "") or "").strip().lower()
        performer_candidates = []
        performer_grade = None

        # Find best grade for UGV* and UAV*
        for grade in ["green", "yellow", "orange"]:
            if val_ugv_star == grade or val_uav_star == grade or val_human_star == grade:
                performer_grade = grade
                break

        # Collect all possible performers for this task
        task_performers = []
        for agent, val in [("Human*", val_human_star), ("UGV*", val_ugv_star), ("UAV*", val_uav_star)]:
            if val == performer_grade and val != "red":
                task_performers.append(agent)
        all_task_performers.append(task_performers)

        if val_ugv_star in VALID_COLORS:
            dots.append({"task": task_idx, "agent": "UGV*", "color": val_ugv_star})
        if val_uav_star in VALID_COLORS:
            dots.append({"task": task_idx, "agent": "UAV*", "color": val_uav_star})

        val_human = str(row.get("Human", "") or "").strip().lower()
        if val_human in VALID_COLORS:
            dots.append({"task": task_idx, "agent": "Human", "color": val_human})
            # Supporter arrows
            if val_ugv_star in VALID_COLORS and val_ugv_star != "red":
                grey_to_black_arrows.append({
                    "start_agent": "Human",
                    "end_agent": "UGV*",
                    "task": task_idx
                })
            if val_uav_star in VALID_COLORS and val_uav_star != "red":
                grey_to_black_arrows.append({
                    "start_agent": "Human",
                    "end_agent": "UAV*",
                    "task": task_idx
                })

        # If both UGV* and UAV* are performers with same grade, draw horizontal bidirectional arrow
        if "UGV*" in task_performers and "UAV*" in task_performers:
            horizontal_bidirectional_arrows.append({
                "left_agent": "UGV*",
                "right_agent": "UAV*",
                "task": task_idx
            })

    # Draw solid arrows for performer transitions (from previous to current)
    for i in range(1, len(df)):
        prev_performers = all_task_performers[i-1]
        curr_performers = all_task_performers[i]

        # If both UGV* and UAV* are top performers in the PREVIOUS task, only use UGV* as source
        if "UGV*" in prev_performers and "UAV*" in prev_performers:
            filtered_prev_performers = [p for p in prev_performers if p != "UAV*"]
        else:
            filtered_prev_performers = prev_performers

        # If both UGV* and UAV* are top performers in the CURRENT task, only draw arrows to UGV*
        if "UGV*" in curr_performers and "UAV*" in curr_performers:
            filtered_curr_performers = [p for p in curr_performers if p != "UAV*"]
        else:
            filtered_curr_performers = curr_performers

        for prev_agent in filtered_prev_performers:
            for curr_agent in filtered_curr_performers:
                black_to_black_arrows.append({
                    "start_task": i-1,
                    "start_agent": prev_agent,
                    "end_task": i,
                    "end_agent": curr_agent
                })

    agent_pos = {agent: i for i, agent in enumerate(agents)}
    fig = go.Figure()

    for dot in dots:
        row = df.iloc[dot["task"]]
        hover_text = (
            f"<b>Task:</b> {wrap_text(row['Task'])}<br>"
            f"<b>Agent:</b> {dot['agent']}<br><br>"
            f"<b>Observability:</b><br>{wrap_text(row.get('Observability', ''))}<br><br>"
            f"<b>Predictability:</b><br>{wrap_text(row.get('Predictability', ''))}<br><br>"
            f"<b>Directability:</b><br>{wrap_text(row.get('Directability', ''))}"
        )
        fig.add_trace(go.Scatter(
            x=[agent_pos[dot["agent"]]],
            y=[dot["task"]],
            mode="markers",
            marker=dict(size=20, color=dot["color"], symbol="circle"),
            showlegend=False,
            hoverinfo="text",
            hovertext=hover_text
        ))

    for arrow in grey_to_black_arrows:
        fig.add_shape(
            type="line",
            x0=agent_pos[arrow["start_agent"]],
            y0=arrow["task"],
            x1=agent_pos[arrow["end_agent"]],
            y1=arrow["task"],
            line=dict(
                color="black",
                width=2,
                dash="dot"
            )
        )

    for arrow in black_to_black_arrows:
        is_highlighted = False
        if highlight_track == "most_reliable":
            is_highlighted = (
                (arrow["start_task"], arrow["start_agent"]) in most_reliable_track and
                (arrow["end_task"], arrow["end_agent"]) in most_reliable_track
            )
        fig.add_annotation(
            x=agent_pos[arrow["end_agent"]],
            y=arrow["end_task"],
            ax=agent_pos[arrow["start_agent"]],
            ay=arrow["start_task"],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=4 if is_highlighted else 2,
            arrowcolor="crimson" if is_highlighted else "black",
            opacity=0.9
        )

    for arrow in horizontal_bidirectional_arrows:
        y = arrow["task"]
        x_left = agent_pos[arrow["left_agent"]]
        x_right = agent_pos[arrow["right_agent"]]
        
        # Check if most reliable track goes through UGV* for this task
        is_highlighted = False
        if highlight_track == "most_reliable":
            is_highlighted = (y, "UGV*") in most_reliable_track
        
        # Draw left-to-right arrow
        fig.add_annotation(
            x=x_right, y=y, ax=x_left, ay=y,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=4 if is_highlighted else 2,
            arrowcolor="crimson" if is_highlighted else "black",
            opacity=0.9
        )
        # Draw right-to-left arrow
        fig.add_annotation(
            x=x_left, y=y, ax=x_right, ay=y,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=4 if is_highlighted else 2,
            arrowcolor="crimson" if is_highlighted else "black",
            opacity=0.9
        )

    fig.update_layout(
        title="Combined Workflow Graph (All Procedures)",
        xaxis=dict(
            tickvals=list(agent_pos.values()),
            ticktext=list(agent_pos.keys()),
            title="Agent",
            showgrid=True,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            tickvals=list(df["task_idx"]),
            ticktext=[wrap_text(f"{p} | {t}") for p, t in zip(df["Procedure"], df["Task"])],
            title="Procedure | Task",
            autorange="reversed",
            showgrid=True,
            gridcolor='lightgrey'
        ),
        margin=dict(l=250, r=50, t=50, b=50),
        height=height,
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    return fig


app.layout = html.Div([
    html.Div(
        "© Benjamin R. Berton 2025 Polytechnique Montreal",
        style={
            "textAlign": "center",
            "color": "#888",
            "fontSize": "14px",
            "marginBottom": "10px"
        }
    ),
    # The table goes here
    html.H2("Interdependence Analysis Table", style={"textAlign": "center"}),
    html.Div(id="table-wrapper", children=[table]),
    html.Div([
        html.Div([
            # Left-aligned buttons
            html.Div([
                dcc.Upload(
                    id='upload-data',
                    children=html.Button('📂 Load Table', id='load-button', n_clicks=0),
                    multiple=False,
                    style={
                        'display': 'inline-block',
                        'marginRight': '10px'
                    }
                ),
                html.Button("➕ Add Row", id="add-row-button", n_clicks=0),
                html.Button("⬇️ Copy Cell Down", id="copy-down-button", n_clicks=0),
            ], style={"display": "flex", "gap": "10px"}),

            # Right-aligned Save button
            html.Div([
                html.Button("💾 Save Table", disabled=False, id="save-button", n_clicks=0),
                dcc.Download(id="download-csv")
            ], style={"marginLeft": "auto"})  # pushes this div to the right
        ], style={"display": "flex", "width": "100%"}),
        html.Div(id="save-confirmation", style={"marginTop": "10px", "fontStyle": "italic"})
    ]),

    html.H2("Interdependence Workflow Graph", style={"textAlign": "center"}),

    # Dropdown menu to select procedure
    html.Div([
        dcc.Dropdown(
            id="procedure-dropdown",
            options=[{"label": proc, "value": proc} for proc in df["Procedure"].unique()],
            value=None,
            placeholder="Select a procedure to filter the graph...",
            clearable=True,
            style={"width": "50%", "margin": "0 auto"}
        )
    ]),

    dcc.RadioItems(
    id="highlight-selector",
    options=[
        {"label": "No highlight", "value": "none"},
        {"label": "Most reliable path", "value": "most_reliable"}
    ],
    value="none",
    labelStyle={'display': 'inline-block', 'margin-right': '20px'},
    style={"textAlign": "center", "marginTop": "20px"}
    ),

    # Graph
    dcc.Graph(id="interdependence-graph", config={
        "displayModeBar": False
    }),
    # Labels
    html.Div([
        html.Div("Team Alternative 1", style={
            "width": "50%", "display": "inline-block", "textAlign": "center", "marginTop": "10px", "marginLeft": "150px"
        }),
        html.Div("Team Alternative 2", style={
            "width": "50%", "display": "inline-block", "textAlign": "center", "marginTop": "10px"
        }),
    ], style={"display": "flex", "width": "100%"}),

    # Bar chart for most reliable path color counts
    html.Div([
        dcc.Graph(id="overall-scenario-bar-chart"),
        dcc.Graph(id="most-reliable-bar-chart"),
        dcc.Graph(id="allocation-type-bar-chart"),
        dcc.Graph(id="agent-autonomy-bar-chart")
    ]),

    # Footer with copyright
    html.Footer(
        "© Benjamin R. Berton 2025 Polytechnique Montreal",
        style={
            "textAlign": "center",
            "marginTop": "40px",
            "padding": "10px 0",
            "color": "#888",
            "fontSize": "14px"
        }
    ),

], style={"fontFamily": "'Roboto', 'Helvetica', 'Arial', sans-serif"})

@app.callback(
    Output("interdependence-graph", "figure"),
    Output("overall-scenario-bar-chart", "figure"),
    Output("most-reliable-bar-chart", "figure"),
    Output("allocation-type-bar-chart", "figure"),
    Output("agent-autonomy-bar-chart", "figure"),
    Input("procedure-dropdown", "value"),
    Input("highlight-selector", "value"),
    Input("responsibility-table", "data")
)
def update_graph_and_bar(procedure, highlight_track, data):
    df = pd.DataFrame(data)
    if df.empty:
        return go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure()
    if highlight_track == "none":
        highlight_track = None

    # --- Workflow Graph ---
    if procedure is None:
        workflow_fig = build_combined_interdependence_figure(df, highlight_track)
        df_bar = df
    else:
        figures = build_interdependence_figures(df, highlight_track)
        workflow_fig = figures.get(procedure, go.Figure())
        df_bar = df[df["Procedure"] == procedure]

# --- Bar Chart for the whole scenario---
    performer_green = 0
    performer_yellow = 0
    performer_orange = 0
    supporter_green = 0
    supporter_yellow = 0
    supporter_orange = 0
    human_performer_green = 0
    human_performer_yellow = 0
    human_performer_orange = 0
    human_supporter_green = 0
    human_supporter_yellow = 0
    human_supporter_orange = 0
    ugv_performer_green = 0
    ugv_performer_yellow = 0
    ugv_performer_orange = 0
    ugv_supporter_green = 0
    ugv_supporter_yellow = 0
    ugv_supporter_orange = 0
    uav_performer_green = 0
    uav_performer_yellow = 0
    uav_performer_orange = 0
    uav_supporter_green = 0
    uav_supporter_yellow = 0
    uav_supporter_orange = 0

    for idx, row in df_bar.iterrows():
        agent_colors = {
            "HUMAN*": str(row.get("Human*", "") or "").strip().lower(),
            "UGV*": str(row.get("UGV*", "") or "").strip().lower(),
            "UAV*": str(row.get("UAV*", "") or "").strip().lower(),
        }
        supporter_colors = {
            "HUMAN": str(row.get("Human", "") or "").strip().lower(),
            "UGV": str(row.get("UGV", "") or "").strip().lower(),
            "UAV": str(row.get("UAV", "") or "").strip().lower(),
        }
        # Find performer (most reliable)
        for agent in ["HUMAN*", "UGV*", "UAV*"]:
            if agent_colors[agent] == "green":
                if agent ==  "HUMAN*":
                    human_performer_green += 1
                if agent == "UGV*":
                    ugv_performer_green += 1
                if agent == "UAV*":
                    uav_performer_green += 1
                performer_green += 1
            if agent_colors[agent] == "yellow":
                if agent ==  "HUMAN*":
                    human_performer_yellow += 1
                if agent == "UGV*":
                    ugv_performer_yellow += 1
                if agent == "UAV*":
                    uav_performer_yellow += 1
                performer_yellow += 1
            if agent_colors[agent] == "orange":
                if agent ==  "HUMAN*":
                    human_performer_orange += 1
                if agent == "UGV*":
                    ugv_performer_orange += 1
                if agent == "UAV*":
                    uav_performer_orange += 1
                performer_orange += 1
        for agent in ["HUMAN", "UGV", "UAV"]:
            if supporter_colors[agent] == "green":
                if agent ==  "HUMAN":
                    human_supporter_green += 1
                if agent == "UGV":
                    ugv_supporter_green += 1
                if agent == "UAV":
                    uav_supporter_green += 1
                supporter_green += 1
            if supporter_colors[agent] == "yellow":
                if agent ==  "HUMAN":
                    human_supporter_yellow += 1
                if agent == "UGV":
                    ugv_supporter_yellow += 1
                if agent == "UAV":
                    uav_supporter_yellow += 1
                supporter_yellow += 1
            if supporter_colors[agent] == "orange":
                if agent ==  "HUMAN":
                    human_supporter_orange += 1
                if agent == "UGV":
                    ugv_supporter_orange += 1
                if agent == "UAV":
                    uav_supporter_orange += 1
                supporter_orange += 1

    bar_fig_whole_scenario = go.Figure()
    
    # Individual bars for each agent and color - Performers
    bar_fig_whole_scenario.add_trace(go.Bar(
        name="Human",
        x=["Performer Green"],
        y=[human_performer_green],
        marker_color="seagreen",
        showlegend=False,
        text=["Human"],
        textposition='outside',
        textangle=0
    ))
    bar_fig_whole_scenario.add_trace(go.Bar(
        name="UGV",
        x=["Performer Green"],
        y=[ugv_performer_green],
        marker_color="limegreen",
        showlegend=False,
        text=["UGV"],
        textposition='outside',
        textangle=0
    ))
    bar_fig_whole_scenario.add_trace(go.Bar(
        name="UAV",
        x=["Performer Green"],
        y=[uav_performer_green],
        marker_color="mediumspringgreen",
        showlegend=False,
        text=["UAV"],
        textposition='outside',
        textangle=0
    ))
    
    bar_fig_whole_scenario.add_trace(go.Bar(
        name="Human",
        x=["Performer Yellow"],
        y=[human_performer_yellow],
        marker_color="gold",
        showlegend=False,
        text=["Human"],
        textposition='outside',
        textangle=0
    ))
    bar_fig_whole_scenario.add_trace(go.Bar(
        name="UGV",
        x=["Performer Yellow"],
        y=[ugv_performer_yellow],
        marker_color="khaki",
        showlegend=False,
        text=["UGV"],
        textposition='outside',
        textangle=0
    ))
    bar_fig_whole_scenario.add_trace(go.Bar(
        name="UAV",
        x=["Performer Yellow"],
        y=[uav_performer_yellow],
        marker_color="lemonchiffon",
        showlegend=False,
        text=["UAV"],
        textposition='outside',
        textangle=0
    ))
    
    bar_fig_whole_scenario.add_trace(go.Bar(
        name="Human",
        x=["Performer Orange"],
        y=[human_performer_orange],
        marker_color="darkorange",
        showlegend=False,
        text=["Human"],
        textposition='outside',
        textangle=0
    ))
    bar_fig_whole_scenario.add_trace(go.Bar(
        name="UGV",
        x=["Performer Orange"],
        y=[ugv_performer_orange],
        marker_color="orange",
        showlegend=False,
        text=["UGV"],
        textposition='outside',
        textangle=0
    ))
    bar_fig_whole_scenario.add_trace(go.Bar(
        name="UAV",
        x=["Performer Orange"],
        y=[uav_performer_orange],
        marker_color="coral",
        showlegend=False,
        text=["UAV"],
        textposition='outside',
        textangle=0
    ))
    
    # Individual bars for each agent and color - Supporters
    bar_fig_whole_scenario.add_trace(go.Bar(
        name="Human",
        x=["Supporter Green"],
        y=[human_supporter_green],
        marker_color="seagreen",
        showlegend=False,
        text=["Human"],
        textposition='outside',
        textangle=0
    ))
    bar_fig_whole_scenario.add_trace(go.Bar(
        name="UGV",
        x=["Supporter Green"],
        y=[ugv_supporter_green],
        marker_color="limegreen",
        showlegend=False,
        text=["UGV"],
        textposition='outside',
        textangle=0
    ))
    bar_fig_whole_scenario.add_trace(go.Bar(
        name="UAV",
        x=["Supporter Green"],
        y=[uav_supporter_green],
        marker_color="mediumspringgreen",
        showlegend=False,
        text=["UAV"],
        textposition='outside',
        textangle=0
    ))
    
    bar_fig_whole_scenario.add_trace(go.Bar(
        name="Human",
        x=["Supporter Yellow"],
        y=[human_supporter_yellow],
        marker_color="gold",
        showlegend=False,
        text=["Human"],
        textposition='outside',
        textangle=0
    ))
    bar_fig_whole_scenario.add_trace(go.Bar(
        name="UGV",
        x=["Supporter Yellow"],
        y=[ugv_supporter_yellow],
        marker_color="khaki",
        showlegend=False,
        text=["UGV"],
        textposition='outside',
        textangle=0
    ))
    bar_fig_whole_scenario.add_trace(go.Bar(
        name="UAV",
        x=["Supporter Yellow"],
        y=[uav_supporter_yellow],
        marker_color="lemonchiffon",
        showlegend=False,
        text=["UAV"],
        textposition='outside',
        textangle=0
    ))
    
    bar_fig_whole_scenario.add_trace(go.Bar(
        name="Human",
        x=["Supporter Orange"],
        y=[human_supporter_orange],
        marker_color="darkorange",
        showlegend=False,
        text=["Human"],
        textposition='outside',
        textangle=0
    ))
    bar_fig_whole_scenario.add_trace(go.Bar(
        name="UGV",
        x=["Supporter Orange"],
        y=[ugv_supporter_orange],
        marker_color="orange",
        showlegend=False,
        text=["UGV"],
        textposition='outside',
        textangle=0
    ))
    bar_fig_whole_scenario.add_trace(go.Bar(
        name="UAV",
        x=["Supporter Orange"],
        y=[uav_supporter_orange],
        marker_color="coral",
        showlegend=False,
        text=["UAV"],
        textposition='outside',
        textangle=0
    ))
    
    bar_fig_whole_scenario.update_layout(
        title="Performer and Supporter Capacities in Mixed Initiative",
        xaxis_title="Role and Capacity",
        yaxis_title="Number of Tasks",
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False
    )

    # --- Bar Chart for Most Reliable Path ---
    performer_green = 0
    performer_yellow = 0
    performer_orange = 0
    supporter_green = 0
    supporter_yellow = 0
    supporter_orange = 0
    human_performer_green = 0
    human_performer_yellow = 0
    human_performer_orange = 0
    human_supporter_green = 0
    human_supporter_yellow = 0
    human_supporter_orange = 0
    ugv_performer_green = 0
    ugv_performer_yellow = 0
    ugv_performer_orange = 0
    ugv_supporter_green = 0
    ugv_supporter_yellow = 0
    ugv_supporter_orange = 0
    uav_performer_green = 0
    uav_performer_yellow = 0
    uav_performer_orange = 0
    uav_supporter_green = 0
    uav_supporter_yellow = 0
    uav_supporter_orange = 0
    for idx, row in df_bar.iterrows():
        agent_colors = {
            "HUMAN*": str(row.get("Human*", "") or "").strip().lower(),
            "UGV*": str(row.get("UGV*", "") or "").strip().lower(),
            "UAV*": str(row.get("UAV*", "") or "").strip().lower(),
        }
        supporter_colors = {
            "HUMAN": str(row.get("Human", "") or "").strip().lower(),
            "UGV": str(row.get("UGV", "") or "").strip().lower(),
            "UAV": str(row.get("UAV", "") or "").strip().lower(),
        }
        performer = None
        # Find performer (most reliable)
        if agent_colors["HUMAN*"] == "green":
            performer = "HUMAN*"
            performer_green += 1
            human_performer_green += 1
        elif agent_colors["UGV*"] == "green":
            performer = "UGV*"
            performer_green += 1
            ugv_performer_green += 1
            if agent_colors["UAV*"] == "green":
                performer += ";UAV*"
                performer_green += 1
                uav_performer_green += 1
        elif agent_colors["UAV*"] == "green":
            performer = "UAV*"
            performer_green += 1
            uav_performer_green += 1
        elif agent_colors["HUMAN*"] == "yellow":
            performer = "HUMAN*"
            performer_yellow += 1
            human_performer_yellow += 1
        elif agent_colors["UGV*"] == "yellow":
            performer = "UGV*"
            performer_yellow += 1
            ugv_performer_yellow += 1
            if agent_colors["UAV*"] == "yellow":
                performer += ";UAV*"
                performer_yellow += 1
                uav_performer_yellow += 1
        elif agent_colors["UAV*"] == "yellow":
            performer = "UAV*"
            performer_yellow += 1
            uav_performer_yellow += 1
        elif agent_colors["HUMAN*"] == "orange":
            performer = "HUMAN*"
            performer_orange += 1
            human_performer_orange += 1
        elif agent_colors["UGV*"] == "orange":
            performer = "UGV*"
            performer_orange += 1
            ugv_performer_orange += 1
            if agent_colors["UAV*"] == "orange":
                performer += ";UAV*"
                performer_orange += 1
                uav_performer_orange += 1
        elif agent_colors["UAV*"] == "orange":
            performer = "UAV*"
            performer_orange += 1
            uav_performer_orange += 1

        # Now check for supporter (the other agent)
        if performer == "HUMAN*":
            if supporter_colors["UGV"] == "green" or supporter_colors['UGV'] == "yellow" or supporter_colors['UGV'] == "orange":
                supporter_color = supporter_colors["UGV"]
                if supporter_color == "green":
                    supporter_green += 1
                    ugv_supporter_green += 1
                elif supporter_color == "yellow":
                    supporter_yellow += 1
                    ugv_supporter_yellow += 1
                elif supporter_color == "orange":
                    supporter_orange += 1
                    ugv_supporter_orange += 1
            if supporter_colors["UAV"] == "green" or supporter_colors['UAV'] == "yellow" or supporter_colors['UAV'] == "orange":
                supporter_color = supporter_colors["UAV"]
                if supporter_color == "green":
                    supporter_green += 1
                    uav_supporter_green += 1
                elif supporter_color == "yellow":
                    supporter_yellow += 1
                    uav_supporter_yellow += 1
                elif supporter_color == "orange":
                    supporter_orange += 1
                    uav_supporter_orange += 1
        elif performer == "UGV*" :
            if supporter_colors["UAV"] == "green" or supporter_colors['UAV'] == "yellow" or supporter_colors['UAV'] == "orange":
                supporter_color = supporter_colors["UAV"]
                if supporter_color == "green":
                    supporter_green += 1
                    uav_supporter_green += 1
                elif supporter_color == "yellow":
                    supporter_yellow += 1
                    uav_supporter_yellow += 1
                elif supporter_color == "orange":
                    supporter_orange += 1
                    uav_supporter_orange += 1
                    supporter_yellow += 1
                elif supporter_color == "orange":
                    supporter_orange += 1
                    uav_supporter_orange += 1
            if supporter_colors["HUMAN"] == "green" or supporter_colors['HUMAN'] == "yellow" or supporter_colors['HUMAN'] == "orange":
                supporter_color = supporter_colors["HUMAN"]
                if supporter_color == "green":
                    supporter_green += 1
                    human_supporter_green += 1
                elif supporter_color == "yellow":
                    supporter_yellow += 1
                    human_supporter_yellow += 1
                elif supporter_color == "orange":
                    supporter_orange += 1
                    human_supporter_orange += 1
        elif performer == "UAV*":
            if supporter_colors["UGV"] == "green" or supporter_colors['UGV'] == "yellow" or supporter_colors['UGV'] == "orange":
                supporter_color = supporter_colors["UGV"]
                if supporter_color == "green":
                    supporter_green += 1
                    ugv_supporter_green += 1
                elif supporter_color == "yellow":
                    supporter_yellow += 1
                    ugv_supporter_yellow += 1
                elif supporter_color == "orange":
                    supporter_orange += 1
                    ugv_supporter_orange += 1
            if supporter_colors["HUMAN"] == "green" or supporter_colors['HUMAN'] == "yellow" or supporter_colors['HUMAN'] == "orange":
                supporter_color = supporter_colors["HUMAN"]
                if supporter_color == "green":
                    supporter_green += 1
                    human_supporter_green += 1
                elif supporter_color == "yellow":
                    supporter_yellow += 1
                    human_supporter_yellow += 1
                elif supporter_color == "orange":
                    supporter_orange += 1
                    human_supporter_orange += 1
        elif performer == "UGV*;UAV*":
            if supporter_colors["HUMAN"] == "green" or supporter_colors['HUMAN'] == "yellow" or supporter_colors['HUMAN'] == "orange":
                supporter_color = supporter_colors["HUMAN"]
                if supporter_color == "green":
                    supporter_green += 1
                    human_supporter_green += 1
                elif supporter_color == "yellow":
                    supporter_yellow += 1
                    human_supporter_yellow += 1
                elif supporter_color == "orange":
                    supporter_orange += 1
                    human_supporter_orange += 1

    bar_fig = go.Figure()
    
    # Individual bars for each agent and color - Performers
    bar_fig.add_trace(go.Bar(
        name="Human",
        x=["Performer Green"],
        y=[human_performer_green],
        marker_color="seagreen",
        showlegend=False,
        text=["Human"],
        textposition='outside',
        textangle=0
    ))
    bar_fig.add_trace(go.Bar(
        name="UGV",
        x=["Performer Green"],
        y=[ugv_performer_green],
        marker_color="limegreen",
        showlegend=False,
        text=["UGV"],
        textposition='outside',
        textangle=0
    ))
    bar_fig.add_trace(go.Bar(
        name="UAV",
        x=["Performer Green"],
        y=[uav_performer_green],
        marker_color="mediumspringgreen",
        showlegend=False,
        text=["UAV"],
        textposition='outside',
        textangle=0
    ))
    
    bar_fig.add_trace(go.Bar(
        name="Human",
        x=["Performer Yellow"],
        y=[human_performer_yellow],
        marker_color="gold",
        showlegend=False,
        text=["Human"],
        textposition='outside',
        textangle=0
    ))
    bar_fig.add_trace(go.Bar(
        name="UGV",
        x=["Performer Yellow"],
        y=[ugv_performer_yellow],
        marker_color="khaki",
        showlegend=False,
        text=["UGV"],
        textposition='outside',
        textangle=0
    ))
    bar_fig.add_trace(go.Bar(
        name="UAV",
        x=["Performer Yellow"],
        y=[uav_performer_yellow],
        marker_color="lemonchiffon",
        showlegend=False,
        text=["UAV"],
        textposition='outside',
        textangle=0
    ))
    
    bar_fig.add_trace(go.Bar(
        name="Human",
        x=["Performer Orange"],
        y=[human_performer_orange],
        marker_color="darkorange",
        showlegend=False,
        text=["Human"],
        textposition='outside',
        textangle=0
    ))
    bar_fig.add_trace(go.Bar(
        name="UGV",
        x=["Performer Orange"],
        y=[ugv_performer_orange],
        marker_color="orange",
        showlegend=False,
        text=["UGV"],
        textposition='outside',
        textangle=0
    ))
    bar_fig.add_trace(go.Bar(
        name="UAV",
        x=["Performer Orange"],
        y=[uav_performer_orange],
        marker_color="coral",
        showlegend=False,
        text=["UAV"],
        textposition='outside',
        textangle=0
    ))
    
    # Individual bars for each agent and color - Supporters
    bar_fig.add_trace(go.Bar(
        name="Human",
        x=["Supporter Green"],
        y=[human_supporter_green],
        marker_color="seagreen",
        showlegend=False,
        text=["Human"],
        textposition='outside',
        textangle=0
    ))
    bar_fig.add_trace(go.Bar(
        name="UGV",
        x=["Supporter Green"],
        y=[ugv_supporter_green],
        marker_color="limegreen",
        showlegend=False,
        text=["UGV"],
        textposition='outside',
        textangle=0
    ))
    bar_fig.add_trace(go.Bar(
        name="UAV",
        x=["Supporter Green"],
        y=[uav_supporter_green],
        marker_color="mediumspringgreen",
        showlegend=False,
        text=["UAV"],
        textposition='outside',
        textangle=0
    ))
    
    bar_fig.add_trace(go.Bar(
        name="Human",
        x=["Supporter Yellow"],
        y=[human_supporter_yellow],
        marker_color="gold",
        showlegend=False,
        text=["Human"],
        textposition='outside',
        textangle=0
    ))
    bar_fig.add_trace(go.Bar(
        name="UGV",
        x=["Supporter Yellow"],
        y=[ugv_supporter_yellow],
        marker_color="khaki",
        showlegend=False,
        text=["UGV"],
        textposition='outside',
        textangle=0
    ))
    bar_fig.add_trace(go.Bar(
        name="UAV",
        x=["Supporter Yellow"],
        y=[uav_supporter_yellow],
        marker_color="lemonchiffon",
        showlegend=False,
        text=["UAV"],
        textposition='outside',
        textangle=0
    ))
    
    bar_fig.add_trace(go.Bar(
        name="Human",
        x=["Supporter Orange"],
        y=[human_supporter_orange],
        marker_color="darkorange",
        showlegend=False,
        text=["Human"],
        textposition='outside',
        textangle=0
    ))
    bar_fig.add_trace(go.Bar(
        name="UGV",
        x=["Supporter Orange"],
        y=[ugv_supporter_orange],
        marker_color="orange",
        showlegend=False,
        text=["UGV"],
        textposition='outside',
        textangle=0
    ))
    bar_fig.add_trace(go.Bar(
        name="UAV",
        x=["Supporter Orange"],
        y=[uav_supporter_orange],
        marker_color="coral",
        showlegend=False,
        text=["UAV"],
        textposition='outside',
        textangle=0
    ))
    
    bar_fig.update_layout(
        title="Most Reliable Path: Performer and Supporter Capacities",
        xaxis_title="Role and Capacity",
        yaxis_title="Number of Tasks",
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False
    )

    # --- Allocation Type Analysis ---
    single_allocation_independent = 0
    multiple_allocation_independent = 0
    interdependent = 0

    for idx, row in df_bar.iterrows():
        # Get all performer and supporter values
        human_star = str(row.get("Human*", "") or "").strip().lower()
        ugv_star = str(row.get("UGV*", "") or "").strip().lower()
        uav_star = str(row.get("UAV*", "") or "").strip().lower()
        human = str(row.get("Human", "") or "").strip().lower()
        ugv = str(row.get("UGV", "") or "").strip().lower()
        uav = str(row.get("UAV", "") or "").strip().lower()

        VALID_COLORS = {"red", "yellow", "green", "orange"}
        
        # Count valid performers (not red)
        performers = []
        if human_star in VALID_COLORS and human_star != "red":
            performers.append("Human*")
        if ugv_star in VALID_COLORS and ugv_star != "red":
            performers.append("UGV*")
        if uav_star in VALID_COLORS and uav_star != "red":
            performers.append("UAV*")
        
        # Count valid supporters (not red)
        supporters = []
        if human in VALID_COLORS and human != "red":
            supporters.append("Human")
        if ugv in VALID_COLORS and ugv != "red":
            supporters.append("UGV")
        if uav in VALID_COLORS and uav != "red":
            supporters.append("UAV")
        
        # Determine task type
        if len(supporters) > 0:
            # Has support available = interdependent
            interdependent += 1
        elif len(performers) == 1:
            # Only one performer, no support = single allocation independent
            single_allocation_independent += 1
        elif len(performers) > 1:
            # Multiple performers, no support = multiple allocation independent
            multiple_allocation_independent += 1
    
    total_tasks = len(df_bar)
    
    # Create horizontal stacked bar chart
    allocation_fig = go.Figure()
    
    allocation_fig.add_trace(go.Bar(
        name='Single Allocation Independent',
        y=['Task Allocation Types'],
        x=[single_allocation_independent],
        orientation='h',
        marker=dict(color='lightcoral'),
        text=[f'Single Allocation Independent: {single_allocation_independent} ({single_allocation_independent/total_tasks*100:.1f}%)' if total_tasks > 0 and single_allocation_independent > 0 else ''],
        textposition='inside',
        textfont=dict(color='black', size=12),
        hovertemplate='Single Allocation Independent<br>Count: %{x}<br>Percentage: ' + (f'{single_allocation_independent/total_tasks*100:.1f}%' if total_tasks > 0 else '0%') + '<extra></extra>'
    ))
    
    allocation_fig.add_trace(go.Bar(
        name='Multiple Allocation Independent',
        y=['Task Allocation Types'],
        x=[multiple_allocation_independent],
        orientation='h',
        marker=dict(color='lightskyblue'),
        text=[f'Multiple Allocation Independent: {multiple_allocation_independent} ({multiple_allocation_independent/total_tasks*100:.1f}%)' if total_tasks > 0 and multiple_allocation_independent > 0 else ''],
        textposition='inside',
        textfont=dict(color='black', size=12),
        hovertemplate='Multiple Allocation Independent<br>Count: %{x}<br>Percentage: ' + (f'{multiple_allocation_independent/total_tasks*100:.1f}%' if total_tasks > 0 else '0%') + '<extra></extra>'
    ))
    
    allocation_fig.add_trace(go.Bar(
        name='Interdependent (Support Available)',
        y=['Task Allocation Types'],
        x=[interdependent],
        orientation='h',
        marker=dict(color='lightgreen'),
        text=[f'Interdependent: {interdependent} ({interdependent/total_tasks*100:.1f}%)' if total_tasks > 0 and interdependent > 0 else ''],
        textposition='inside',
        textfont=dict(color='black', size=12),
        hovertemplate='Interdependent (Support Available)<br>Count: %{x}<br>Percentage: ' + (f'{interdependent/total_tasks*100:.1f}%' if total_tasks > 0 else '0%') + '<extra></extra>'
    ))
    
    allocation_fig.update_layout(
        title="Task Type Distribution",
        xaxis_title="Number of Tasks",
        barmode='stack',
        height=200,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # --- Agent Autonomy Analysis ---
    # Track tasks by agent and autonomy across the entire scenario
    agent_autonomy = {
        "Human*": {"autonomous": 0, "non_autonomous": 0},
        "UGV*": {"autonomous": 0, "non_autonomous": 0},
        "UAV*": {"autonomous": 0, "non_autonomous": 0}
    }
    
    # Process all tasks in sequence (not grouped by procedure)
    prev_performers = []
    
    for idx, row in df_bar.iterrows():
        # Get current task performers
        human_star = str(row.get("Human*", "") or "").strip().lower()
        ugv_star = str(row.get("UGV*", "") or "").strip().lower()
        uav_star = str(row.get("UAV*", "") or "").strip().lower()
        
        VALID_COLORS = {"red", "yellow", "green", "orange"}
        
        # Map agents to their colors for this task
        agent_colors = {
            "Human*": human_star,
            "UGV*": ugv_star,
            "UAV*": uav_star
        }
        
        current_performers = []
        if human_star in VALID_COLORS and human_star != "red":
            current_performers.append("Human*")
        if ugv_star in VALID_COLORS and ugv_star != "red":
            current_performers.append("UGV*")
        if uav_star in VALID_COLORS and uav_star != "red":
            current_performers.append("UAV*")
        
        # For each agent that can perform this task
        for agent in current_performers:
            # Orange tasks are always non-autonomous
            if agent_colors[agent] == "orange":
                agent_autonomy[agent]["non_autonomous"] += 1
            elif idx == 0 or agent not in prev_performers:
                # First task or agent couldn't perform previous task
                agent_autonomy[agent]["non_autonomous"] += 1
            else:
                # Agent could also perform previous task = autonomous (green or yellow only)
                agent_autonomy[agent]["autonomous"] += 1
        
        prev_performers = current_performers
    
    # Create horizontal stacked bar chart for agent autonomy
    autonomy_fig = go.Figure()
    
    agents = ["Human*", "UGV*", "UAV*"]
    colors_autonomous = {"Human*": "lightgreen", "UGV*": "lightgreen", "UAV*": "lightgreen"}
    colors_non_autonomous = {"Human*": "lightcoral", "UGV*": "lightcoral", "UAV*": "lightcoral"}
    
    for agent in agents:
        autonomous = agent_autonomy[agent]["autonomous"]
        non_autonomous = agent_autonomy[agent]["non_autonomous"]
        total = autonomous + non_autonomous
        
        # Non-autonomous tasks (first section)
        autonomy_fig.add_trace(go.Bar(
            name=f'{agent} Non-Autonomous',
            y=[agent],
            x=[non_autonomous],
            orientation='h',
            marker=dict(color=colors_non_autonomous[agent]),
            text=[f'Non-Autonomous: {non_autonomous} ({non_autonomous/total*100:.1f}%)' if total > 0 and non_autonomous > 0 else ''],
            textposition='inside',
            textfont=dict(color='black', size=11),
            hovertemplate=f'{agent}<br>Non-Autonomous Tasks: {non_autonomous}<br>Percentage: ' + (f'{non_autonomous/total*100:.1f}%' if total > 0 else '0%') + '<extra></extra>',
            showlegend=False
        ))
        
        # Autonomous tasks (second section)
        autonomy_fig.add_trace(go.Bar(
            name=f'{agent} Autonomous',
            y=[agent],
            x=[autonomous],
            orientation='h',
            marker=dict(color=colors_autonomous[agent]),
            text=[f'Autonomous: {autonomous} ({autonomous/total*100:.1f}%)' if total > 0 and autonomous > 0 else ''],
            textposition='inside',
            textfont=dict(color='black', size=11),
            hovertemplate=f'{agent}<br>Autonomous Tasks: {autonomous}<br>Percentage: ' + (f'{autonomous/total*100:.1f}%' if total > 0 else '0%') + '<extra></extra>',
            showlegend=False
        ))
    
    autonomy_fig.update_layout(
        title="Agent Autonomy: Task Continuity Across Entire Scenario",
        xaxis_title="Number of Tasks",
        yaxis_title="Agent",
        barmode='stack',
        height=300,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        margin=dict(l=100, r=50, t=50, b=50)
    )

    return workflow_fig, bar_fig_whole_scenario, bar_fig, allocation_fig, autonomy_fig


def ensure_all_columns(df, columns):
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df

@app.callback(
    Output("table-wrapper", "children"),
    Output("save-confirmation", "children"),
    Output("download-csv", "data"),
    Input("responsibility-table", "data"),
    Input("save-button", "n_clicks"),
    Input("upload-data", "contents"),
    Input("add-row-button", "n_clicks"),
    Input("copy-down-button", "n_clicks"),
    State("responsibility-table", "active_cell"),
    State("upload-data", "filename"),
    prevent_initial_call=True
)
def handle_table(data, save_clicks, upload_contents, add_row_clicks, copy_clicks, active_cell, upload_filename):
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]

    color_options = ["red", "yellow", "green", "orange"]
    dropdowns = {
        col: {
            'options': [{'label': c.capitalize(), 'value': c} for c in color_options]
        }
        for col in agent_columns
    }

    save_message = ""

    # Case 1: Save button clicked
    if triggered == "save-button":
        df = pd.DataFrame(data)
        df = ensure_all_columns(df, [col["id"] for col in columns])
        save_message = f"✅ Table downloaded as table_hat_game.csv"

    # Case 2: File uploaded
    elif triggered == "upload-data" and upload_contents is not None:
        content_type, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            df = ensure_all_columns(df, [col["id"] for col in columns])
            save_message = f"✅ Loaded {upload_filename}"
        except Exception as e:
            return dash.no_update, f"⚠️ Error loading file: {str(e)}", None

    # Case 3: Add Row button clicked
    elif triggered == "add-row-button":
        df = pd.DataFrame(data)
        df = ensure_all_columns(df, [col["id"] for col in columns])
        new_row = {col: "" for col in df.columns}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Case 4: Copy Down clicked
    elif triggered == "copy-down-button":
        df = pd.DataFrame(data)
        df = ensure_all_columns(df, [col["id"] for col in columns])
        if active_cell and "row" in active_cell and "column_id" in active_cell:
            row = active_cell["row"]
            col = active_cell["column_id"]
            if row is not None and col is not None and row + 1 < len(df):
                df.at[row + 1, col] = df.at[row, col]

    # Case 5: Table edited directly
    else:
        df = pd.DataFrame(data)
        df = ensure_all_columns(df, [col["id"] for col in columns])

    updated_table = dash_table.DataTable(
        id='responsibility-table',
        columns=columns,
        data=df.assign(Row=lambda x: x.index + 1).to_dict("records"),
        editable=True,
        dropdown=dropdowns,
        style_data_conditional=style_table(df),
        style_cell={'textAlign': 'left', 'padding': '5px', 'whiteSpace': 'normal'},
        style_cell_conditional=[
            {
                'if': {'column_id': 'Human*'},
                'borderLeft': '3px solid black'
            },
            {
                'if': {'column_id': 'Human'},
                'borderRight': '3px solid black'
            },
            {
                'if': {'column_id': 'TARS'},
                'borderRight': '2px solid black'
            }
        ],
        style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0'},
        style_table={'overflowX': 'auto', 'border': '1px solid lightgrey'},
        row_deletable=True,
    )
    
    # Trigger download if save button was clicked
    download_data = None
    if triggered == "save-button":
        df_to_download = pd.DataFrame(data)
        df_to_download = ensure_all_columns(df_to_download, [col["id"] for col in columns])
        download_data = dcc.send_data_frame(df_to_download.to_csv, "table_hat_game.csv", index=False)
    
    return updated_table, save_message, download_data


# Expose server for gunicorn
server = app.server

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)