import dash
from dash import html, dcc, dash_table, Input, Output, State, callback_context
import plotly.graph_objects as go
import pandas as pd
import os
import base64
import io

app = dash.Dash(__name__)
DATA_FILE = "table_data_with_opd.csv"
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)

color_options = ["red", "yellow", "green", "orange"]
dropdowns = {
    col: {
        'options': [{'label': c.capitalize(), 'value': c} for c in color_options]
    }
    for col in ["Human*", "TARS", "TARS*", "Human"]
}
def style_table(df):
    styles = []
    for i, row in df.iterrows():
        for col in editable_columns:
            value = row[col]
            color = value.lower() if isinstance(value, str) else "white"
            styles.append({
                'if': {'row_index': i, 'column_id': col},
                'backgroundColor': color if color != "white" else "#ffffff",
                'color': color,  # Hide the text
                'textAlign': 'center'
            })
    return styles
df = df.assign(Row=lambda x: x.index + 1)
editable_columns = ["Human*", "TARS", "TARS*", "Human"]
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

columns=[
    {"name": "Row", "id": "Row", "editable": False},
    {"name": "Procedure", "id": "Procedure", "editable": True},
    {"name": "Task", "id": "Task", "editable": True},
    {"name": "Human*", "id": "Human*", "editable": True, "presentation": "dropdown"},
    {"name": "TARS", "id": "TARS", "editable": True, "presentation": "dropdown"},
    {"name": "TARS*", "id": "TARS*", "editable": True, "presentation": "dropdown"},
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


def build_interdependence_figures(df, highlight_track=None):
    agents = ["HUMAN*", "TARS", "TARS*", "HUMAN"]
    VALID_COLORS = {"red", "yellow", "green", "orange", "black", "grey"}

    figures = {}

    for procedure, proc_df in df.groupby("Procedure"):
        tasks = proc_df["Task"].tolist()
        height = 300 + len(tasks) * 100
        dots = []
        grey_to_black_arrows = []
        black_to_black_arrows = []
        most_reliable_track = []
        dashed_arrows_to_highlight = []

        proc_df = proc_df.reset_index(drop=True)
        proc_df["task_idx"] = proc_df.index

        # Step 1: Add points
        for idx, row in proc_df.iterrows():
            task_idx = row["task_idx"]
            black_points = []

            agent_colors = {
                "HUMAN*": row["Human*"].strip().lower() if isinstance(row["Human*"], str) else "",
                "TARS*": row["TARS*"].strip().lower() if isinstance(row["TARS*"], str) else "",
            }

            # Determine most reliable agent
            chosen_agent = None
            if agent_colors["HUMAN*"] == "green":
                chosen_agent = "HUMAN*"
            elif agent_colors["TARS*"] == "green":
                chosen_agent = "TARS*"
            elif agent_colors["HUMAN*"] == "yellow":
                chosen_agent = "HUMAN*"
            elif agent_colors["TARS*"] == "yellow":
                chosen_agent = "TARS*"

            if chosen_agent:
                most_reliable_track.append((task_idx, chosen_agent))

            for col in ["Human*", "TARS*"]:
                val = row[col].strip().lower() if isinstance(row[col], str) else ""
                if val in VALID_COLORS:
                    dots.append({"task": task_idx, "agent": col.upper(), "color": val})
                    if val != "red":
                        black_points.append(col.upper())

            for col in ["Human", "TARS"]:
                val = row[col].strip().lower() if isinstance(row[col], str) else ""
                if val in VALID_COLORS:
                    agent = col.upper()
                    dots.append({"task": task_idx, "agent": agent, "color": val})

                    if (agent == "TARS"
                        and "HUMAN*" in black_points
                        and row["Human*"].strip().lower() != "red"
                        and row["TARS"].strip().lower() != "red"):
                        dashed_arrow = {
                            "start_agent": "TARS",
                            "end_agent": "HUMAN*",
                            "task": task_idx
                        }
                        grey_to_black_arrows.append(dashed_arrow)
                        if highlight_track == "most_reliable" and (task_idx, "HUMAN*") in most_reliable_track:
                            dashed_arrows_to_highlight.append(dashed_arrow)

                    elif (agent == "HUMAN"
                          and "TARS*" in black_points
                          and row["TARS*"].strip().lower() != "red"
                          and row["Human"].strip().lower() != "red"):
                        dashed_arrow = {
                            "start_agent": "HUMAN",
                            "end_agent": "TARS*",
                            "task": task_idx
                        }
                        grey_to_black_arrows.append(dashed_arrow)
                        if highlight_track == "most_reliable" and (task_idx, "TARS*") in most_reliable_track:
                            dashed_arrows_to_highlight.append(dashed_arrow)

        # Step 2: Black-to-black transitions
        for i in range(len(tasks) - 1):
            current_blacks = [d for d in dots if d["task"] == i and d["agent"] in ["HUMAN*", "TARS*"] and d["color"] != "red"]
            next_blacks = [d for d in dots if d["task"] == i + 1 and d["agent"] in ["HUMAN*", "TARS*"] and d["color"] != "red"]
            for curr in current_blacks:
                for nxt in next_blacks:
                    black_to_black_arrows.append({
                        "start_task": i,
                        "start_agent": curr["agent"],
                        "end_task": i + 1,
                        "end_agent": nxt["agent"]
                    })

        # Step 3: Create figure
        agent_pos = {agent: i for i, agent in enumerate(agents)}
        fig = go.Figure()

        for dot in dots:
            row = proc_df.iloc[dot["task"]]  # get the row by task index
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
                marker=dict(size=20, color=dot["color"]),
                showlegend=False,
                hoverinfo="text",
                hovertext=hover_text
            ))


        for arrow in grey_to_black_arrows:
            is_highlighted = highlight_track == "most_reliable" and arrow in dashed_arrows_to_highlight
            fig.add_shape(
                type="line",
                x0=agent_pos[arrow["start_agent"]],
                y0=arrow["task"],
                x1=agent_pos[arrow["end_agent"]],
                y1=arrow["task"],
                line=dict(
                    color="crimson" if is_highlighted else "black",
                    width=4 if is_highlighted else 2,
                    dash="dot"
                )
            )

        for arrow in black_to_black_arrows:
            is_highlighted = False
            if highlight_track == "human_baseline":
                is_highlighted = arrow["start_agent"] == "HUMAN*" and arrow["end_agent"] == "HUMAN*"
            elif highlight_track == "most_reliable":
                is_highlighted = (arrow["start_task"], arrow["start_agent"]) in most_reliable_track and \
                                 (arrow["end_task"], arrow["end_agent"]) in most_reliable_track

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
    agents = ["HUMAN*", "TARS", "TARS*", "HUMAN"]
    VALID_COLORS = {"red", "yellow", "green", "orange", "black", "grey"}
    tasks = df["Task"].tolist()
    height = 600 + len(tasks) * 100
    dots = []
    grey_to_black_arrows = []
    black_to_black_arrows = []

    df = df.reset_index(drop=True)
    df["task_idx"] = df.index  # Unique index across all tasks

    # Track the most reliable agent at each step
    most_reliable_track = []
    dashed_arrows_to_highlight = []

    for idx, row in df.iterrows():
        task_idx = row["task_idx"]
        black_points = []

        agent_colors = {
            "HUMAN*": row["Human*"].strip().lower() if isinstance(row["Human*"], str) else "",
            "TARS*": row["TARS*"].strip().lower() if isinstance(row["TARS*"], str) else "",
        }

        # Determine most reliable path based on green/yellow preference for HUMAN*
        chosen_agent = None
        if agent_colors["HUMAN*"] == "green":
            chosen_agent = "HUMAN*"
        elif agent_colors["TARS*"] == "green":
            chosen_agent = "TARS*"
        elif agent_colors["HUMAN*"] == "yellow":
            chosen_agent = "HUMAN*"
        elif agent_colors["TARS*"] == "yellow":
            chosen_agent = "TARS*"

        if chosen_agent:
            most_reliable_track.append((task_idx, chosen_agent))

        for col in ["Human*", "TARS*"]:
            val = row[col].strip().lower() if isinstance(row[col], str) else ""
            if val in VALID_COLORS:
                dots.append({"task": task_idx, "agent": col.upper(), "color": val})
                if val != "red":
                    black_points.append(col.upper())

        for col in ["Human", "TARS"]:
            val = row[col].strip().lower() if isinstance(row[col], str) else ""
            if val in VALID_COLORS:
                agent = col.upper()
                dots.append({"task": task_idx, "agent": agent, "color": val})

                if (
                    agent == "TARS"
                    and "HUMAN*" in black_points
                    and row["Human*"].strip().lower() != "red"
                    and row["TARS"].strip().lower() != "red"
                ):
                    dashed_arrow = {
                        "start_agent": "TARS",
                        "end_agent": "HUMAN*",
                        "task": task_idx
                    }
                    grey_to_black_arrows.append(dashed_arrow)
                    if highlight_track == "most_reliable" and (task_idx, "HUMAN*") in most_reliable_track:
                        dashed_arrows_to_highlight.append(dashed_arrow)

                elif (
                    agent == "HUMAN"
                    and "TARS*" in black_points
                    and row["TARS*"].strip().lower() != "red"
                    and row["Human"].strip().lower() != "red"
                ):
                    dashed_arrow = {
                        "start_agent": "HUMAN",
                        "end_agent": "TARS*",
                        "task": task_idx
                    }
                    grey_to_black_arrows.append(dashed_arrow)
                    if highlight_track == "most_reliable" and (task_idx, "TARS*") in most_reliable_track:
                        dashed_arrows_to_highlight.append(dashed_arrow)

    for i in range(len(df) - 1):
        current_blacks = [d for d in dots if d["task"] == i and d["agent"] in ["HUMAN*", "TARS*"] and d["color"] != "red"]
        next_blacks = [d for d in dots if d["task"] == i + 1 and d["agent"] in ["HUMAN*", "TARS*"] and d["color"] != "red"]
        for curr in current_blacks:
            for nxt in next_blacks:
                black_to_black_arrows.append({
                    "start_task": i,
                    "start_agent": curr["agent"],
                    "end_task": i + 1,
                    "end_agent": nxt["agent"]
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
            marker=dict(size=20, color=dot["color"]),
            showlegend=False,
            hoverinfo="text",
            hovertext=hover_text
        ))


    for arrow in grey_to_black_arrows:
        is_highlighted = highlight_track == "most_reliable" and arrow in dashed_arrows_to_highlight
        fig.add_shape(
            type="line",
            x0=agent_pos[arrow["start_agent"]],
            y0=arrow["task"],
            x1=agent_pos[arrow["end_agent"]],
            y1=arrow["task"],
            line=dict(
                color="crimson" if is_highlighted else "black",
                width=4 if is_highlighted else 2,
                dash="dot"
            )
        )

    for arrow in black_to_black_arrows:
        is_highlighted = False
        if highlight_track == "human_baseline":
            is_highlighted = arrow["start_agent"] == "HUMAN*" and arrow["end_agent"] == "HUMAN*"
        elif highlight_track == "most_reliable":
            is_highlighted = (arrow["start_task"], arrow["start_agent"]) in most_reliable_track and \
                             (arrow["end_task"], arrow["end_agent"]) in most_reliable_track

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
    # The table goes here
    html.H2("Task Hierarchy | Team Alternatives | Capacity Assessment"),
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
                html.Button("💾 Save Table", disabled=False, id="save-button", n_clicks=0)
            ], style={"marginLeft": "auto"})  # pushes this div to the right
        ], style={"display": "flex", "width": "100%"}),
        html.Div(id="save-confirmation", style={"marginTop": "10px", "fontStyle": "italic"})
    ]),

    html.H2("Workflow graph", style={"textAlign": "center"}),

    # Dropdown menu to select procedure
    html.Div([
        dcc.Dropdown(
            id="procedure-dropdown",
            options=[{"label": proc, "value": proc} for proc in df["Procedure"].unique()],
            value=df["Procedure"].unique()[0],
            clearable=True,
            style={"width": "50%", "margin": "0 auto"}
        )
    ]),

    dcc.RadioItems(
    id="highlight-selector",
    options=[
        {"label": "No highlight", "value": "none"},
        {"label": "Baseline SPO", "value": "human_baseline"},
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

], style={"fontFamily": "'Roboto', 'Helvetica', 'Arial', sans-serif"})

@app.callback(
    Output("interdependence-graph", "figure"),
    Input("procedure-dropdown", "value"),
    Input("highlight-selector", "value"),
    State("responsibility-table", "data")
)
def update_graph(procedure, highlight_track, data):
    df = pd.DataFrame(data)
    if procedure is None:
        # 👇 Combine all procedures into one graph
        return build_combined_interdependence_figure(df, highlight_track)
    if highlight_track == "none":
        highlight_track = None

    
    figures = build_interdependence_figures(df, highlight_track)
    return figures.get(procedure, go.Figure())


@app.callback(
    Output("table-wrapper", "children"),
    Output("save-confirmation", "children"),
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
        for col in ["Human*", "TARS", "TARS*", "Human"]
    }

    save_message = ""

    # Case 1: Save button clicked
    if triggered == "save-button":
        df = pd.DataFrame(data)
        df.to_csv(DATA_FILE, index=False)
        save_message = f"✅ Table saved to {DATA_FILE}"

    # Case 2: File uploaded
    elif triggered == "upload-data" and upload_contents is not None:
        content_type, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            save_message = f"✅ Loaded {upload_filename}"
        except Exception as e:
            return dash.no_update, f"⚠️ Error loading file: {str(e)}"

    # Case 3: Add Row button clicked
    elif triggered == "add-row-button":
        df = pd.DataFrame(data)
        new_row = {col: "" for col in df.columns}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Case 4: Copy Down clicked
    elif triggered == "copy-down-button":
        df = pd.DataFrame(data)
        if active_cell and "row" in active_cell and "column_id" in active_cell:
            row = active_cell["row"]
            col = active_cell["column_id"]
            if row is not None and col is not None and row + 1 < len(df):
                df.at[row + 1, col] = df.at[row, col]

    # Case 5: Table edited directly
    else:
        df = pd.DataFrame(data)

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
    return updated_table, save_message


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)