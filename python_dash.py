import dash
from dash import html, dcc, dash_table, Input, Output, State, callback_context
import plotly.graph_objects as go
import pandas as pd
import os

app = dash.Dash(__name__)
DATA_FILE = "table_data.csv"
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
editable_columns = ["Human*", "TARS", "TARS*", "Human"]
table = dash_table.DataTable(
    id='responsibility-table',
    columns=[
        {"name": col, "id": col, "editable": col in editable_columns}
        for col in df.columns
    ],
    data=df.to_dict('records'),
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

def build_interdependence_figure(df):
    agents = ["HUMAN*", "TARS", "TARS*", "HUMAN"]
    tasks = df["Task"].tolist()

    dots = []
    grey_to_black_arrows = []
    black_to_black_arrows = []

    VALID_COLORS = {"red", "yellow", "green", "orange", "black", "grey"}

    # Step 1: Add points with actual cell colors
    for idx, row in df.iterrows():
        black_points = []

        for col in ["Human*", "TARS*"]:
            val = row[col].strip().lower() if isinstance(row[col], str) else ""
            if val in VALID_COLORS:
                dots.append({"task": idx, "agent": col.upper(), "color": val})
                black_points.append(col.upper())

        for col in ["Human", "TARS"]:
            val = row[col].strip().lower() if isinstance(row[col], str) else ""
            if val in VALID_COLORS:
                agent = col.upper()
                dots.append({"task": idx, "agent": agent, "color": val})
                if (
                    agent == "TARS"
                    and "HUMAN*" in black_points
                    and row["Human*"].strip().lower() != "red"
                    and row["TARS"].strip().lower() != "red"
                ):
                    grey_to_black_arrows.append({
                        "start_agent": "TARS",
                        "end_agent": "HUMAN*",
                        "task": idx
                    })
                elif (
                    agent == "HUMAN"
                    and "TARS*" in black_points
                    and row["TARS*"].strip().lower() != "red"
                    and row["Human"].strip().lower() != "red"
                ):
                    grey_to_black_arrows.append({
                        "start_agent": "HUMAN",
                        "end_agent": "TARS*",
                        "task": idx
                    })

    # Step 2: Add black-to-black transitions (same logic)
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

    # Step 3: Plot with Plotly
    agent_pos = {agent: i for i, agent in enumerate(agents)}
    fig = go.Figure()

    for dot in dots:
        fig.add_trace(go.Scatter(
            x=[agent_pos[dot["agent"]]],
            y=[dot["task"]],
            mode="markers",
            marker=dict(size=20, color=dot["color"]),
            showlegend=False
        ))

    for arrow in grey_to_black_arrows:
        fig.add_shape(
            type="line",
            x0=agent_pos[arrow["start_agent"]],
            y0=arrow["task"],
            x1=agent_pos[arrow["end_agent"]],
            y1=arrow["task"],
            line=dict(color="black", width=2, dash="dot")
        )


    for arrow in black_to_black_arrows:
        fig.add_annotation(
            x=agent_pos[arrow["end_agent"]],
            y=arrow["end_task"],
            ax=agent_pos[arrow["start_agent"]],
            ay=arrow["start_task"],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1,
            arrowwidth=2, opacity=0.9
        )

    fig.update_layout(
        title="",
        xaxis=dict(
            tickvals=list(agent_pos.values()),
            ticktext=list(agent_pos.keys()),
            title="Agent",
            showgrid=True,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            tickvals=list(range(len(tasks))),
            ticktext=tasks,
            title="Task",
            autorange="reversed",
            showgrid=True,
            gridcolor='lightgrey'
        ),
        height=600,
        margin=dict(l=200, r=50, t=50, b=50),
        plot_bgcolor='white',
        paper_bgcolor ='white',
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
                html.Button("📂 Load Table", id="load-button", n_clicks=0),
                html.Button("➕ Add Row", id="add-row-button", n_clicks=0),
                html.Button("⬇️ Copy Cell Down", id="copy-down-button", n_clicks=0),
            ], style={"display": "flex", "gap": "10px"}),

            # Right-aligned Save button
            html.Div([
                html.Button("💾 Save Table", id="save-button", n_clicks=0)
            ], style={"marginLeft": "auto"})  # pushes this div to the right
        ], style={"display": "flex", "width": "100%"}),
        html.Div(id="save-confirmation", style={"marginTop": "10px", "fontStyle": "italic"})
    ]),

    html.H2("Workflow graph", style={"textAlign": "center"}),
    html.Div([
    html.Div("Team Alternative 1", style={
        "width": "50%",
        "display": "inline-block",
        "textAlign": "center",
        "marginTop": "10px",
        "marginLeft": "150px"
    }),
    html.Div("Team Alternative 2", style={
        "width": "50%",
        "display": "inline-block",
        "textAlign": "center",
        "marginTop": "10px"
    }),
    ], style={"display": "flex", "width": "100%"}),
    dcc.Graph(id="interdependence-graph"),
], style={"fontFamily": "'Roboto', 'Helvetica', 'Arial', sans-serif"})

@app.callback(
    Output("table-wrapper", "children"),
    Output("save-confirmation", "children"),
    Output("interdependence-graph", "figure"),
    Input("responsibility-table", "data"),
    Input("save-button", "n_clicks"),
    Input("load-button", "n_clicks"),
    Input("add-row-button", "n_clicks"),
    Input("copy-down-button", "n_clicks"),  # 👈 Added copy-down input
    State("responsibility-table", "active_cell"),  # 👈 To know which cell is selected
    prevent_initial_call=True
)
def handle_table(data, save_clicks, load_clicks, add_row_clicks, copy_clicks, active_cell):
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
        interdependence_fig = build_interdependence_figure(df)

    # Case 2: Load button clicked
    elif triggered == "load-button":
        if not os.path.exists(DATA_FILE):
            return dash.no_update, "⚠️ File not found."
        df = pd.read_csv(DATA_FILE)

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
        data=df.to_dict('records'),
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
    interdependence_fig = build_interdependence_figure(df)

    return updated_table, save_message, interdependence_fig


if __name__ == "__main__":
    app.run(debug=True)