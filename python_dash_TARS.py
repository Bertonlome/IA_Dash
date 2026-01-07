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

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=['assets/styles.css'])

# Navigation menu component
def create_navbar(current_pathname="/"):
    return html.Div([
        html.Div([
            dcc.Link("Interdependence Analysis Table", href="/", className="nav-link" + (" active" if current_pathname == "/" else "")),
            dcc.Link("Assumptions", href="/assumptions", className="nav-link" + (" active" if current_pathname == "/assumptions" else ""))
        ], style={"textAlign": "left", "marginBottom": "20px"}),
    ])

DATA_FILE = "./V6/IA_V6.csv"
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)

color_options = ["red", "yellow", "green", "orange"]

# Updated dropdown options for all categorical columns
dropdowns = {
    # Original color-based columns
    "Human*": {
        'options': [{'label': c.capitalize(), 'value': c} for c in color_options]
    },
    "TARS": {
        'options': [{'label': c.capitalize(), 'value': c} for c in color_options]
    },
    "TARS*": {
        'options': [{'label': c.capitalize(), 'value': c} for c in color_options]
    },
    "Human": {
        'options': [{'label': c.capitalize(), 'value': c} for c in color_options]
    },
    # New categorical columns
    "Category": {
        'options': [
            {'label': 'NORM', 'value': 'NORM'},
            {'label': 'ABNORM', 'value': 'ABNORM'},
            {'label': 'EMER', 'value': 'EMER'}
        ]
    },
    "Type": {
        'options': [
            {'label': 'SOP', 'value': 'SOP'},
            {'label': 'Checklist', 'value': 'Checklist'},
            {'label': 'Memory Item', 'value': 'Memory Item'}
        ]
    },
    "Constraint Type": {
        'options': [
            {'label': 'Soft-time', 'value': 'Soft-time'},
            {'label': 'Hard-time', 'value': 'Hard-time'}
        ]
    },
    "Execution Type": {
        'options': [
            {'label': 'One-off', 'value': 'One-off'},
            {'label': 'Continuous', 'value': 'Continuous'},
            {'label': 'Cyclic', 'value': 'Cyclic'},
            {'label': 'Monitoring', 'value': 'Monitoring'}
        ]
    }
}
def style_table(df):
    styles = []
    # Only apply color styling to the color-based columns
    color_columns = ["Human*", "TARS", "TARS*", "Human"]
    
    for i, row in df.iterrows():
        for col in color_columns:
            if col in df.columns:
                value = row[col]
                if isinstance(value, str) and value.strip().lower() in ["red", "yellow", "green", "orange"]:
                    color = value.strip().lower()
                    styles.append({
                        'if': {'row_index': i, 'column_id': col},
                        'backgroundColor': color,
                        'color': color,  # Text color matches background (invisible text)
                        'textAlign': 'center',
                        'fontWeight': 'bold'
                    })
    return styles

# No need to assign Row since it already exists in the CSV
# df = df.assign(Row=lambda x: x.index + 1)

# Updated editable columns to include new fields
editable_columns = [
    "Procedure", "Category", "Type", "Task Object", "Value", 
    "Human*", "TARS", "TARS*", "Human", 
    "Observability", "Predictability", "Directability", 
    "Information Requirement", "Constraint Type", "Time constraint (in s)", "Execution Type"
]

table = dash_table.DataTable(
    id='responsibility-table',
    columns=[
        {"name": col, "id": col, "editable": col in editable_columns}
        for col in df.columns
    ],
    data=df.to_dict("records"),  # Remove the .assign() since Row already exists
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
    {"name": "Category", "id": "Category", "editable": True, "presentation": "dropdown"},
    {"name": "Type", "id": "Type", "editable": True, "presentation": "dropdown"},
    {"name": "Task Object", "id": "Task Object", "editable": True},
    {"name": "Value", "id": "Value", "editable": True},
    {"name": "Human*", "id": "Human*", "editable": True, "presentation": "dropdown"},
    {"name": "TARS", "id": "TARS", "editable": True, "presentation": "dropdown"},
    {"name": "TARS*", "id": "TARS*", "editable": True, "presentation": "dropdown"},
    {"name": "Human", "id": "Human", "editable": True, "presentation": "dropdown"},
    {"name": "Observability", "id": "Observability", "editable": True},
    {"name": "Predictability", "id": "Predictability", "editable": True},
    {"name": "Directability", "id": "Directability", "editable": True},
    {"name": "Information Requirement", "id": "Information Requirement", "editable": True},
    {"name": "Constraint Type", "id": "Constraint Type", "editable": True, "presentation": "dropdown"},
    {"name": "Time constraint (in s)", "id": "Time constraint (in s)", "editable": True, "type": "numeric"},
    {"name": "Execution Type", "id": "Execution Type", "editable": True, "presentation": "dropdown"},
]

def wrap_text(text, max_width=30):
    import textwrap
    if not isinstance(text, str):
        return ""
    return '<br>'.join(textwrap.wrap(text, width=max_width))


def build_performers_only_figures(df, highlight_track=None):
    """Build workflow graphs showing only HUMAN* and TARS* performers (no supporters)"""
    agents = ["HUMAN*", "TARS*"]
    VALID_COLORS = {"red", "yellow", "green", "orange", "black", "grey"}

    figures = {}

    for procedure, proc_df in df.groupby("Procedure"):
        tasks = proc_df["Task Object"].tolist()
        height = 300 + len(tasks) * 100
        dots = []
        grey_to_black_arrows = []
        black_to_black_arrows = []
        most_reliable_track = []
        dashed_arrows_to_highlight = []

        proc_df = proc_df.reset_index(drop=True)
        proc_df["task_idx"] = proc_df.index

        # Step 1: Add points for performers only
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

            # Add dots for each agent column showing their own capabilities
            # HUMAN* column: HUMAN* performer (circle)
            # TARS* column: TARS* performer (circle) + TARS supporter (square)
            
            # HUMAN* performer in HUMAN* column
            human_performer_val = row["Human*"].strip().lower() if isinstance(row["Human*"], str) else ""
            if human_performer_val in VALID_COLORS and human_performer_val != "red":
                dots.append({
                    "task": task_idx,
                    "agent": "HUMAN*",
                    "shape": "circle",
                    "color": human_performer_val,
                    "role": "performer"
                })
                black_points.append("HUMAN*")
            
            # TARS* performer in TARS* column
            tars_performer_val = row["TARS*"].strip().lower() if isinstance(row["TARS*"], str) else ""
            if tars_performer_val in VALID_COLORS and tars_performer_val != "red":
                dots.append({
                    "task": task_idx,
                    "agent": "TARS*",
                    "shape": "circle",
                    "color": tars_performer_val,
                    "role": "performer"
                })
                black_points.append("TARS*")
            
            # TARS supporter in TARS* column (as square)
            tars_supporter_val = row["TARS"].strip().lower() if isinstance(row["TARS"], str) and row["TARS"] else ""
            if tars_supporter_val in VALID_COLORS and tars_supporter_val != "red":
                dots.append({
                    "task": task_idx,
                    "agent": "TARS*",
                    "shape": "square",
                    "color": tars_supporter_val,
                    "role": "supporter"
                })
            
            # HUMAN supporter in HUMAN* column (as square)
            human_supporter_val = row["Human"].strip().lower() if isinstance(row["Human"], str) and row["Human"] else ""
            if human_supporter_val in VALID_COLORS and human_supporter_val != "red":
                dots.append({
                    "task": task_idx,
                    "agent": "HUMAN*",
                    "shape": "square",
                    "color": human_supporter_val,
                    "role": "supporter"
                })

            # Add dashed arrows from supporters to performers (even though supporters aren't shown)
            # TARS -> HUMAN*
            tars_val = row["TARS"].strip().lower() if isinstance(row["TARS"], str) else ""
            human_val = row["Human"].strip().lower() if isinstance(row["Human"], str) else ""
            
            if ("HUMAN*" in black_points and 
                row["Human*"].strip().lower() != "red" and 
                tars_val != "" and tars_val != "red"):
                dashed_arrow = {
                    "start_agent": "TARS*",  # TARS supporter helps HUMAN* performer
                    "end_agent": "HUMAN*",
                    "task": task_idx,
                    "is_support": True  # Flag to indicate this is a support relationship
                }
                grey_to_black_arrows.append(dashed_arrow)
                if highlight_track == "most_reliable" and (task_idx, "HUMAN*") in most_reliable_track:
                    dashed_arrows_to_highlight.append(dashed_arrow)

            # HUMAN -> TARS*
            if ("TARS*" in black_points and 
                row["TARS*"].strip().lower() != "red" and 
                human_val != "" and human_val != "red"):
                dashed_arrow = {
                    "start_agent": "HUMAN*",  # HUMAN supporter helps TARS* performer
                    "end_agent": "TARS*",
                    "task": task_idx,
                    "is_support": True
                }
                grey_to_black_arrows.append(dashed_arrow)
                if highlight_track == "most_reliable" and (task_idx, "TARS*") in most_reliable_track:
                    dashed_arrows_to_highlight.append(dashed_arrow)

        # Step 2: Black-to-black transitions between performers (circles only)
        for i in range(len(tasks) - 1):
            current_blacks = [d for d in dots if d["task"] == i and d["agent"] in ["HUMAN*", "TARS*"] and d["shape"] == "circle"]
            next_blacks = [d for d in dots if d["task"] == i + 1 and d["agent"] in ["HUMAN*", "TARS*"] and d["shape"] == "circle"]
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
            row = proc_df.iloc[dot["task"]]
            hover_text = (
                f"<b>Task:</b> {wrap_text(row['Task Object'])}<br>"
                f"<b>Agent:</b> {dot['agent']}<br>"
                f"<b>Role:</b> {dot['role'].capitalize()}<br><br>"
                f"<b>Observability:</b><br>{wrap_text(row.get('Observability', ''))}<br><br>"
                f"<b>Predictability:</b><br>{wrap_text(row.get('Predictability', ''))}<br><br>"
                f"<b>Directability:</b><br>{wrap_text(row.get('Directability', ''))}"
            )

            if dot["shape"] == "circle":
                # Draw performer circle
                fig.add_trace(go.Scatter(
                    x=[agent_pos[dot["agent"]]],
                    y=[dot["task"]],
                    mode="markers",
                    marker=dict(size=30, color=dot["color"]),
                    showlegend=False,
                    hoverinfo="text",
                    hovertext=hover_text
                ))
            else:  # square
                # Draw black border square
                fig.add_trace(go.Scatter(
                    x=[agent_pos[dot["agent"]]],
                    y=[dot["task"]],
                    mode="markers",
                    marker=dict(
                        size=20,
                        color="black",
                        symbol="square"
                    ),
                    showlegend=False,
                    hoverinfo="skip"
                ))
                # Draw colored square on top
                fig.add_trace(go.Scatter(
                    x=[agent_pos[dot["agent"]]],
                    y=[dot["task"]],
                    mode="markers",
                    marker=dict(
                        size=17,
                        color=dot["color"],
                        symbol="square"
                    ),
                    showlegend=False,
                    hoverinfo="text",
                    hovertext=hover_text
                ))

        # Add dashed horizontal lines for support relationships
        # Group arrows by task to detect overlaps
        arrows_by_task = {}
        for arrow in grey_to_black_arrows:
            if arrow.get("is_support"):
                task = arrow["task"]
                if task not in arrows_by_task:
                    arrows_by_task[task] = []
                arrows_by_task[task].append(arrow)
        
        for task, task_arrows in arrows_by_task.items():
            # If there are 2 arrows at the same task, offset them vertically
            if len(task_arrows) == 2:
                offsets = [-0.08, 0.08]  # Small vertical offset
            else:
                offsets = [0] * len(task_arrows)
            
            for arrow, offset in zip(task_arrows, offsets):
                is_highlighted = highlight_track == "most_reliable" and arrow in dashed_arrows_to_highlight
                # Draw dashed line from start_agent to end_agent with offset
                fig.add_shape(
                    type="line",
                    x0=agent_pos[arrow["start_agent"]],
                    y0=arrow["task"] + offset,
                    x1=agent_pos[arrow["end_agent"]],
                    y1=arrow["task"] + offset,
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
                opacity=0.9,
                standoff=18,  # Stop arrow at end circle edge (circle radius=15, add margin)
                startstandoff=18  # Start arrow from start circle edge
            )

        fig.update_layout(
            title=f"Workflow for {procedure} (Performers Only)",
            xaxis=dict(
                tickvals=list(agent_pos.values()),
                ticktext=list(agent_pos.keys()),
                title="Agent",
                showgrid=True,
                gridcolor='lightgrey',
                range=[-0.5, 1.5]  # Constrain x-axis to keep agents closer
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


def build_performers_only_combined_figure(df, highlight_track=None):
    """Build combined workflow graph showing only HUMAN* and TARS* performers (no supporters)"""
    agents = ["HUMAN*", "TARS*"]
    VALID_COLORS = {"red", "yellow", "green", "orange", "black", "grey"}
    tasks = df["Task Object"].tolist()
    height = 600 + len(tasks) * 100
    dots = []
    grey_to_black_arrows = []
    black_to_black_arrows = []

    df = df.reset_index(drop=True)
    df["task_idx"] = df.index

    most_reliable_track = []
    dashed_arrows_to_highlight = []

    for idx, row in df.iterrows():
        task_idx = row["task_idx"]
        black_points = []

        agent_colors = {
            "HUMAN*": row["Human*"].strip().lower() if isinstance(row["Human*"], str) else "",
            "TARS*": row["TARS*"].strip().lower() if isinstance(row["TARS*"], str) else "",
        }

        # Determine most reliable path
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

        # Add dots for each agent column showing their own capabilities
        # HUMAN* column: HUMAN* performer (circle)
        # TARS* column: TARS* performer (circle) + TARS supporter (square)
        
        # HUMAN* performer in HUMAN* column
        human_performer_val = row["Human*"].strip().lower() if isinstance(row["Human*"], str) else ""
        if human_performer_val in VALID_COLORS and human_performer_val != "red":
            dots.append({
                "task": task_idx,
                "agent": "HUMAN*",
                "shape": "circle",
                "color": human_performer_val,
                "role": "performer"
            })
            black_points.append("HUMAN*")
        
        # TARS* performer in TARS* column
        tars_performer_val = row["TARS*"].strip().lower() if isinstance(row["TARS*"], str) else ""
        if tars_performer_val in VALID_COLORS and tars_performer_val != "red":
            dots.append({
                "task": task_idx,
                "agent": "TARS*",
                "shape": "circle",
                "color": tars_performer_val,
                "role": "performer"
            })
            black_points.append("TARS*")
        
        # TARS supporter in TARS* column (as square)
        tars_supporter_val = row["TARS"].strip().lower() if isinstance(row["TARS"], str) and row["TARS"] else ""
        if tars_supporter_val in VALID_COLORS and tars_supporter_val != "red":
            dots.append({
                "task": task_idx,
                "agent": "TARS*",
                "shape": "square",
                "color": tars_supporter_val,
                "role": "supporter"
            })
        
        # HUMAN supporter in HUMAN* column (as square)
        human_supporter_val = row["Human"].strip().lower() if isinstance(row["Human"], str) and row["Human"] else ""
        if human_supporter_val in VALID_COLORS and human_supporter_val != "red":
            dots.append({
                "task": task_idx,
                "agent": "HUMAN*",
                "shape": "square",
                "color": human_supporter_val,
                "role": "supporter"
            })

        # Add dashed arrows from supporters to performers (even though supporters aren't shown)
        # TARS -> HUMAN*
        tars_val = row["TARS"].strip().lower() if isinstance(row["TARS"], str) else ""
        human_val = row["Human"].strip().lower() if isinstance(row["Human"], str) else ""
        
        if ("HUMAN*" in black_points and 
            row["Human*"].strip().lower() != "red" and 
            tars_val != "" and tars_val != "red"):
            dashed_arrow = {
                "start_agent": "TARS*",  # TARS supporter helps HUMAN* performer
                "end_agent": "HUMAN*",
                "task": task_idx,
                "is_support": True
            }
            grey_to_black_arrows.append(dashed_arrow)
            if highlight_track == "most_reliable" and (task_idx, "HUMAN*") in most_reliable_track:
                dashed_arrows_to_highlight.append(dashed_arrow)

        # HUMAN -> TARS*
        if ("TARS*" in black_points and 
            row["TARS*"].strip().lower() != "red" and 
            human_val != "" and human_val != "red"):
            dashed_arrow = {
                "start_agent": "HUMAN*",  # HUMAN supporter helps TARS* performer
                "end_agent": "TARS*",
                "task": task_idx,
                "is_support": True
            }
            grey_to_black_arrows.append(dashed_arrow)
            if highlight_track == "most_reliable" and (task_idx, "TARS*") in most_reliable_track:
                dashed_arrows_to_highlight.append(dashed_arrow)

    # Black-to-black transitions between performers (circles only)
    for i in range(len(df) - 1):
        current_blacks = [d for d in dots if d["task"] == i and d["agent"] in ["HUMAN*", "TARS*"] and d["shape"] == "circle"]
        next_blacks = [d for d in dots if d["task"] == i + 1 and d["agent"] in ["HUMAN*", "TARS*"] and d["shape"] == "circle"]
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
            f"<b>Task:</b> {wrap_text(row['Task Object'])}<br>"
            f"<b>Agent:</b> {dot['agent']}<br>"
            f"<b>Role:</b> {dot['role'].capitalize()}<br><br>"
            f"<b>Observability:</b><br>{wrap_text(row.get('Observability', ''))}<br><br>"
            f"<b>Predictability:</b><br>{wrap_text(row.get('Predictability', ''))}<br><br>"
            f"<b>Directability:</b><br>{wrap_text(row.get('Directability', ''))}"
        )

        if dot["shape"] == "circle":
            # Draw performer circle
            fig.add_trace(go.Scatter(
                x=[agent_pos[dot["agent"]]],
                y=[dot["task"]],
                mode="markers",
                marker=dict(size=40, color=dot["color"]),
                showlegend=False,
                hoverinfo="text",
                hovertext=hover_text
            ))
        else:  # square
            # Draw black border square
            fig.add_trace(go.Scatter(
                x=[agent_pos[dot["agent"]]],
                y=[dot["task"]],
                mode="markers",
                marker=dict(
                    size=20,
                    color="black",
                    symbol="square"
                ),
                showlegend=False,
                hoverinfo="skip"
            ))
            # Draw colored square on top
            fig.add_trace(go.Scatter(
                x=[agent_pos[dot["agent"]]],
                y=[dot["task"]],
                mode="markers",
                marker=dict(
                    size=17,
                    color=dot["color"],
                    symbol="square"
                ),
                showlegend=False,
                hoverinfo="text",
                hovertext=hover_text
            ))

    # Add dashed horizontal lines for support relationships
    # Group arrows by task to detect overlaps
    arrows_by_task = {}
    for arrow in grey_to_black_arrows:
        if arrow.get("is_support"):
            task = arrow["task"]
            if task not in arrows_by_task:
                arrows_by_task[task] = []
            arrows_by_task[task].append(arrow)
    
    for task, task_arrows in arrows_by_task.items():
        # If there are 2 arrows at the same task, offset them vertically
        if len(task_arrows) == 2:
            offsets = [-0.08, 0.08]  # Small vertical offset
        else:
            offsets = [0] * len(task_arrows)
        
        for arrow, offset in zip(task_arrows, offsets):
            is_highlighted = highlight_track == "most_reliable" and arrow in dashed_arrows_to_highlight
            # Draw dashed line from start_agent to end_agent with offset
            fig.add_shape(
                type="line",
                x0=agent_pos[arrow["start_agent"]],
                y0=arrow["task"] + offset,
                x1=agent_pos[arrow["end_agent"]],
                y1=arrow["task"] + offset,
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
            opacity=0.9,
            standoff=15  # Stop arrow at circle edge
        )

    fig.update_layout(
        title="Combined Workflow Graph - Performers Only (All Procedures)",
        xaxis=dict(
            tickvals=list(agent_pos.values()),
            ticktext=list(agent_pos.keys()),
            title="Agent",
            showgrid=True,
            gridcolor='lightgrey',
            range=[-0.5, 1.5]  # Constrain x-axis to keep agents closer
        ),
        yaxis=dict(
            tickvals=list(df["task_idx"]),
            ticktext=[wrap_text(f"{p} | {t}") for p, t in zip(df["Procedure"], df["Task Object"])],
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


def build_interdependence_figures(df, highlight_track=None):
    agents = ["HUMAN*", "TARS", "TARS*", "HUMAN"]
    VALID_COLORS = {"red", "yellow", "green", "orange", "black", "grey"}

    figures = {}

    for procedure, proc_df in df.groupby("Procedure"):
        tasks = proc_df["Task Object"].tolist()
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
                f"<b>Task:</b> {wrap_text(row['Task Object'])}<br>"
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
    tasks = df["Task Object"].tolist()
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
            f"<b>Task:</b> {wrap_text(row['Task Object'])}<br>"
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
            ticktext=[wrap_text(f"{p} | {t}") for p, t in zip(df["Procedure"], df["Task Object"])],
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


# Page layouts
def interdependence_analysis_page():
    return html.Div([
        # The table goes here
        html.H2("Interdependence Table - Engine Failure at Take Off", style={"textAlign": "center"}),
        html.Div(id="table-wrapper", children=[table]),
        html.Div([
            html.Div([
                # Left-aligned buttons
                html.Div([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Button('Load Table', id='load-button', n_clicks=0),
                        multiple=False,
                        style={
                            'display': 'inline-block',
                            'marginRight': '10px'
                        }
                    ),
                    html.Button("Add Row", id="add-row-button", n_clicks=0),
                    html.Button("Copy Cell Down", id="copy-down-button", n_clicks=0),
                ], style={"display": "flex", "gap": "10px"}),

                # Right-aligned Save button
                html.Div([
                    html.Button("Save Table", disabled=False, id="save-button", n_clicks=0)
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

        html.Div([
            dcc.RadioItems(
                id="view-selector",
                options=[
                    {"label": "Full View (All Agents)", "value": "full"},
                    {"label": "Performers Only (HUMAN* & TARS*)", "value": "performers"}
                ],
                value="full",
                labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                style={"textAlign": "center", "marginTop": "20px"}
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

        # Bar chart for most reliable path color counts
        html.Div([
            dcc.Graph(id="bar-chart-whole-scenario"),
            dcc.Graph(id="most-reliable-bar-chart"),
            dcc.Graph(id="spo_baseline-bar-chart"),
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


def assumptions_page():
    return html.Div([
        html.H2("Assumptions", style={"textAlign": "center"}),
        html.Div([
            html.H3("Operational Scenario", style={"marginBottom": "20px"}),
            html.Div([
                html.H4("Takeoff with an engine failure caused by a bird strike", style={"marginTop": "30px", "marginBottom": "15px"}),
                html.Ul([
                    html.Li([html.B("Departure Airport"), ": CYUL"]),
                    html.Li([html.B("Arrival Airport"), ": CYOW"]),
                    html.Li([html.B("Time of day"), ": Day (14:00 local time)"]),
                    html.Li([html.B("Weather conditions"), ": CAVOK"]),
                    html.Li([html.B("Wind"), ": Calm, 5 knots, 160°"]),
                    html.Li([html.B("Temperature"), ": 22°C"]),
                    html.Li([html.B("Atmospheric Pressure"), ": 29.92 inHg"]),
                    html.Li([html.B("Departure runway"), ": 24R"]),
                    html.Li([html.B("Arrival Runway"), ": 14"]),
                    html.Li([html.B("Aircraft"), ": Cessna Citation Mustang Very Light Jet"]),
                    html.Li([html.B("Crew"), ": 1 pilot, 2 pax"]),
                    html.Li([html.B("Flight phase"), ": Takeoff"]),
                    html.Li([html.B("Emergency"), ": Engine failure caused by a bird strike"]),
                ], style={"fontSize": "16px", "lineHeight": "1.6"}),
            ], style={"maxWidth": "800px", "margin": "0 auto", "padding": "20px"})
        ]),
        html.Div([
            html.H3("Single Pilot", style={"marginBottom": "20px"}),
            html.Div([
                html.H4("Key Assumptions:", style={"marginTop": "30px", "marginBottom": "15px"}),
                html.Ul([
                    html.Li([html.B("Role"), ": The pilot is the captain and retains final authority and responsibility for decision making throughout the flight."]),
                    html.Li([html.B("Experience"), ": >5,000 flight hours, including >1,000 hours in type."]),
                    html.Li([html.B("Training"), ": Recent training on emergency procedures, including engine failure at takeoff scenarios."]),
                    html.Li([html.B("Health"), ": Good health, well-rested, and fit to fly."]),
                    html.Li([html.B("Support"), ": The pilot has access to TARS for assistance with monitoring and procedural guidance."]),
                ], style={"fontSize": "16px", "lineHeight": "1.6"}),
            ], style={"maxWidth": "800px", "margin": "0 auto", "padding": "20px"})
        ]),
        html.Div([
            html.H3("Autonomous System : TARS", style={"marginBottom": "20px"}),
            html.Div([
                html.H4("Key Assumptions:", style={"marginTop": "30px", "marginBottom": "15px"}),
                html.Ul([
                    html.Li([html.B("Role"), ": TARS assists the pilot on multiple tasks during normal, abnormal and emergency situation according to the briefing. But the pilot retains final authority. TARS is designed to complement human decision-making, not replace it. The pilot must remain engaged and aware of the flight situation at all times."]),
                    html.Li([html.B("Description: "), "TARS is implemented as a reactive software designed to assist the pilot in managing the flight scenario. TARS organizes its behavior into a structured hierarchy of states and substates. Each state encapsulates a specific procedural phase of the scenario, with transitions triggered by defined environmental conditions."]),
                    html.Li([html.B("Sensors"), ": TARS has access to all flight parameters available in the avionics system, including aircraft state and system status. TARS does not have access to environmental parameters outside of the avionics system."]),
                    html.Li([html.B("Actuators"), ": TARS can interact with the pilot through the interface with the pilot, for checklist support. TARS can interact with the airplane through a predefined set of commands, including the FMS and autopilot, but TARS cannot perform physical actions in the cockpit (e.g., moving flight controls, switching levers, pushing buttons)."]),
                    html.Li([html.B("Reliability"), ": TARS is assumed to operate without technical failures during the scenario. TARS is assumed to correctly interpret sensor data and execute commands without errors and under 1ms latency."]),
                    html.Li([html.B("Graphical Interface"), ": The Human-TARS interface is an EFB-like tactile application run on a tablet placed on the left-side of the single pilot in the cockpit that provides access to TARS current state and behavioral intent. The interface allows the pilot to monitor TARS actions, receive alerts, and input commands."]),
                    html.Li([html.B("Voice Interface"), ": TARS can communicate with the pilot using auditory alerts and speech synthesis. The pilot can issue commands to TARS vocally (e.g. cancel action)."]),
                ], style={"fontSize": "16px", "lineHeight": "1.6"}),
            ], style={"maxWidth": "800px", "margin": "0 auto", "padding": "20px"})
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


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(
        "© Benjamin R. Berton 2025 Polytechnique Montreal",
        style={
            "textAlign": "right",
            "color": "#888",
            "fontSize": "14px",
            "marginBottom": "10px"
        }
    ),
    html.Div(id='navbar'),
    html.Div(id='page-content')
])

# Callback to update navbar based on current pathname
@app.callback(Output('navbar', 'children'), Input('url', 'pathname'))
def update_navbar(pathname):
    return create_navbar(current_pathname=pathname)

# Callback for page routing
@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/assumptions':
        return assumptions_page()
    else:
        return interdependence_analysis_page()

@app.callback(
    Output("interdependence-graph", "figure"),
    Output("bar-chart-whole-scenario", "figure"),
    Output("most-reliable-bar-chart", "figure"),
    Output("spo_baseline-bar-chart", "figure"),
    Output("allocation-type-bar-chart", "figure"),
    Output("agent-autonomy-bar-chart", "figure"),
    Input("procedure-dropdown", "value"),
    Input("highlight-selector", "value"),
    Input("view-selector", "value"),
    State("responsibility-table", "data")
)
def update_graph_and_bar(procedure, highlight_track, view_mode, data):
    df = pd.DataFrame(data)
    # --- Workflow Graph ---
    if procedure is None:
        # Combined view
        if view_mode == "performers":
            workflow_fig = build_performers_only_combined_figure(df, highlight_track)
        else:
            workflow_fig = build_combined_interdependence_figure(df, highlight_track)
        df_bar = df
    else:
        # Single procedure view
        if highlight_track == "none":
            highlight_track_val = None
        else:
            highlight_track_val = highlight_track
        
        if view_mode == "performers":
            figures = build_performers_only_figures(df, highlight_track_val)
        else:
            figures = build_interdependence_figures(df, highlight_track_val)
        
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
    tars_performer_green = 0
    tars_performer_yellow = 0
    tars_performer_orange = 0
    tars_supporter_green = 0
    tars_supporter_yellow = 0
    tars_supporter_orange = 0
    for idx, row in df_bar.iterrows():
        agent_colors = {
            "HUMAN*": str(row.get("Human*", "") or "").strip().lower(),
            "TARS*": str(row.get("TARS*", "") or "").strip().lower(),
        }
        supporter_colors = {
            "HUMAN": str(row.get("Human", "") or "").strip().lower(),
            "TARS": str(row.get("TARS", "") or "").strip().lower(),
        }
        # Find performer (most reliable)
        for agent in ["HUMAN*", "TARS*"]:
            if agent_colors[agent] == "green":
                if agent ==  "HUMAN*":
                    human_performer_green += 1
                if agent == "TARS*":
                    tars_performer_green += 1
                performer_green += 1
            if agent_colors[agent] == "yellow":
                if agent ==  "HUMAN*":
                    human_performer_yellow += 1
                if agent == "TARS*":
                    tars_performer_yellow += 1
                performer_yellow += 1
            if agent_colors[agent] == "orange":
                if agent ==  "HUMAN*":
                    human_performer_orange += 1
                if agent == "TARS*":
                    tars_performer_orange += 1
                performer_orange += 1
        for agent in ["HUMAN", "TARS"]:
            if supporter_colors[agent] == "green":
                if agent ==  "HUMAN":
                    human_supporter_green += 1
                if agent == "TARS":
                    tars_supporter_green += 1
                supporter_green += 1
            if supporter_colors[agent] == "yellow":
                if agent ==  "HUMAN":
                    human_supporter_yellow += 1
                if agent == "TARS":
                    tars_supporter_yellow += 1
                supporter_yellow += 1
            if supporter_colors[agent] == "orange":
                if agent ==  "HUMAN":
                    human_supporter_orange += 1
                if agent == "TARS":
                    tars_supporter_orange += 1
                supporter_orange += 1

    bar_fig_whole_scenario = go.Figure()
    # Stacked bars for performer colors
    bar_fig_whole_scenario.add_trace(go.Bar(
        name="Human Performer",
        x=["Performer Green", "Performer Yellow", "Performer Orange"],
        y=[human_performer_green, human_performer_yellow, human_performer_orange],
        marker_color=["seagreen", "gold", "darkorange"]
    ))
    bar_fig_whole_scenario.add_trace(go.Bar(
        name="TARS Performer",
        x=["Performer Green", "Performer Yellow", "Performer Orange"],
        y=[tars_performer_green, tars_performer_yellow, tars_performer_orange],
        marker_color=["limegreen", "khaki", "orange"]
    ))
    # Stacked bars for supporter colors
    bar_fig_whole_scenario.add_trace(go.Bar(
        name="Human Supporter",
        x=["Supporter Green", "Supporter Yellow", "Supporter Orange"],
        y=[human_supporter_green, human_supporter_yellow, human_supporter_orange],
        marker_color=["seagreen", "gold", "darkorange"]
    ))
    bar_fig_whole_scenario.add_trace(go.Bar(
        name="TARS Supporter",
        x=["Supporter Green", "Supporter Yellow", "Supporter Orange"],
        y=[tars_supporter_green, tars_supporter_yellow, tars_supporter_orange],
        marker_color=["limegreen", "khaki", "orange"]
    ))
    bar_fig_whole_scenario.update_layout(
        title="Performer and Supporter Capacities in Mixed Initiative",
        xaxis_title="Role and Capacity",
        yaxis_title="Number of Tasks",
        barmode='stack',
        bargap=0.3,
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
    tars_performer_green = 0
    tars_performer_yellow = 0
    tars_performer_orange = 0
    tars_supporter_green = 0
    tars_supporter_yellow = 0
    tars_supporter_orange = 0
    for idx, row in df_bar.iterrows():
        agent_colors = {
            "HUMAN*": str(row.get("Human*", "") or "").strip().lower(),
            "TARS*": str(row.get("TARS*", "") or "").strip().lower(),
        }
        supporter_colors = {
            "HUMAN": str(row.get("Human", "") or "").strip().lower(),
            "TARS": str(row.get("TARS", "") or "").strip().lower(),
        }
        # Find performer (most reliable)
        performer = None
        for agent in ["HUMAN*", "TARS*"]:
            if agent_colors[agent] == "green":
                performer = agent
                performer_color = "green"
                break
        if not performer:
            for agent in ["HUMAN*", "TARS*"]:
                if agent_colors[agent] == "yellow":
                    performer = agent
                    performer_color = "yellow"
                    break
        if not performer:
            for agent in ["HUMAN*", "TARS*"]:
                if agent_colors[agent] == "orange":
                    performer = agent
                    performer_color = "orange"
                    break
        # Count performer color
        if performer:
            if performer_color == "green":
                performer_green += 1
                if performer == "HUMAN*":
                    human_performer_green += 1
                else:
                    tars_performer_green += 1
            elif performer_color == "yellow":
                performer_yellow += 1
                if performer == "HUMAN*":
                    human_performer_yellow += 1
                else:
                    tars_performer_yellow += 1
            elif performer_color == "orange":
                performer_orange += 1
                if performer == "HUMAN*":
                    human_performer_orange += 1
                else:
                    tars_performer_orange += 1
            # Now check for supporter (the other agent)
            supporter = "TARS" if performer == "HUMAN*" else "HUMAN"
            supporter_color = supporter_colors[supporter]
            if supporter_color == "green":
                supporter_green += 1
                if supporter == "HUMAN":
                    human_supporter_green += 1
                else:
                    tars_supporter_green += 1
            elif supporter_color == "yellow":
                supporter_yellow += 1
                if supporter == "HUMAN":
                    human_supporter_yellow += 1
                else:
                    tars_supporter_yellow += 1
            elif supporter_color == "orange":
                supporter_orange += 1
                if supporter == "HUMAN":
                    human_supporter_orange += 1
                else:
                    tars_supporter_orange += 1

    bar_fig = go.Figure()
    # Stacked bars for performer colors
    bar_fig.add_trace(go.Bar(
        name="Human Performer",
        x=["Performer Green", "Performer Yellow", "Performer Orange"],
        y=[human_performer_green, human_performer_yellow, human_performer_orange],
        marker_color=["seagreen", "gold", "darkorange"]
    ))
    bar_fig.add_trace(go.Bar(
        name="TARS Performer",
        x=["Performer Green", "Performer Yellow", "Performer Orange"],
        y=[tars_performer_green, tars_performer_yellow, tars_performer_orange],
        marker_color=["limegreen", "khaki", "orange"]
    ))
    # Stacked bars for supporter colors
    bar_fig.add_trace(go.Bar(
        name="Human Supporter",
        x=["Supporter Green", "Supporter Yellow", "Supporter Orange"],
        y=[human_supporter_green, human_supporter_yellow, human_supporter_orange],
        marker_color=["seagreen", "gold", "darkorange"]
    ))
    bar_fig.add_trace(go.Bar(
        name="TARS Supporter",
        x=["Supporter Green", "Supporter Yellow", "Supporter Orange"],
        y=[tars_supporter_green, tars_supporter_yellow, tars_supporter_orange],
        marker_color=["limegreen", "khaki", "orange"]
    ))
    bar_fig.update_layout(
        title="Most Reliable Path: Performer and Supporter Capacities",
        xaxis_title="Role and Capacity",
        yaxis_title="Number of Tasks",
        barmode='stack',
        bargap=0.3,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False
    )

    # --- Bar Chart for spo baseline Path ---
    performer_green = 0
    performer_yellow = 0
    performer_orange = 0
    for idx, row in df_bar.iterrows():
        agent_colors = {
            "HUMAN*": str(row.get("Human*", "") or "").strip().lower(),
        }
        performer = "HUMAN*"
        if performer:
            if agent_colors[performer] == "green":
                performer_green += 1
            elif agent_colors[performer] == "yellow":
                performer_yellow += 1
            elif agent_colors[performer] == "orange":
                performer_orange += 1
            elif agent_colors[performer] == "orange":
                performer_orange += 1

    bar_fig_spo = go.Figure()
    # Stacked bars for performer colors
    bar_fig_spo.add_trace(go.Bar(
        name="Performer Green",
        x=["Performer Green"],
        y=[performer_green],
        marker_color="seagreen"
    ))
    bar_fig_spo.add_trace(go.Bar(
        name="Performer Yellow",
        x=["Performer Yellow"],
        y=[performer_yellow],
        marker_color="gold"
    ))
    bar_fig_spo.add_trace(go.Bar(
        name="Performer Orange",
        x=["Performer Orange"],
        y=[performer_orange],
        marker_color="darkorange"
    ))
    bar_fig_spo.update_layout(
        title="SPO Baseline Path: Human pilot capacities",
        xaxis_title="Role and Capacity",
        yaxis_title="Number of Tasks",
        bargap=0.3,
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
        tars_star = str(row.get("TARS*", "") or "").strip().lower()
        human = str(row.get("Human", "") or "").strip().lower()
        tars = str(row.get("TARS", "") or "").strip().lower()

        VALID_COLORS = {"red", "yellow", "green", "orange"}
        
        # Count valid performers (not red)
        performers = []
        if human_star in VALID_COLORS and human_star != "red":
            performers.append("Human*")
        if tars_star in VALID_COLORS and tars_star != "red":
            performers.append("TARS*")
        
        # Count valid supporters (not red)
        supporters = []
        if human in VALID_COLORS and human != "red":
            supporters.append("Human")
        if tars in VALID_COLORS and tars != "red":
            supporters.append("TARS")
        
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
        "TARS*": {"autonomous": 0, "non_autonomous": 0}
    }
    
    # Process all tasks in sequence (not grouped by procedure)
    prev_performers = []
    
    for idx, row in df_bar.iterrows():
        # Skip the first task (task 0) as there's no previous task to compare
        if idx == 0:
            # Get current task performers for next iteration
            human_star = str(row.get("Human*", "") or "").strip().lower()
            tars_star = str(row.get("TARS*", "") or "").strip().lower()
            
            VALID_COLORS = {"red", "yellow", "green", "orange"}
            
            current_performers = []
            if human_star in VALID_COLORS and human_star != "red":
                current_performers.append("Human*")
            if tars_star in VALID_COLORS and tars_star != "red":
                current_performers.append("TARS*")
            
            prev_performers = current_performers
            continue
        
        # Get current task performers
        human_star = str(row.get("Human*", "") or "").strip().lower()
        tars_star = str(row.get("TARS*", "") or "").strip().lower()
        
        VALID_COLORS = {"red", "yellow", "green", "orange"}
        
        # Map agents to their colors for this task
        agent_colors = {
            "Human*": human_star,
            "TARS*": tars_star
        }
        
        current_performers = []
        if human_star in VALID_COLORS and human_star != "red":
            current_performers.append("Human*")
        if tars_star in VALID_COLORS and tars_star != "red":
            current_performers.append("TARS*")
        
        # For each agent that can perform this task
        for agent in current_performers:
            # Orange tasks are always non-autonomous
            if agent_colors[agent] == "orange":
                agent_autonomy[agent]["non_autonomous"] += 1
            elif agent not in prev_performers:
                # Agent couldn't perform previous task
                agent_autonomy[agent]["non_autonomous"] += 1
            else:
                # Agent could also perform previous task = autonomous (green or yellow only)
                agent_autonomy[agent]["autonomous"] += 1
        
        prev_performers = current_performers
    
    # Create horizontal stacked bar chart for agent autonomy
    autonomy_fig = go.Figure()
    
    agents = ["Human*", "TARS*"]
    colors_autonomous = {"Human*": "lightgreen", "TARS*": "lightgreen"}
    colors_non_autonomous = {"Human*": "lightcoral", "TARS*": "lightcoral"}
    
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

    return workflow_fig, bar_fig_whole_scenario, bar_fig, bar_fig_spo, allocation_fig, autonomy_fig


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

    save_message = ""

    # Case 1: Save button clicked
    if triggered == "save-button":
        df = pd.DataFrame(data)
        # Don't recreate Row column if it already exists
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

    # Ensure Row column is properly maintained (no need to reassign if it exists)
    if 'Row' not in df.columns:
        df = df.assign(Row=lambda x: x.index + 1)

    updated_table = dash_table.DataTable(
        id='responsibility-table',
        columns=columns,
        data=df.to_dict("records"),
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