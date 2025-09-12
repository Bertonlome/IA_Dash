import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html

# Initialize Dash app
app = dash.Dash(__name__)

# Sample CSV data (replace with pd.read_csv('your_file.csv'))
data = {
    'x': [1, 2, 3, 4, 5],
    'y': [2, 4, 1, 3, 5],
    'z': [1, 1, 2, 2, 3],
    'color': ['red', 'blue', 'green', 'purple', 'orange'],
    'label': ['A', 'B', 'C', 'D', 'E']
}
df = pd.DataFrame(data)

# Create Plotly figure
fig = go.Figure()

# Base scatter plot (all nodes, semi-transparent)
fig.add_trace(go.Scatter3d(
    x=df['x'], y=df['y'], z=df['z'],
    mode='markers+text',
    marker=dict(size=8, color=df['color'], opacity=0.3),
    text=df['label'], textposition='top center',
    name='Nodes'
))

# Animation frames: highlight one node at a time
frames = []
for i in range(len(df)):
    frame_data = go.Scatter3d(
        x=df['x'], y=df['y'], z=df['z'],
        mode='markers+text',
        marker=dict(
            size=[12 if j == i else 8 for j in range(len(df))],
            color=df['color'],
                opacity=1.0 if i == 0 else 0.3
        ),
        text=df['label'], textposition='top center'
    )
    frames.append(go.Frame(data=[frame_data], name=str(i)))

fig.frames = frames

# Layout with buttons and slider
fig.update_layout(
    scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
        aspectmode='cube'
    ),
    updatemenus=[
        dict(
            type='buttons',
            showactive=True,
            buttons=[
                dict(
                    label='Play',
                    method='animate',
                    args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True, mode='immediate')]
                ),
                dict(
                    label='Stop',
                    method='animate',
                    args=[[None], dict(frame=dict(duration=0, redraw=True), mode='immediate')]
                ),
                dict(
                    label='Next',
                    method='animate',
                    args=[None, dict(frame=dict(duration=0, redraw=True), transition=dict(duration=0), fromcurrent=True)]
                )
            ],
            direction='left',
            pad={'r': 10, 't': 10},
            x=0.1, xanchor='left',
            y=1.1, yanchor='top'
        )
    ],
    sliders=[dict(
        steps=[dict(method='animate', args=[[str(i)], dict(mode='immediate', frame=dict(duration=500, redraw=True), transition=dict(duration=0))], label=f'Node {df["label"][i]}') for i in range(len(df))],
        active=0,
        x=0.1, len=0.9, y=0
    )],
    title='3D Animated Nodes'
)

# Dash layout
app.layout = html.Div([
    html.H1("3D Node Animation App"),
    dcc.Graph(id='3d-graph', figure=fig),
    html.P("Use the buttons to play, stop, or step through node highlights, or use the slider to jump to specific nodes.")
])

# Run the app
if __name__ == '__main__':
    app.run(debug=True)