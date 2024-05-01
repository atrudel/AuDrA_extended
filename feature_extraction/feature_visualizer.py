import dash
from dash import dcc
import plotly.graph_objs as go
from dash import html, no_update, callback
from dash.dependencies import Input, Output


def create_dash_visualizer(processed_data: dict) -> dash.Dash:
    images_base64 = processed_data['images_base64']
    tsne_results = processed_data['tsne_results']
    originalities = processed_data['originalities'].squeeze().numpy()

    @callback(
        Output("graph-tooltip-5", "show"),
        Output("graph-tooltip-5", "bbox"),
        Output("graph-tooltip-5", "children"),
        Input("graph-5", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        # demo only shows the first point, but other points may also be available
        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]
        num = hover_data["pointNumber"]

        im_url = images_base64[num]
        children = [
            html.Div([
                html.Img(
                    src=im_url,
                    style={"width": "50px", 'display': 'block', 'margin': '0 auto'},
                )
            ])
        ]

        return True, bbox, children

    fig = go.Figure(data=[go.Scatter3d(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        z=tsne_results[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=originalities,
            colorscale='Viridis',
            colorbar=dict(title='Originality'),
            showscale=True
        )
    )])
    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None
    )
    fig.update_layout(width=1400, height=800)

    app = dash.Dash(__name__)

    app.layout = html.Div(
        className="container",
        children=[
            dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip-5", direction='bottom')
        ],
    )
    return app
