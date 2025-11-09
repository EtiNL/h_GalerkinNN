import plotly.io as pio
import plotly.graph_objects as go

def plot_sim_result(x,y,z,zaxis_title,notebook_plot = True):
    if notebook_plot == False:
        pio.renderers.default = "browser"

    fig = go.Figure(data=[
            go.Surface(
                x=x,         # z-axis in formula
                y=y,         # t-axis
                z=z,
                colorscale="Viridis",
                opacity=0.95,
                contours=dict(
                    z=dict(
                        usecolormap=True,
                        highlight=False,
                    )
                ),
            )
        ])

    fig.update_layout(
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_title="z",
            yaxis_title="t",
            zaxis_title=zaxis_title,
        )
    )

    fig.show()