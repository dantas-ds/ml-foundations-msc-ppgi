import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def scatter(
    X: np.ndarray,
    y: np.ndarray,
    *,
    title: str = "Bivariate Scatter",
    show_ellipses: bool = True,
    opacity: float = 0.85,
):
    df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "cls": y.astype(str)})

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cls",
        opacity=opacity,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    if show_ellipses:
        for cls in df["cls"].unique():
            sub = df[df["cls"] == cls]
            mx, my = sub["x"].mean(), sub["y"].mean()
            sx, sy = sub["x"].std(), sub["y"].std()
            t = np.linspace(0, 2 * np.pi, 200)

            xe = mx + sx * np.cos(t)
            ye = my + sy * np.sin(t)

            fig.add_trace(
                go.Scatter(
                    x=xe,
                    y=ye,
                    mode="lines",
                    name=f"ellipse {cls}",
                    line=dict(width=2),
                    showlegend=False,
                )
            )

    fig.update_layout(
        width=650,
        height=500,
        title_x=0.02,
        title_font=dict(size=18),
        legend_title_text="Class",
    )

    return fig
