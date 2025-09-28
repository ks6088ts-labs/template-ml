import logging
from typing import Annotated

import pandas as pd
import plotly.express as px
import typer

# Import packages
from dash import Dash, Input, Output, callback, dash_table, dcc, html
from dotenv import load_dotenv

from template_ml.loggers import get_logger

app = typer.Typer(
    add_completion=False,
    help="Dash Playground CLI",
)

logger = get_logger(__name__)


def set_verbose_logging(
    verbose: bool,
):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)


@app.command(help="ref. https://dash.plotly.com/minimal-app")
def minimal_app(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = True,
):
    set_verbose_logging(verbose)

    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv")
    app = Dash()
    app.layout = [
        html.H1(
            children="Title of Dash App",
            style={"textAlign": "center"},
        ),
        dcc.Dropdown(
            df.country.unique(),
            "Canada",
            id="dropdown-selection",
        ),
        dcc.Graph(
            id="graph-content",
        ),
    ]

    @callback(Output("graph-content", "figure"), Input("dropdown-selection", "value"))
    def update_graph(value):
        dff = df[df.country == value]
        return px.line(dff, x="year", y="pop")

    app.run(debug=True)


@app.command(help="ref. https://dash.plotly.com/tutorial")
def tutorial(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = True,
):
    set_verbose_logging(verbose)

    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv")
    app = Dash()
    app.layout = [
        html.Div(
            children="My First App with Data, Graph, and Controls",
        ),
        html.Hr(),
        dcc.RadioItems(
            options=["pop", "lifeExp", "gdpPercap"],
            value="lifeExp",
            id="my-final-radio-item-example",
        ),
        dash_table.DataTable(
            data=df.to_dict("records"),
            page_size=6,
        ),
        dcc.Graph(
            figure={},
            id="my-final-graph-example",
        ),
    ]

    # Add controls to build the interaction
    @callback(
        Output(component_id="my-final-graph-example", component_property="figure"),
        Input(component_id="my-final-radio-item-example", component_property="value"),
    )
    def update_graph(col_chosen):
        fig = px.histogram(df, x="continent", y=col_chosen, histfunc="avg")
        return fig

    app.run(debug=True)


@app.command(help="ref. https://dash-example-index.herokuapp.com/")
def simple_examples(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = True,
):
    set_verbose_logging(verbose)
    print("Visit https://dash-example-index.herokuapp.com/ for examples")


if __name__ == "__main__":
    assert load_dotenv(
        override=True,
        verbose=True,
    ), "Failed to load environment variables"
    app()
