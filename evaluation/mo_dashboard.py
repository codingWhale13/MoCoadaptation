# WEB-BASED GUI USING PLOTLY DASH
import os
import sys

from dash import Dash, dcc, html, Input, Output
from dash.exceptions import PreventUpdate
import dash_ag_grid as dag
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DESIGN_PARAMETERS, ORIGINAL_CONFIG, get_reward_names_from_csv, load_csv, load_config, pref2str


# ========== CONSTANTS ==========
TEST_DATA_DIR = "MoCoadaptation/evaluation/test_data"  # Contains experiments
active_exp_names = list(next(os.walk(TEST_DATA_DIR))[1])
OPTIONS = list(next(os.walk(TEST_DATA_DIR))[1])

EXP_NAME = "Experiment Name"

DESIGN_CYCLE = "Design Cycle"
RUN_NAME = "Run Name"
RUN_ID = "Run ID"
SOURCE_PREF = "Source Pref."
TARGET_PREF = "Target Pref."
# The following "constants" will be set during initial data reading
N_DESIGN_PARAMETERS = None
REWARD_NAMES = None


# ========== HELPERS ==========
def run_id_to_path(run_id):
    for exp in active_exp_names:
        latest = os.path.join(TEST_DATA_DIR, exp, "latest")
        for run_dir in os.listdir(latest):
            run_path = os.path.join(latest, run_dir)
            config = load_config(run_path, ORIGINAL_CONFIG)
            if config["run_id"] == run_id:
                return run_path


def load_state_or_action(
    run_id: str,
    which: str,  # "state" or "action"
    precision=2,
    n_samples=100,
    random_state=42,
):
    path = run_id_to_path(run_id)
    filename = f"{which}_values_1.csv"
    values = pd.read_csv(os.path.join(path, filename), header=None)
    values = values.sample(n=n_samples, random_state=random_state)  # Speed up loading
    values = values.round(precision)

    if which == "state":
        # Remove constant design parameters that are included in the state
        values = values.iloc[:, :-N_DESIGN_PARAMETERS]

    return values


# ========== DATA PRE-PROCESSING ==========
def load_test_data():
    global N_DESIGN_PARAMETERS, REWARD_NAMES

    some_csv_path = None
    for dirpath, _, files in os.walk(os.path.join(TEST_DATA_DIR)):
        for filename in files:
            if filename.startswith("episodic_rewards"):
                some_csv_path = os.path.join(dirpath, filename)
                break
        if some_csv_path is not None:
            break

    REWARD_NAMES = get_reward_names_from_csv(some_csv_path)

    # cb9ef5b8
    # ecef62de
    # 0e56e3dc

    # 1) Load test results from latest iteration
    reward_data = {
        EXP_NAME: [],
        RUN_ID: [],
        SOURCE_PREF: [],
        TARGET_PREF: [],
        **{reward_name: [] for reward_name in REWARD_NAMES},
    }
    for exp_name in active_exp_names:
        latest = os.path.join(TEST_DATA_DIR, exp_name, "latest")
        for run_dir in os.listdir(latest):
            mean_rewards = {rn: [] for rn in REWARD_NAMES}
            run_path = os.path.join(latest, run_dir)
            for filename in os.listdir(run_path):
                if filename.startswith("episodic_rewards"):
                    test_data = load_csv(os.path.join(run_path, filename))
                    for rew_name in REWARD_NAMES:
                        mean_rewards[rew_name].append(test_data[rew_name])
                    if N_DESIGN_PARAMETERS is None:
                        N_DESIGN_PARAMETERS = len(test_data[DESIGN_PARAMETERS])

            config = load_config(run_path, ORIGINAL_CONFIG)
            run_id = config["run_id"]
            target_pref = pref2str(config["weight_preference"])

            reward_data[RUN_ID].append(run_id)
            for reward_name in REWARD_NAMES:
                reward_data[reward_name].append(np.mean(mean_rewards[reward_name]))

            if (
                "previous_weight_preferences" not in config
                or len(config["previous_weight_preferences"]) == 0
            ):
                source_pref = "Baseline"
            else:
                source_pref = pref2str(config["previous_weight_preferences"][-1][1])

            reward_data[SOURCE_PREF].append(source_pref)
            reward_data[TARGET_PREF].append(target_pref)
            reward_data[EXP_NAME].append(exp_name)

    df_latest = pd.DataFrame.from_dict(reward_data)

    # 2) Load results over time (considering past iterations)
    # TODO: string together full history of training, not just the last run

    history = {RUN_ID: [], DESIGN_CYCLE: [], REWARD_NAMES[0]: [], REWARD_NAMES[1]: []}
    rewards = [[] for _ in range(len(REWARD_NAMES))]
    # mapping from run_id to tuple previous run id (run IDs without predecessor are not included)
    designs = {}  # Map run_id -> design parameters
    pref_history = {}

    for exp_name in active_exp_names:
        per_iteration = os.path.join(TEST_DATA_DIR, exp_name, "per_iteration")
        for run_dir in os.listdir(per_iteration):
            run_path = os.path.join(per_iteration, run_dir)
            csv_files = [
                f for f in os.listdir(run_path) if f.startswith("episodic_rewards")
            ]
            previous_cycle = None  # keep track of previous cycle (will be x in graph)

            config = load_config(run_path, "original_config.json")
            run_id = config["run_id"]

            for filename in sorted(csv_files, key=lambda x: int(x.split("_")[-2])):
                cycle = int(filename.split("_")[-2])
                csv_data = load_csv(os.path.join(run_path, filename))
                if run_id not in designs:
                    designs[run_id] = [exp_name] + list(csv_data[DESIGN_PARAMETERS])

                if cycle != previous_cycle:
                    if previous_cycle is not None:
                        history[RUN_ID].append(run_id)
                        history[DESIGN_CYCLE].append(previous_cycle)
                        for i in range(len(REWARD_NAMES)):
                            history[REWARD_NAMES[i]].append(np.mean(rewards[i]))

                    # get ready for next design cycle
                    previous_cycle = cycle
                    rewards = [[] for _ in range(len(REWARD_NAMES))]

                rewards[0].append(csv_data[REWARD_NAMES[0]])
                rewards[1].append(csv_data[REWARD_NAMES[1]])

            reward_data[RUN_ID].append(run_id)
            for reward_name in REWARD_NAMES:
                reward_data[reward_name].append(np.mean(mean_rewards[reward_name]))

            # Update pref_history
            if len(config["previous_weight_preferences"]) != 0:
                previous_run_id = config["previous_weight_preferences"][-1][0]
                pref_history[run_id] = previous_run_id

    df_history = pd.DataFrame.from_dict(history)
    df_design = pd.DataFrame.from_dict(
        designs,
        orient="index",
        columns=[EXP_NAME] + [str(i) for i in range(N_DESIGN_PARAMETERS)],
    )
    df_design[RUN_ID] = df_design.index

    return df_latest, df_history, df_design, pref_history


df_latest, df_history, df_design, pref_history = load_test_data()
active_df_latest = pd.DataFrame().reindex_like(df_latest)  # Dynamic subset of df_latest


# ========== LAYOUT ==========
def get_heading(title: str, h=1):
    if h == 1:
        return html.H2(children=title, style={"textAlign": "center"})
    elif h == 2:
        return html.H2(children=title, style={"textAlign": "center"})
    elif h == 3:
        return html.H3(children=title, style={"textAlign": "center"})


app = Dash(__name__)
app.layout = html.Div(
    style={
        "display": "flex",
        "flex-direction": "row",
        "height": "100vh",
        "font-family": "Cantarell",
    },
    children=[
        # Left column with two rows
        html.Div(
            style={"flex": "1", "display": "flex", "flex-direction": "column"},
            children=[
                # Row 1
                html.Div(
                    style={"flex": "1", "display": "flex"},
                    children=[
                        html.Div(
                            style={"flex": 3, "border": "1px solid black"},
                            children=[
                                html.Div(
                                    dcc.Dropdown(
                                        id="multi-choice-dropdown",
                                        options=OPTIONS,
                                        multi=True,
                                        placeholder="Select experiment(s)",
                                    )
                                ),
                                html.Div(
                                    dag.AgGrid(
                                        id="reward-grid",
                                        columnDefs=[],
                                        rowData=[],
                                        columnSize="sizeToFit",
                                        dashGridOptions={
                                            "animateRows": False,
                                            "pagination": True,
                                            "rowSelection": "single",
                                        },
                                    ),
                                ),
                            ],
                        ),
                        html.Div(
                            id="episode-video",
                            style={
                                "flex": 2,
                                "border": "1px solid black",
                                "display": "flex",
                                "justify-content": "center",
                                "align-items": "center",
                            },
                        ),
                        html.Div(
                            id="actions",
                            style={"flex": 2, "border": "1px solid black"},
                        ),
                    ],
                ),
                # Row 2
                html.Div(
                    style={"flex": "1", "display": "flex"},
                    children=[
                        html.Div(
                            style={"flex": 1, "border": "1px solid black"},
                            children=[dcc.Graph(id="pareto-front")],
                        ),
                        html.Div(
                            id="design", style={"flex": 1, "border": "1px solid black"}
                        ),
                    ],
                ),
            ],
        ),
        # Right column
        html.Div(
            id="states",
            style={
                "flex": "0 0 20%",
                "border": "1px solid black",
            },
        ),
    ],
)


# Add external CSS for removing margin and padding
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        <title>MO Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Cantarell:wght@400;700&display=swap" rel="stylesheet">
        <style>
            body, html {
                margin: 0;
                padding: 0;
                height: 100%;
                overflow: hidden;
                font-family: 'Cantarell', sans-serif;  
            }
            #react-entry-point {
                height: 100%;
            }
        </style>
    </head>
    <body>
        <div id="react-entry-point">
            {%app_entry%}
        </div>
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


# ========== CALLBACKS TRIGGERED BY EXP_NAME SELECTION ==========
@app.callback(
    Output("reward-grid", "columnDefs"),
    Output("reward-grid", "rowData"),
    Input("multi-choice-dropdown", "value"),
)
def update_output(selected_options):
    global active_exp_names, active_df_latest

    active_exp_names = [] if selected_options is None else selected_options
    if len(active_exp_names) == 0:
        return [], []
    else:
        column_defs = [{"field": RUN_ID, "filter": True}]
        # Add reward values, showing only 2 decimal places
        for reward_name in REWARD_NAMES:
            column_defs.append(
                {
                    "field": reward_name,
                    "valueFormatter": {"function": 'd3.format(".2f")(params.value)'},
                }
            )

        cols_to_exclude = ["Source Preference", "Target Preference"]
        if len(active_exp_names) == 1:
            cols_to_exclude.append(EXP_NAME)
        else:
            column_defs = [{"field": EXP_NAME, "filter": True}] + column_defs

        active_df_latest = df_latest.loc[df_latest[EXP_NAME].isin(active_exp_names)]
        row_data = active_df_latest.loc[
            :, ~df_latest.columns.isin(cols_to_exclude)
        ].to_dict("records")

        return column_defs, row_data


# ========== CALLBACKS TRIGGERED BY RUN SELECTION ==========
@app.callback(Output("pareto-front", "figure"), Input("reward-grid", "selectedRows"))
def update_graph(selected_rows):
    if selected_rows is None:
        raise PreventUpdate

    # Determine marker sizes
    active_df_latest["size"] = 3
    if len(selected_rows) == 1:
        mask = active_df_latest[RUN_ID] == selected_rows[0][RUN_ID]
        active_df_latest.loc[mask, "size"] = 20

    fig = px.scatter(
        active_df_latest,
        x=REWARD_NAMES[0],
        y=REWARD_NAMES[1],
        size="size",
        color=TARGET_PREF,
        symbol=SOURCE_PREF,
        hover_data=[EXP_NAME, RUN_ID],  # Show experiment and run info on hover
    )
    fig.update_layout(title="Approx. Pareto Front", title_x=0.5, autosize=True)

    return fig


@app.callback(Output("episode-video", "children"), Input("reward-grid", "selectedRows"))
def update_line_chart(selected_rows):
    if selected_rows is None or len(selected_rows) != 1:
        raise PreventUpdate

    run_id = selected_rows[0][RUN_ID]
    div_children = [
        # get_heading("Policy Behaviour", h=2),
        html.Video(
            id="design-video",
            autoPlay=True,
            loop=True,
            controls=True,
            src=os.path.join("assets", "videos", f"{run_id}_last.mp4"),
            style={"width": "100%"},
        ),
    ]

    return div_children


@app.callback(Output("actions", "children"), Input("reward-grid", "selectedRows"))
def update_actions(selected_rows):
    if selected_rows is None or len(selected_rows) != 1:
        raise PreventUpdate

    values = load_state_or_action(run_id=selected_rows[0][RUN_ID], which="action")

    # Create figure (1 violin plot per state dimension)
    fig = go.Figure()
    for col in values.columns:
        fig.add_trace(
            go.Violin(x=values[col], orientation="h", name=col, meanline_visible=True)
        )
    fig.update_layout(
        title="Action Distribution",
        title_x=0.5,
        margin=dict(b=70, t=50, l=20, r=20),
        autosize=True,
        showlegend=False,
    )

    div_children = [dcc.Graph(id="action-dist", figure=fig, style={"height": "100%"})]

    return div_children


@app.callback(Output("states", "children"), Input("reward-grid", "selectedRows"))
def update_states(selected_rows):
    if selected_rows is None or len(selected_rows) != 1:
        raise PreventUpdate

    state_values = load_state_or_action(run_id=selected_rows[0][RUN_ID], which="state")
    fig = go.Figure()
    for col in state_values.columns:
        violin_trace = go.Violin(
            x=state_values[col],
            orientation="h",
            name=col,
            meanline_visible=True,
        )
        fig.add_trace(violin_trace)
    fig.update_layout(
        title="State Distribution",
        title_x=0.5,
        margin=dict(t=50, l=20, b=20, r=0),
        autosize=True,
        showlegend=False,
    )

    div_children = [dcc.Graph(id="state-dist", figure=fig, style={"height": "100%"})]

    return div_children


@app.callback(Output("design", "children"), Input("reward-grid", "selectedRows"))
def update_designs(selected_rows):
    if selected_rows is None or len(selected_rows) != 1:
        raise PreventUpdate

    run_id = selected_rows[0][RUN_ID]
    active_df_design = df_design.loc[df_design[EXP_NAME].isin(active_exp_names)]
    active_df_design["color"] = active_df_design[RUN_ID].apply(
        lambda x: 1 if x == run_id else 0
    )

    # Create the parallel coordinates plot
    fig = px.parallel_coordinates(
        active_df_design,
        dimensions=[str(i) for i in range(N_DESIGN_PARAMETERS)],
        color="color",
        color_continuous_scale=[[0, "gray"], [1, "red"]],
        range_color=[0, 1],  # Define range to match the Highlight values
    )

    fig.update_layout(
        title="Design Parameters",
        title_x=0.5,
        margin=dict(t=100, b=50, l=20, r=20),
        autosize=True,
        showlegend=False,
        coloraxis_showscale=False,
    )

    div_children = [dcc.Graph(id="action-dist", figure=fig, style={"height": "100%"})]

    return div_children


if __name__ == "__main__":
    app.run(debug=True)
