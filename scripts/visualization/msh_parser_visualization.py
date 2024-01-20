import os
import plotly.graph_objs as go
import numpy as np
import data.simulation.ansys_msh_parser as AMP
from scripts.helpers import load_config

# https://plotly.com/python/v3/3d-network-graph/

def plot_outlet_area(msh_dict):
    # generate nodes
    x = [
        msh_dict["points"]["point_data"][i]["xyz"][0]
        for i in msh_dict["points"]["point_data"]
        if msh_dict["points"]["point_data"][i]["xyz"][0] < 0.05
    ]
    y = [
        msh_dict["points"]["point_data"][i]["xyz"][1]
        for i in msh_dict["points"]["point_data"]
        if msh_dict["points"]["point_data"][i]["xyz"][0] < 0.05
    ]
    z = [
        msh_dict["points"]["point_data"][i]["xyz"][2]
        for i in msh_dict["points"]["point_data"]
        if msh_dict["points"]["point_data"][i]["xyz"][0] < 0.05
    ]

    # generate edges
    e_x = []
    e_y = []
    e_z = []

    for i in msh_dict["points"]["point_data"]:
        for node in msh_dict["points"]["point_data"][i]["connected_nodes"]:
            if (msh_dict["points"]["point_data"][i]["xyz"][0] < 0.05) and (
                msh_dict["points"]["point_data"][node]["xyz"][0] < 0.05
            ):
                edge_x = [
                    msh_dict["points"]["point_data"][i]["xyz"][0],
                    msh_dict["points"]["point_data"][node]["xyz"][0],
                    None,
                ]
                edge_y = [
                    msh_dict["points"]["point_data"][i]["xyz"][1],
                    msh_dict["points"]["point_data"][node]["xyz"][1],
                    None,
                ]
                edge_z = [
                    msh_dict["points"]["point_data"][i]["xyz"][2],
                    msh_dict["points"]["point_data"][node]["xyz"][2],
                    None,
                ]

                e_x += edge_x
                e_y += edge_y
                e_z += edge_z

    p1 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=5,
            color=x,  # set color to an array/list of desired values
            colorscale="Viridis",  # choose a colorscale
            opacity=0.8,
        ),
    )

    p2 = go.Scatter3d(
        x=e_x,
        y=e_y,
        z=e_z,
        mode="lines",
        line=dict(
            width=2,
            color="rgb(50,50,50)",  # set color to an array/list of desired values
        ),
    )

    fig = go.Figure(data=[p1, p2])
    fig.show()


def plot_full_volume_mesh(msh_dict):
    # generate nodes
    x = [
        msh_dict["points"]["point_data"][i]["xyz"][0]
        for i in msh_dict["points"]["point_data"]
    ]
    y = [
        msh_dict["points"]["point_data"][i]["xyz"][1]
        for i in msh_dict["points"]["point_data"]
    ]
    z = [
        msh_dict["points"]["point_data"][i]["xyz"][2]
        for i in msh_dict["points"]["point_data"]
    ]

    # generate edges
    e_x = []
    e_y = []
    e_z = []

    for i in msh_dict["points"]["point_data"]:
        for node in msh_dict["points"]["point_data"][i]["connected_nodes"]:
            edge_x = [
                msh_dict["points"]["point_data"][i]["xyz"][0],
                msh_dict["points"]["point_data"][node]["xyz"][0],
                None,
            ]
            edge_y = [
                msh_dict["points"]["point_data"][i]["xyz"][1],
                msh_dict["points"]["point_data"][node]["xyz"][1],
                None,
            ]
            edge_z = [
                msh_dict["points"]["point_data"][i]["xyz"][2],
                msh_dict["points"]["point_data"][node]["xyz"][2],
                None,
            ]

            e_x += edge_x
            e_y += edge_y
            e_z += edge_z

    p1 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=2,
            color=x,  # set color to an array/list of desired values
            colorscale="Viridis",  # choose a colorscale
            opacity=0.8,
        ),
    )

    p2 = go.Scatter3d(
        x=e_x,
        y=e_y,
        z=e_z,
        mode="lines",
        line=dict(
            width=1,
            color="rgb(50,50,50)",  # set color to an array/list of desired values
        ),
    )

    fig = go.Figure(data=[p1, p2])
    fig.show()


def plot_mesh_nodes_w_zones(msh_dict):
    # generate nodes
    x = [
        msh_dict["points"]["point_data"][i]["xyz"][0]
        for i in msh_dict["points"]["point_data"]
    ]
    y = [
        msh_dict["points"]["point_data"][i]["xyz"][1]
        for i in msh_dict["points"]["point_data"]
    ]
    z = [
        msh_dict["points"]["point_data"][i]["xyz"][2]
        for i in msh_dict["points"]["point_data"]
    ]

    # generate edges
    e_x = []
    e_y = []
    e_z = []

    for i in msh_dict["points"]["point_data"]:
        for node in msh_dict["points"]["point_data"][i]["connected_nodes"]:
            edge_x = [
                msh_dict["points"]["point_data"][i]["xyz"][0],
                msh_dict["points"]["point_data"][node]["xyz"][0],
                None,
            ]
            edge_y = [
                msh_dict["points"]["point_data"][i]["xyz"][1],
                msh_dict["points"]["point_data"][node]["xyz"][1],
                None,
            ]
            edge_z = [
                msh_dict["points"]["point_data"][i]["xyz"][2],
                msh_dict["points"]["point_data"][node]["xyz"][2],
                None,
            ]

            e_x += edge_x
            e_y += edge_y
            e_z += edge_z

    node_colors = []
    for i in msh_dict["points"]["point_data"]:
        if "wall" in msh_dict["points"]["point_data"][i]["zone_name"]:
            node_colors.extend(["green"])

        elif "inlet" in msh_dict["points"]["point_data"][i]["zone_name"]:
            node_colors.extend(["red"])

        elif "outlet" in msh_dict["points"]["point_data"][i]["zone_name"]:
            node_colors.extend(["yellow"])

        elif "interior" in msh_dict["points"]["point_data"][i]["zone_name"]:
            node_colors.extend(["blue"])

        else:
            node_colors.extend(["black"])

    p1 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=3,
            color=node_colors,  # set color to an array/list of desired values
            opacity=0.8,
        ),
    )

    fig = go.Figure(data=[p1])
    fig.show()


if __name__ == "__main__":
    config = load_config(
        os.path.join("scripts", "visualizations", "msh_parser_visualization.json")
    )
    thing_to_plot = config.thingToPlot

    demo_file_path = config.demoFilePath
    _, msh_dict = AMP.read(demo_file_path)

    if thing_to_plot == "outlet":
        plot_outlet_area(msh_dict)

    elif thing_to_plot == "mesh":
        plot_full_volume_mesh(msh_dict)

    else:
        plot_mesh_nodes_w_zones(msh_dict)
