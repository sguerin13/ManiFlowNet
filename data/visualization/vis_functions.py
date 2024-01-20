import copy
import os

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from plotly.tools import make_subplots
from pyntcloud import PyntCloud as PC

import data.wrangling.data_utils as DU


def display_mesh(file_name):
    # read obj mesh files
    o3d_mesh = o3d.io.read_triangle_mesh(file_name, print_progress=True)
    o3d_mesh.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([o3d_mesh])
    return


def display_pnt_cloud(
    file_name, size=None, opacity=None, zone_filter=None, n_points=None
):
    pnt_cld = DU.read_ply(file_name)
    if n_points:
        pnt_cld = pnt_cld.sample(n_points)
    _point_cloud_plot(pnt_cld, size, opacity, zone_filter)


def display_voxel_grid(file_name):
    pnt_cld = PC.from_file(file_name)
    grid_id = pnt_cld.add_structure("voxelgrid", n_x=64, n_y=64, n_z=64)
    voxel_grid = pnt_cld.structures[grid_id]
    return voxel_grid.plot(d=3, mode="density", cmap="hsv")


def _point_cloud_plot(pnt_cld, size=4, opacity=0.7, zone_filter=None):
    zone_dict = {"inlet": 5, "outlet": 6, "wall": 7, "fluid": 1}
    if zone_filter:
        zone_dict_list = [zone_dict[i] for i in zone_filter]
        pnt_cld = pnt_cld[pnt_cld["zone"].isin(zone_dict_list)]

    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]])
    params = [(row["zone"]) for i, row in pnt_cld.iterrows()]
    fig.add_traces(
        [
            go.Scatter3d(
                x=pnt_cld["x"],
                y=pnt_cld["y"],
                z=pnt_cld["z"],
                mode="markers",
                hovertext=params,
                marker=dict(
                    size=size,
                    # set color to an array/list of desired values
                    color=pnt_cld["zone"],
                    colorscale="Viridis",  # choose a colorscale
                    opacity=opacity,
                ),
            )
        ],
        rows=[1],
        cols=[1],
    )
    fig.data[0].visible = True

    # # Create and add slider
    # steps = []
    # for i in range(int(len(fig.data))):
    #     step = dict(
    #         method="update",
    #         args=[{"visible": [False] * len(fig.data)},
    #             {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    #     )
    #     step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    #     # step["args"][0]["visible"][i+20] = True # TODO: see what this is about
    #     steps.append(step)

    # sliders = [dict(
    #     active=0,
    #     currentvalue={"prefix": "X-Bin: "},
    #     pad={"t": 50},
    #     steps=steps
    # )]

    # fig.update_layout(
    #     sliders=sliders,
    #     title_text="Pressure Values"
    # )
    fig.update_layout(scene_aspectmode="data")
    fig.show()


def _point_cloud_input_plot(pnt_cld, zone_filter=None):
    if zone_filter:
        if zone_filter == "inlet_outlet":
            pnt_cld = pnt_cld[
                (pnt_cld["zone_inlet"] == 1) | (pnt_cld["zone_outlet"] == 1)
            ]

    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]])
    fig.add_traces(
        [
            go.Scatter3d(
                x=pnt_cld["x"],
                y=pnt_cld["y"],
                z=pnt_cld["z"],
                mode="markers",
                marker=dict(
                    size=5,
                    # set color to an array/list of desired values
                    color=_hash_zones_to_colors(
                        pnt_cld[["zone_inlet", "zone_outlet"]].values.tolist()
                    ),
                    colorscale="Viridis",  # choose a colorscale
                    opacity=0.9,
                ),
            )
        ],
        rows=[1],
        cols=[1],
    )
    fig.data[0].visible = True

    # # Create and add slider
    # steps = []
    # for i in range(int(len(fig.data))):
    #     step = dict(
    #         method="update",
    #         args=[{"visible": [False] * len(fig.data)},
    #             {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    #     )
    #     step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    #     # step["args"][0]["visible"][i+20] = True # TODO: see what this is about
    #     steps.append(step)

    # sliders = [dict(
    #     active=0,
    #     currentvalue={"prefix": "X-Bin: "},
    #     pad={"t": 50},
    #     steps=steps
    # )]

    # fig.update_layout(
    #     sliders=sliders,
    #     title_text="Pressure Values"
    # )
    fig.update_layout(scene_aspectmode="data")
    fig.show()


def _hash_zones_to_colors(zones, mode="inlet_outlet"):
    color_list = []
    if mode == "inlet_outlet":
        for row in zones:
            if row[0] == 1:
                color_list.append(0)
            else:
                color_list.append(1)
    return color_list


def get_point_cloud_scale(pnt_cld):
    x_scale_min = min(pnt_cld["x-velocity"])
    x_scale_max = max(pnt_cld["x-velocity"])
    x_scale_tuple = (x_scale_min, x_scale_max)

    y_scale_min = min(pnt_cld["y-velocity"])
    y_scale_max = max(pnt_cld["y-velocity"])
    y_scale_tuple = (y_scale_min, y_scale_max)

    z_scale_min = min(pnt_cld["z-velocity"])
    z_scale_max = max(pnt_cld["z-velocity"])
    z_scale_tuple = (z_scale_min, z_scale_max)

    p_scale_min = min(pnt_cld["pressure"])
    p_scale_max = max(pnt_cld["pressure"])
    p_scale_tuple = (p_scale_min, p_scale_max)

    scale_factor = abs(max(pnt_cld["x"]) - min(pnt_cld["x"]))

    return x_scale_tuple, y_scale_tuple, z_scale_tuple, p_scale_tuple, scale_factor


def visualize_sim_data_independent_vectors(fpath, include_wall=True):
    pnt_cld = DU.read_ply(os.path.join(fpath, "pnt_cld.ply"))
    if not include_wall:
        pnt_cld = pnt_cld[pnt_cld["zone"] != 7]
    (
        x_scale_tuple,
        y_scale_tuple,
        z_scale_tuple,
        p_scale_tuple,
        scale_factor,
    ) = get_point_cloud_scale(pnt_cld)

    pnt_cld_value_dict = {
        "x-velocity": [],
        "y-velocity": [],
        "z-velocity": [],
        "pressure": [],
    }
    for metric in ["x-velocity", "y-velocity", "z-velocity", "pressure"]:
        if metric == "x-velocity":
            scale_min = x_scale_tuple[0]
            scale_max = x_scale_tuple[1]

        if metric == "y-velocity":
            scale_min = y_scale_tuple[0]
            scale_max = y_scale_tuple[1]

        if metric == "z-velocity":
            scale_min = z_scale_tuple[0]
            scale_max = z_scale_tuple[1]

        if metric == "pressure":
            scale_min = p_scale_tuple[0]
            scale_max = p_scale_tuple[1]

        for step in np.arange(0, scale_factor, scale_factor / 20):
            pnt_cld_at_x = pnt_cld[pnt_cld["x"] < step]
            pnt_cld_value_dict[metric].append(
                go.Scatter3d(
                    x=pnt_cld_at_x["x"],
                    y=pnt_cld_at_x["y"],
                    z=pnt_cld_at_x["z"],
                    mode="markers",
                    visible=False,
                    scene="scene1",
                    marker=dict(
                        size=4,
                        showscale=True,
                        symbol="square",
                        cmin=scale_min,
                        cmax=scale_max,
                        color=pnt_cld_at_x[
                            metric
                        ],  # set color to an array/list of desired values
                        colorscale="Viridis",  # choose a colorscale
                        opacity=0.5,
                    ),
                    text=[metric + ": {}".format(x) for x in pnt_cld_at_x[metric]],
                )
            )

    # build up the dictionary for the slider
    data = (
        pnt_cld_value_dict["x-velocity"]
        + pnt_cld_value_dict["y-velocity"]
        + pnt_cld_value_dict["z-velocity"]
        + pnt_cld_value_dict["pressure"]
    )
    step_len = len(pnt_cld_value_dict["x-velocity"])
    steps = {"X": [], "Y": [], "Z": [], "P": []}

    for step in range(step_len):
        x_velo_false, y_velo_false, z_velo_false, pressure_false = (
            [False] * (step_len),
            [False] * (step_len),
            [False] * (step_len),
            [False] * (step_len),
        )
        x_velo_true, y_velo_true, z_velo_true, pressure_true = (
            copy.copy(x_velo_false),
            copy.copy(y_velo_false),
            copy.copy(z_velo_false),
            copy.copy(pressure_false),
        )
        x_velo_true[step], y_velo_true[step], z_velo_true[step], pressure_true[step] = (
            True,
            True,
            True,
            True,
        )

        step_x = dict(
            method="update",
            args=[
                {"visible": x_velo_true + y_velo_false + z_velo_false + pressure_false},
                {"title": "X step: {}".format(step)},
            ],
        )
        step_y = dict(
            method="update",
            args=[
                {"visible": x_velo_false + y_velo_true + z_velo_false + pressure_false},
                {"title": "Y step: {}".format(step)},
            ],
        )

        step_z = dict(
            method="update",
            args=[
                {"visible": x_velo_false + y_velo_false + z_velo_true + pressure_false},
                {"title": "Z step: {}".format(step)},
            ],
        )

        step_p = dict(
            method="update",
            args=[
                {"visible": x_velo_false + y_velo_false + z_velo_false + pressure_true},
                {"title": "P step: {}".format(step)},
            ],
        )

        steps["X"].append(step_x)
        steps["Y"].append(step_y)
        steps["Z"].append(step_z)
        steps["P"].append(step_p)

    sliders = {}
    for key, traces in steps.items():
        slider = [
            dict(
                active=0,
                currentvalue=dict(prefix="Step: "),
                pad=dict(t=50),
                steps=traces,
            )
        ]
        sliders[key] = slider

    # building up the button list
    buttons = []
    for key, slider in sliders.items():
        slider_active = slider[0]["active"]
        slider_visible = slider[0]["steps"][slider_active]["args"][0]["visible"]
        button = dict(
            label=key,
            method="update",
            args=[
                dict(visible=slider_visible),
                dict(
                    title="{} step: {}".format(key, slider_active),
                    yaxis=dict(title="y {}".format(key)),
                    sliders=slider,
                ),
            ],
        )
        # animate_button = dict(label="Animate" + key,
        #                       frame_dur
        #                       method="animate",
        #                       args=[dict(data=steps[key])])

        buttons.append(button)

    updatemenus = [dict(active=0, type="buttons", buttons=buttons)]

    layout = go.Layout(
        title="X step: 0",
        font=dict(size=16),
        hovermode="x unified",
        hoverlabel=dict(font_size=16),
        sliders=sliders["X"],
        updatemenus=updatemenus,
        showlegend=False,
    )

    fig = go.Figure(dict(data=data, layout=layout))
    fig.set_subplots(
        1,
        1,
        specs=[[{"type": "scene"}]],
        subplot_titles=("Output", "Ground Truth", "Delta"),
        horizontal_spacing=0.1,
    )
    fig.update_scenes(aspectmode="data")
    fig.show()


def get_point_cloud_magnitude_scale(pnt_cld):
    v_scale_min = min(pnt_cld["velo-magnitude"])
    v_scale_max = max(pnt_cld["velo-magnitude"])
    v_scale_tuple = (v_scale_min, v_scale_max)

    p_scale_min = min([min(pnt_cld["pressure"]), min(pnt_cld["pressure"])])
    p_scale_max = max([max(pnt_cld["pressure"]), max(pnt_cld["pressure"])])
    p_scale_tuple = (p_scale_min, p_scale_max)

    scale_factor = abs(max(pnt_cld["x"]) - min(pnt_cld["x"]))

    return v_scale_tuple, p_scale_tuple, scale_factor


def visualize_sim_data_magnitudes(fpath, include_wall=True):
    pnt_cld = DU.read_ply(os.path.join(fpath, "pnt_cld.ply"))
    if not include_wall:
        pnt_cld = pnt_cld[pnt_cld["zone"] != 7]

    pnt_cld["velo-magnitude"] = np.linalg.norm(
        pnt_cld[["x-velocity", "y-velocity", "z-velocity"]].values, axis=1
    )
    v_scale_tuple, p_scale_tuple, scale_factor = get_point_cloud_magnitude_scale(
        pnt_cld
    )

    pnt_cld_value_dict = {"velo-magnitude": [], "pressure": []}
    for metric in ["velo-magnitude", "pressure"]:
        if metric == "velo-magnitude":
            scale_min = v_scale_tuple[0]
            scale_max = v_scale_tuple[1]

        if metric == "pressure":
            scale_min = p_scale_tuple[0]
            scale_max = p_scale_tuple[1]

        for step in np.arange(0, scale_factor * 1.1, scale_factor / 20):
            pnt_cld_at_x = pnt_cld[pnt_cld["x"] < step]
            pnt_cld_value_dict[metric].append(
                go.Scatter3d(
                    x=pnt_cld_at_x["x"],
                    y=pnt_cld_at_x["y"],
                    z=pnt_cld_at_x["z"],
                    mode="markers",
                    visible=False,
                    scene="scene1" if metric == "velo-magnitude" else "scene2",
                    marker=dict(
                        size=4,
                        showscale=True,
                        symbol="square",
                        cmin=scale_min,
                        cmax=scale_max,
                        colorbar={"x": 0.5 if metric == "velo-magnitude" else 1},
                        color=pnt_cld_at_x[
                            metric
                        ],  # set color to an array/list of desired values
                        colorscale="Viridis",  # choose a colorscale
                        opacity=0.5,
                    ),
                    text=[metric + ": {}".format(x) for x in pnt_cld_at_x[metric]],
                )
            )

    # build up the dictionary for the slider
    data = pnt_cld_value_dict["velo-magnitude"] + pnt_cld_value_dict["pressure"]
    step_len = len(pnt_cld_value_dict["velo-magnitude"])
    steps = {"V": []}

    for step in range(step_len):
        velo_false, pressure_false = [False] * (step_len), [False] * (step_len)
        velo_true, pressure_true = copy.copy(velo_false), copy.copy(pressure_false)
        velo_true[step], pressure_true[step] = True, True

        step_v = dict(
            method="update",
            args=[
                {"visible": velo_true + pressure_true},
                {"title": "step: {}".format(step)},
            ],
        )

        steps["V"].append(step_v)

    sliders = {}
    for key, traces in steps.items():
        slider = [
            dict(
                active=0,
                currentvalue=dict(prefix="Step: "),
                pad=dict(t=50),
                steps=traces,
            )
        ]
        sliders[key] = slider

    layout = go.Layout(
        title="X step: 0",
        font=dict(size=16),
        hovermode="x unified",
        hoverlabel=dict(font_size=16),
        sliders=sliders["V"],
        showlegend=False,
    )

    fig = go.Figure(dict(data=data, layout=layout))
    fig.set_subplots(
        1,
        2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Velocity", "Pressure"),
        horizontal_spacing=0.1,
    )
    fig.update_scenes(aspectmode="data")
    fig.show()
    return fig
