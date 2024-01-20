import json
import os
import pickle
import warnings
from types import SimpleNamespace

import pytorch_lightning as pl
import requests
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    early_stopping,
    lr_monitor,
)
from torch.utils.data import DataLoader
from neural.models.convpoint.Dataset import Dataset
from neural.models.convpoint.PLConvPointUNet import PLConvPointUNet
from scripts.helpers import load_config

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    config = load_config(
        os.path.join("scripts", "train", "convpoint", "config", "train.json")
    )

    experiment_name = config.experimentName
    tparams = config.trainParams
    fpaths = config.filePaths
    message_config = config.messageConfig
    network_config = load_config(
        os.path.join(
            "scripts",
            "train",
            "convpoint",
            "config",
            "network_config",
            experiment_name + ".json",
        )
    )
    
    sample_qty = network_config.sampleQty
    batch_size = network_config.batchSize
    multiple_branches = network_config.multipleBranches
    n_centers = network_config.nCenters
    encoder_blocks = network_config.encoderBlocks
    encoder_channels = network_config.encoderChannels
    decoder_blocks = network_config.decoderBlocks
    decoder_channels = network_config.decoderChannels
    first_conv_channel = network_config.firstConvChannel
    n_inlet_outlet_layers = network_config.nInletOutletLayers
    non_linearity = network_config.nonLinearity
    n_pts = network_config.nPts
    lr = network_config.lr
    reduction = network_config.reduction
    val_loss = network_config.valLoss
    train_loss = network_config.trainLoss
    scaler_type = network_config.scalerType
    surface_nodes = network_config.surfaceNodesOnly
    add_normals = network_config.includeSurfaceNormals
    rotate = network_config.randomRotation
    non_dimensionalize = network_config.nonDimensionalize
    output_layer_channel = network_config.outputLayerChannel
    VP_all_inputs = network_config.VPAllInputs

    if non_dimensionalize:
        assert scaler_type in ["ndim", "ndim_all_fps"]
        assert VP_all_inputs == False

    # drop_learning rate by a min factor of 32
    min_lr = lr / 32.0  # TODO: make this a config option

    stopping_patience = tparams.earlyStoppingPatience
    min_delta = tparams.minDelta
    lr_patience = tparams.lrPatience
    terminate_on_non = tparams.terminateOnNan
    max_epochs = tparams.maxEpochs
    gpus = tparams.gpus
    refresh_rate = tparams.refreshRate
    verbose_nans = tparams.verboseNans

    train_path = os.path.join(fpaths.dataPath, "train")
    val_path = os.path.join(fpaths.dataPath, "val")
    exp_path = os.path.join(fpaths.experimentRootPath, experiment_name)
    scaler_path = fpaths.scalerPath

    with open(os.path.join(scaler_path, scaler_type, "bbox_scaler.pkl"), "rb") as f:
        bbox_scaler = pickle.load(f)

    with open(os.path.join(scaler_path, scaler_type, "velo_scaler.pkl"), "rb") as f:
        velo_scaler = pickle.load(f)

    with open(os.path.join(scaler_path, scaler_type, "P_scaler.pkl"), "rb") as f:
        pressure_scaler = pickle.load(f)

    with open(
        os.path.join(scaler_path, scaler_type, "fluid_prop_scaler.pkl"), "rb"
    ) as f:
        fluid_prop_scaler = pickle.load(f)

    with open(os.path.join(scaler_path, scaler_type, "props_included.pkl"), "rb") as f:
        context_values_included = pickle.load(f)

    train_set = Dataset(
        root=train_path,
        bbox_scaler=bbox_scaler,
        velo_scaler=velo_scaler,
        pressure_scaler=pressure_scaler,
        fparams_scaler=fluid_prop_scaler,
        sample_qty=sample_qty,
        non_dimensionalize=non_dimensionalize,
        rotate=rotate,
        surface_nodes=surface_nodes,
        context_values_included=context_values_included,
        add_normals=add_normals,
        VP_all_inputs=VP_all_inputs,
    )

    val_set = Dataset(
        root=val_path,
        bbox_scaler=bbox_scaler,
        velo_scaler=velo_scaler,
        pressure_scaler=pressure_scaler,
        fparams_scaler=fluid_prop_scaler,
        sample_qty=sample_qty,
        non_dimensionalize=non_dimensionalize,
        rotate=rotate,
        surface_nodes=surface_nodes,
        context_values_included=context_values_included,
        add_normals=add_normals,
        VP_all_inputs=VP_all_inputs,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=12,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=12,
        shuffle=False,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=min_delta, patience=stopping_patience, mode="min"
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_monitor = ModelCheckpoint(
        dirpath=exp_path, filename="val_best", mode="min", monitor="val_loss"
    )

    callbacks = [lr_monitor, early_stopping, checkpoint_monitor]

    # CALCULATE INPUT CHANNEL
    channel_in = 4  # wall, inlet, outlet, fluid
    channel_in += (
        len(context_values_included) + 1
    )  # [Re, Eps, rho, Visc, D] + scale or [Re, Eps] + scale

    if add_normals:
        channel_in += 3  # nx,ny,nz

    if surface_nodes:
        channel_in -= 1  # removing nodes in fluid zone

    if not non_dimensionalize:
        if VP_all_inputs:
            channel_in += 2  # inlet V, outlet P Provided to all nodes
        else:
            channel_in += 4  # vx, vy, vz, p - masked at certain nodes

    pl_model = PLConvPointUNet(
        sample_qty=sample_qty,
        multiple_branches=multiple_branches,
        channel_in=channel_in,
        first_conv_channel=first_conv_channel,
        n_inlet_layers=n_inlet_outlet_layers,
        encoder_blocks=encoder_blocks,
        encoder_channel=encoder_channels,
        decoder_blocks=decoder_blocks,
        decoder_channel=decoder_channels,
        n_outlet_layers=n_inlet_outlet_layers,
        output_layer_channel=output_layer_channel,
        nout=4,
        non_linearity=non_linearity,
        n_pts=n_pts,
        n_centers=n_centers,
        lr=lr,
        min_lr=min_lr,
        train_loss=train_loss,
        val_loss=val_loss,
        overfit=False,
        reduction="mean",
        lr_patience=lr_patience,
    )

    try:
        torch.cuda.empty_cache()
        trainer = pl.Trainer(
            detect_anomaly=True,
            gpus=1,
            max_epochs=max_epochs,
            callbacks=callbacks,
            log_every_n_steps=10,
            default_root_dir=exp_path,
        )

        trainer.fit(pl_model, train_loader, val_loader)
        if message_config.sendMessage:
            r = requests.post(
                "https://textbelt.com/text",
                json={
                    "number": message_config.phoneNumber,
                    "message": "Training Completed",
                    "key": message_config.textBeltKey,
                },
            )

    except Exception as e:
        print("training failed", e)
        if message_config.sendMessage:
            r = requests.post(
                "https://textbelt.com/text",
                json={
                    "number": message_config.phoneNumber,
                    "message": "Training Failed",
                    "key": message_config.textBeltKey,
                },
            )
