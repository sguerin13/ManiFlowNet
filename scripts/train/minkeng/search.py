import os
import pickle

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader

from neural.models.minkeng.Dataset import Dataset, custom_collate_fn
from neural.models.minkeng.PLMinkEngUNet import PLMinkEngUNet


class MEObjective(object):
    def __init__(
        self,
        train_path,
        val_path,
        current_exp_folder,
        scaler_root_path,
        n_voxels,
        surface_only,
        rotate,
        normals,
        VP_all_inputs,
        overfit,
        non_dimensionalize,
        max_epochs,
        early_stopping_patience,
        lr_scheduler_patience,
        multiple_branches,
    ):
        self.train_path = train_path
        self.n_voxels = n_voxels
        self.val_path = val_path
        self.current_exp_folder = current_exp_folder
        self.scaler_root_path = scaler_root_path
        self.n_voxels = n_voxels
        self.max_levels = 8
        self.surface_only = surface_only
        self.rotate = rotate
        self.normals = normals
        self.VP_all_inputs = VP_all_inputs
        self.non_dimensionalize = non_dimensionalize
        self.overfit = overfit
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler_patience = lr_scheduler_patience
        self.multiple_branches = multiple_branches

    def __call__(self, trial):
        print("TRIAL STEP")

        ######################################
        #            Sample Quantity         #
        ######################################
        sample_qty = trial.suggest_categorical(
            "sample_qty", [None]
        )  # use all input points

        #############################################
        #           Input/Output  Block             #
        #############################################

        if self.overfit:
            batch_size = 1

        else:
            batch_size = trial.suggest_int("batch_size", low=2, high=8, step=2)

        n_inlet_outlet_layers = trial.suggest_int(
            name="n_inlet_outlet_layers", low=1, high=3
        )
        inlet_outlet_conv_channel = trial.suggest_int(
            name="input_channel", low=48, high=144, step=48
        )
        # first_conv_channel = trial.suggest_int(name = "input_channel",low=16,high=128,step=8)
        # first_conv_channel = trial.suggest_int(name = "input_channel",low=24,high=156,step=24)
        # n_inlet_layers = trial.suggest_int(name="n_inlet_layers",low = 1,high = 3)

        # ######################################
        # #           Output Block             #
        # ######################################
        # n_outlet_layers = trial.suggest_int("n_outlet_layers",1,3)
        # # output_layer_channel = trial.suggest_int("output_channel",low=16,high=128,step=8)
        # output_layer_channel = trial.suggest_int("output_channel",low=24,high=96,step=24)

        # n_levels = trial.suggest_int("UNET_LEVELS", 6, self.max_levels - 1)
        n_levels = 8
        ######################################
        #   N Conv Blocks and Channel Size   #
        ######################################

        if n_levels == 8:
            if self.multiple_branches:
                n_blocks = [[1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2]]
                block_index = trial.suggest_int("block_ind", 0, 1)
                encoder_blocks = n_blocks[block_index]
                decoder_blocks = n_blocks[block_index]

                channel_index = trial.suggest_int("channel_ind", 0, 2)

                encoder_channels_arrays = [
                    [32, 32, 64, 64, 128, 128, 256, 256],
                    [64, 64, 128, 128, 256, 256, 512, 512],
                    [64, 64, 128, 128, 256, 256, 512, 1024],
                ]

                decoder_channels_arrays = [
                    [256, 256, 128, 128, 64, 64, 32, 32],
                    [512, 512, 256, 256, 128, 128, 64, 64],
                    [1024, 512, 256, 256, 128, 128, 64, 64],
                ]

            else:
                n_blocks = [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [2, 2, 2, 2, 2, 2, 2, 2],
                    [4, 4, 4, 4, 4, 4, 4, 4],
                ]
                block_index = trial.suggest_int("block_ind", 0, 2)
                encoder_blocks = n_blocks[block_index]
                decoder_blocks = n_blocks[block_index]

                channel_index = trial.suggest_int("channel_ind", 0, 3)

                encoder_channels_arrays = [
                    [32, 32, 64, 64, 128, 128, 256, 256],
                    [64, 64, 128, 128, 256, 256, 512, 512],
                    [64, 64, 128, 128, 256, 256, 512, 1024],
                    [128, 128, 256, 256, 512, 512, 1024, 2048],
                ]

                decoder_channels_arrays = [
                    [256, 256, 128, 128, 64, 64, 32, 32],
                    [512, 512, 256, 256, 128, 128, 64, 64],
                    [1024, 512, 256, 256, 128, 128, 64, 64],
                    [2048, 1024, 512, 512, 256, 256, 128, 128],
                ]

            encoder_channel = encoder_channels_arrays[channel_index]
            decoder_channel = decoder_channels_arrays[channel_index]

        ######################################
        #   Kernels and strides for blocks   #
        ######################################
        # base_kernel_size = trial.suggest_int("base_conv_kernel_size",2,3)
        # base_stride_size = trial.suggest_int("base_stride_size",1,1)

        ######################################
        #               Pooling              #
        ######################################

        #######################################
        # Up/DownSampling Kernels and Strides #
        #######################################

        #####################################
        #              SCALERS              #
        #####################################

        with open(os.path.join(self.scaler_root_path, "bbox_scaler.pkl"), "rb") as f:
            bbox_scaler = pickle.load(f)

        with open(os.path.join(self.scaler_root_path, "velo_scaler.pkl"), "rb") as f:
            velo_scaler = pickle.load(f)

        with open(os.path.join(self.scaler_root_path, "P_scaler.pkl"), "rb") as f:
            pressure_scaler = pickle.load(f)

        with open(
            os.path.join(self.scaler_root_path, "fluid_prop_scaler.pkl"), "rb"
        ) as f:
            fluid_prop_scaler = pickle.load(f)

        with open(os.path.join(self.scaler_root_path, "props_included.pkl"), "rb") as f:
            context_values_included = pickle.load(f)

        ######################################
        #           Learning Rate            #
        ######################################
        # lr = trial.suggest_float("learning_rate",low = 0.00001,high = .5,log=True)
        # lr = trial.suggest_float("learning_rate",low = 0.00005,high = .001,log=True)
        lr = 0.0003

        ######################################
        #           SURFACE ONLY             #
        ######################################
        if self.surface_only:
            surface_nodes_only = trial.suggest_categorical("surface only", [True])
        else:
            surface_nodes_only = trial.suggest_categorical("surface only", [False])

        ######################################
        #           INCLUDE NORMALS          #
        ######################################
        if self.normals == True:
            include_surface_normals = trial.suggest_categorical(
                "include_normals", [True]
            )
        elif self.normals == False:
            include_surface_normals = trial.suggest_categorical(
                "include_normals", [False]
            )
        else:
            include_surface_normals = trial.suggest_categorical(
                "include_normals", [True, False]
            )
        ######################################
        #             ROTATION               #
        ######################################
        if self.rotate == True:
            random_rotation = trial.suggest_categorical("random_rotation", [True])
        else:
            random_rotation = trial.suggest_categorical("random_rotation", [False])

        ######################################
        #          NON LINEARITY             #
        ######################################
        non_linearity = trial.suggest_categorical("non linearity", ["ELU", "RELU"])

        ######################################
        #             LOSS FN                #
        ######################################
        loss_reduction = trial.suggest_categorical("reduction", ["mean"])
        if self.surface_only:
            train_loss = trial.suggest_categorical("train_loss", ["MaskedMSE"])
            val_loss = trial.suggest_categorical("val_loss", ["MaskedWAPE"])
        else:
            # train_loss = trial.suggest_categorical("train_loss", ["MSE","MAE"])
            train_loss = "MAE"
            val_loss = trial.suggest_categorical("val_loss", ["WAPE"])

        ######################################
        #                LOG                 #
        ######################################

        # TODO: update
        hyperparameters = dict(
            n_voxels=self.n_voxels,
            sample_qty=sample_qty,
            batch_size=batch_size,
            first_conv_channel=inlet_outlet_conv_channel,
            n_inlet_layers=n_inlet_outlet_layers,
            n_levels=n_levels,
            encoder_blocks=encoder_blocks,
            encoder_channel=encoder_channel,
            decoder_blocks=decoder_blocks,
            decoder_channel=decoder_channel,
            non_dimensionalize=self.non_dimensionalize,
            context_values_included=context_values_included,
            multiple_branches=self.multiple_branches,
            #    base_kernel_size=base_kernel_size,
            #    base_stride_size=base_stride_size,
            n_outlet_layers=n_inlet_outlet_layers,
            output_layer_channel=inlet_outlet_conv_channel,
            scaler_type=self.scaler_root_path.split("/")[-1],
            lr=lr,
            surface_nodes_only=surface_nodes_only,
            include_surface_normals=include_surface_normals,
            random_rotation=random_rotation,
            VP_all_inputs=self.VP_all_inputs,
            train_loss=train_loss,
            val_loss=val_loss,
            non_linearity=non_linearity,
        )

        #####################################
        #             Loaders               #
        #####################################
        train_set = Dataset(
            root=self.train_path,
            n_voxels=self.n_voxels,
            bbox_scaler=bbox_scaler,
            velo_scaler=velo_scaler,
            pressure_scaler=pressure_scaler,
            fparams_scaler=fluid_prop_scaler,
            sample_qty=sample_qty,
            non_dimensionalize=self.non_dimensionalize,
            rotate=random_rotation,
            surface_nodes=surface_nodes_only,
            context_values_included=context_values_included,
            add_normals=include_surface_normals,
            VP_all_inputs=self.VP_all_inputs,
        )

        val_set = Dataset(
            root=self.val_path,
            n_voxels=self.n_voxels,
            bbox_scaler=bbox_scaler,
            velo_scaler=velo_scaler,
            pressure_scaler=pressure_scaler,
            fparams_scaler=fluid_prop_scaler,
            sample_qty=sample_qty,
            non_dimensionalize=self.non_dimensionalize,
            rotate=random_rotation,
            surface_nodes=surface_nodes_only,
            context_values_included=context_values_included,
            add_normals=include_surface_normals,
            VP_all_inputs=self.VP_all_inputs,
        )

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            pin_memory=False,
            collate_fn=custom_collate_fn,
            num_workers=6,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            pin_memory=False,
            collate_fn=custom_collate_fn,
            num_workers=6,
            shuffle=True,
        )

        #####################################
        #            Callbacks              #
        #####################################
        patience = 20 if self.overfit else self.early_stopping_patience
        early_stopping = EarlyStopping(
            monitor="val_loss", min_delta=1e-5, patience=patience, mode="min"
        )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        # checkpoint_monitor = ModelCheckpoint(mode='min',monitor='val_loss')
        tb_logger = pl_loggers.TensorBoardLogger(
            self.current_exp_folder, log_graph=True
        )

        #####################################
        #         Model Definition          #
        #####################################

        channel_in = 4  # wall, inlet, outlet, fluid
        channel_in += (
            len(context_values_included) + 1
        )  # [Re, Eps, rho, Visc, D] or [Re, Eps] + scale

        if include_surface_normals:
            channel_in += 3  # nx,ny,nz

        if surface_nodes_only:
            channel_in -= 1  # removing nodes in fluid zone

        if not self.non_dimensionalize:
            if self.VP_all_inputs:
                channel_in += 2  # inlet V, outlet P Provided to all nodes
            else:
                channel_in += 4  # vx, vy, vz, p - masked at certain nodes

        pl_model = PLMinkEngUNet(
            multiple_branches=self.multiple_branches,
            channel_in=channel_in,
            first_conv_channel=inlet_outlet_conv_channel,
            n_inlet_layers=n_inlet_outlet_layers,
            encoder_blocks=encoder_blocks,
            encoder_channel=encoder_channel,
            decoder_blocks=decoder_blocks,
            decoder_channel=decoder_channel,
            n_outlet_layers=n_inlet_outlet_layers,
            output_layer_channel=inlet_outlet_conv_channel,
            nout=4,
            non_linearity=non_linearity,
            lr=lr,
            min_lr=lr / 16.0,
            train_loss=train_loss,
            val_loss=val_loss,
            overfit=self.overfit,
            reduction=loss_reduction,
            lr_patience=self.lr_scheduler_patience,
        )

        #####################################
        #               Trainer             #
        #####################################
        max_epochs = 100 if self.overfit else self.max_epochs

        trainer = pl.Trainer(
            logger=tb_logger,
            max_epochs=max_epochs,
            detect_anomaly=True,
            accelerator="gpu",
            callbacks=[early_stopping, lr_monitor],
            enable_checkpointing=False,
        )
        # PyTorchLightningPruningCallback(trial,monitor='val_loss')
        trainer.logger.log_hyperparams(hyperparameters)

        print(
            "n_voxels: ",
            self.n_voxels,
            "sample_qty: ",
            sample_qty,
            " batch_size: ",
            batch_size,
            " n_levels: ",
            n_levels,
            " inlet_outlet_conv_channel: ",
            inlet_outlet_conv_channel,
            " n_inlet_outlet_layers: ",
            n_inlet_outlet_layers,
            " n_blocks: ",
            n_blocks[block_index],
            " encoder_channel: ",
            encoder_channel,
            " decoder_channel: ",
            decoder_channel,
            " lr: ",
            lr,
            " loss fn: ",
            train_loss,
        )

        trainer.fit(pl_model, train_loader, val_loader)

        return trainer.callback_metrics["val_loss"].item()
