import MinkowskiEngine as ME
import pytorch_lightning as PL
import torch
from neural.models.minkeng.custom_loss import *
from MinkEngUNet import MinkEngUNet
from MinkEngUNetMultiDecoderBranch import MinkEngUNetMultiDecoderBranch
from torchmetrics import WeightedMeanAbsolutePercentageError

loss_fn_map = {
    "MaskedMSE": MaskedMSE,
    "MSE": torch.nn.MSELoss,
    "MAE": torch.nn.L1Loss,
}


class PLMinkEngUNet(PL.LightningModule):
    def __init__(
        self,
        channel_in,
        first_conv_channel,
        n_inlet_layers,
        encoder_blocks,
        decoder_blocks,
        encoder_channel,
        decoder_channel,
        n_outlet_layers,
        output_layer_channel,
        nout,
        lr=0.0001,
        min_lr=None,
        train_loss="MaskedMSE",
        val_loss="MaskedWAPE",
        reduction="mean",
        overfit=False,
        lr_patience=2,
        multiple_branches=False,
        non_linearity="ReLU",
    ):
        super().__init__()
        self.save_hyperparameters()  # logger=False
        self.lr = lr
        self.min_lr = min_lr
        self.lr_patience = lr_patience
        self.val_loss = (
            MaskedWAPE()
            if val_loss == "MaskedWAPE"
            else WeightedMeanAbsolutePercentageError()
        )
        self.train_loss = loss_fn_map[train_loss](reduction=reduction)
        self.overfit = overfit
        self.multiple_branches = multiple_branches
        if multiple_branches:
            self.net = MinkEngUNetMultiDecoderBranch(
                channel_in=channel_in,
                first_conv_channel=first_conv_channel,
                n_inlet_layers=n_inlet_layers,
                encoder_blocks=encoder_blocks,
                encoder_channel=encoder_channel,
                decoder_blocks=decoder_blocks,
                decoder_channel=decoder_channel,
                n_outlet_layers=n_outlet_layers,
                output_layer_channel=output_layer_channel,
                nout=nout,
                non_linearity=non_linearity,
            )
        else:
            self.net = MinkEngUNet(
                channel_in=channel_in,
                first_conv_channel=first_conv_channel,
                n_inlet_layers=n_inlet_layers,
                encoder_blocks=encoder_blocks,
                encoder_channel=encoder_channel,
                decoder_blocks=decoder_blocks,
                decoder_channel=decoder_channel,
                n_outlet_layers=n_outlet_layers,
                output_layer_channel=output_layer_channel,
                nout=nout,
                non_linearity=non_linearity,
            )
        # self.example_input_array = self.create_example_input()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        x_coords, x_feats, y_feats = batch
        sparse_input = ME.SparseTensor(
            features=x_feats,
            coordinates=x_coords,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            device=self.device,
        )
        sparse_output = ME.SparseTensor(
            features=y_feats,
            coordinates=x_coords,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            device=self.device,
        )
        out = self.forward(sparse_input)
        loss = self.train_loss(out.F, sparse_output.F)

        if torch.isnan(loss):
            return None
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x_coords, x_feats, y_feats = batch
        sparse_input = ME.SparseTensor(
            features=x_feats,
            coordinates=x_coords,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            device=self.device,
        )
        sparse_output = ME.SparseTensor(
            features=y_feats,
            coordinates=x_coords,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            device=self.device,
        )

        out = self.forward(sparse_input)
        loss = self.val_loss(out.F, sparse_output.F)
        if torch.isnan(loss):
            return None

        self.log("val_loss", loss, on_epoch=True, logger=True, prog_bar=True)
        self.log("hp_metric", loss, on_epoch=True, logger=True)
        return loss


    def configure_optimizers(self):
        patience = 10 if self.overfit else self.lr_patience
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            min_lr=self.min_lr,
            mode="min",
            factor=0.5,
            patience=patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
