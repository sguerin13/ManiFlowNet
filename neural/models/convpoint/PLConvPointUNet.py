import pytorch_lightning as PL
import torch
from neural.models.convpoint.ConvPointUNet import ConvPointUNet
from neural.models.convpoint.ConvPointUNetMultiDecoderBranch import ConvPointUNetMultiDecoderBranch
from neural.models.convpoint.custom_loss import *
from torchmetrics import WeightedMeanAbsolutePercentageError

loss_fn_map = {
    "MaskedMSE": MaskedMSE,
    "MSE": torch.nn.MSELoss,
    "MAE": torch.nn.L1Loss,
}


class PLConvPointUNet(PL.LightningModule):
    def __init__(
        self,
        sample_qty,
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
        n_centers,
        n_pts,
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
        self.sample_qty = sample_qty
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
            self.net = ConvPointUNetMultiDecoderBranch(
                sample_qty=sample_qty,
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
                n_centers=n_centers,
                n_pts=n_pts,
                non_linearity=non_linearity,
            )
        else:
            self.net = ConvPointUNet(
                sample_qty=sample_qty,
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
                n_centers=n_centers,
                n_pts=n_pts,
                non_linearity=non_linearity,
            )

    def forward(self, feats, pts):
        return self.net(feats, pts)

    def training_step(self, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        x_pts, x_feats, y_feats = batch
        out = self.forward(x_feats, x_pts)
        loss = self.train_loss(out, y_feats)

        if torch.isnan(loss):
            return None
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        x_pts, x_feats, y_feats = batch
        out = self.forward(x_feats, x_pts)
        loss = self.val_loss(out, y_feats)
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
