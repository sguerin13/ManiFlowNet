import torch.nn as nn
import torch
from neural.models.convpoint.ConvPoint.convpoint.nn import PtConv


class ConvPointUNetMultiDecoderBranch(torch.nn.Module):
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
        non_linearity="ReLU",
    ):
        """
        - n_pts: number of points at each layer of the network
        - n_centers: number of points to include in the convolution
        """

        super(ConvPointUNetMultiDecoderBranch, self).__init__()
        self.non_linearity = non_linearity
        self.encoder_blocks = encoder_blocks
        self.encoder_channel = encoder_channel
        self.decoder_blocks = decoder_blocks
        self.decoder_channel = decoder_channel
        self.n_centers = n_centers
        self.channel_in = channel_in
        self.first_conv_channel = first_conv_channel
        self.n_inlet_layers = n_inlet_layers
        self.n_outlet_layers = n_outlet_layers
        self.output_layer_channel = output_layer_channel
        self.nout = nout
        self.pts_in = n_pts[0]  # inlet and outlet level of points
        self.n_pts_per_level_enc = n_pts[:]  # of points at the downsampling levels
        self.n_pts_per_level_dec = list(
            reversed(n_pts[:-1])
        )  # of points at the upsampling levels
        self.n_pts_per_level_dec.append(sample_qty)
        self.stages = len(self.encoder_blocks)

        # inlet
        self.create_inlet_layers()

        # encoder
        self.create_downsampling_layers()
        self.create_encoder_layers()

        # decoder
        self.create_upsampling_layers()
        self.create_decoder_layers()

        # header
        self.header = nn.ModuleDict(
            {
                "X": self.make_predict_module(self.decoder_channel[-1], 1),
                "Y": self.make_predict_module(self.decoder_channel[-1], 1),
                "Z": self.make_predict_module(self.decoder_channel[-1], 1),
                "P": self.make_predict_module(self.decoder_channel[-1], 1),
            }
        )

        # weight initialization, pt conv weights are initialized in the PtConv class
        self._init_weights()

    def create_downsampling_layers(self):
        """
        The number of points are specified in the forward pass
        """
        self.downsample = nn.ModuleList([])
        for i in range(self.stages):
            if i == 0:
                self.downsample.extend(
                    [
                        ConvPointRelu(
                            in_features=self.first_conv_channel,
                            out_features=self.encoder_channel[i],
                            n_centers=self.n_centers,
                            dim=3,
                            nonlinearity=self.non_linearity,
                        )
                    ]
                )

            else:
                self.downsample.extend(
                    [
                        ConvPointRelu(
                            in_features=self.encoder_channel[i - 1],
                            out_features=self.encoder_channel[i],
                            n_centers=self.n_centers,
                            dim=3,
                            nonlinearity=self.non_linearity,
                        )
                    ]
                )

    def create_encoder_layers(self):
        self.encoder = nn.ModuleList([])
        for i in range(self.stages):
            self.encoder.extend(
                [
                    ConvPointResidualBlocks(
                        in_features=self.encoder_channel[i],
                        out_features=self.encoder_channel[i],
                        n_centers=self.n_centers,
                        n_blocks=self.encoder_blocks[i],
                        nonlinearity=self.non_linearity,
                    )
                ]
            )

    def create_upsampling_layers(self):
        self.upsample = nn.ModuleDict(
            {
                "X": nn.ModuleList([]),
                "Y": nn.ModuleList([]),
                "Z": nn.ModuleList([]),
                "P": nn.ModuleList([]),
            }
        )

        for value in self.upsample:
            for i in range(self.stages):
                if i == 0:
                    self.upsample[value].extend(
                        [
                            ConvPointRelu(
                                in_features=self.encoder_channel[-1],
                                n_centers=self.n_centers,
                                out_features=self.decoder_channel[0],
                                dim=3,
                                nonlinearity=self.non_linearity,
                            )
                        ]
                    )
                else:
                    self.upsample[value].extend(
                        [
                            ConvPointRelu(
                                in_features=self.decoder_channel[i - 1],
                                out_features=self.decoder_channel[i],
                                n_centers=self.n_centers,
                                dim=3,
                                nonlinearity=self.non_linearity,
                            )
                        ]
                    )

    def create_decoder_layers(self):
        concat_in_channel = []
        for i in range(self.stages):
            if i < (self.stages - 1):
                concat_in_channel.extend(
                    [self.decoder_channel[i] + self.encoder_channel[-i - 2]]
                )
                # print("concat channel in", i, " ", self.decoder_channel[i] + self.encoder_channel[-i-2])
            else:
                concat_in_channel.extend(
                    [self.decoder_channel[i] + self.first_conv_channel]
                )
                # print("concat channel in", i, " ", self.decoder_channel[i] + self.first_conv_channel)

        self.decoder = nn.ModuleDict(
            {
                "X": nn.ModuleList([]),
                "Y": nn.ModuleList([]),
                "Z": nn.ModuleList([]),
                "P": nn.ModuleList([]),
            }
        )
        for value in self.decoder:
            for i in range(self.stages):
                self.decoder[value].extend(
                    [
                        ConvPointResidualBlocks(
                            in_features=concat_in_channel[i],
                            out_features=self.decoder_channel[i],
                            n_centers=self.n_centers,
                            n_blocks=self.decoder_blocks[i],
                            nonlinearity=self.non_linearity,
                        )
                    ]
                )

    def create_inlet_layers(self):
        module_list = nn.ModuleList()
        for i in range(self.n_inlet_layers):
            if i == 0:
                module_list.append(
                    ConvPointRelu(
                        dim=3,
                        in_features=self.channel_in,
                        n_centers=self.n_centers,
                        out_features=self.first_conv_channel,
                        nonlinearity=self.non_linearity,
                    )
                )

            else:
                module_list.append(
                    ConvPointRelu(
                        dim=3,
                        in_features=self.first_conv_channel,
                        n_centers=self.n_centers,
                        out_features=self.first_conv_channel,
                        nonlinearity=self.non_linearity,
                    )
                )

        self.inlet_conv = module_list

    def make_predict_module(self, in_channels, out_channels=1):
        """
        Linear Layer Input:
            - (batch_size,*, H_in) where H_in is the in_features

        Output:
            - (batch_size,*, H_out) where H_out is the outfeatures

        By mapping the shape of the input to (batch_size,n_pts,in_feats) it will process the inputs in a point-wise fashion
        """
        module_list = []

        if self.n_outlet_layers > 1:  # multi layer
            for i in range(self.n_outlet_layers):
                if i == 0:  # first layer
                    module_list.append(
                        nn.Linear(
                            in_features=in_channels,
                            out_features=self.output_layer_channel,
                        )
                    )
                    if self.non_linearity == "ELU":
                        module_list.append(nn.ELU(inplace=True))
                    else:
                        module_list.append(nn.ReLU(inplace=True))

                else:
                    module_list.append(
                        nn.Linear(
                            in_features=self.output_layer_channel,
                            out_features=self.output_layer_channel,
                        )
                    )
                    if self.non_linearity == "ELU":
                        module_list.append(nn.ELU(inplace=True))
                    else:
                        module_list.append(nn.ReLU(inplace=True))
            # output layer
            module_list.append(
                nn.Linear(
                    in_features=self.output_layer_channel, out_features=out_channels
                )
            )

            return torch.nn.Sequential(*module_list)

        else:  # single layer
            activation = (
                nn.ELU(inplace=True)
                if self.non_linearity == "ELU"
                else nn.ReLU(inplace=True)
            )
            return torch.nn.Sequential(
                nn.Linear(
                    in_features=in_channels, out_features=self.output_layer_channel
                ),
                activation,
                nn.Linear(
                    in_features=self.output_layer_channel, out_features=out_channels
                ),
            )

    def forward(self, feats, pts):
        """
        pts: x,y,z coordinates
        feats: input features


        PtConv Forward params: (features, pts, kernel_n_points, n_points_out)

        """
        pts_list = []
        # pass through the inlet
        if self.n_inlet_layers == 1:
            # in_n_pts = sample_qty, out_n_pts=sample_qty        # pts, feats, kernel size = neighborhood size
            # print("inlet_0_pts shape: ",pts.shape)
            # print("inlet_features_0 shape: ",feats.shape )
            inlet_layer_feats, inlet_layer_pts = self.inlet_conv[0](
                in_features=feats,
                in_pts=pts,
                n_kernel_pts=self.n_centers,
                output_points=None,
            )

        else:
            for i in range(self.n_inlet_layers):
                if i == 0:
                    # print("inlet_0_pts shape: ",pts.shape)
                    # print("inlet_features_0 shape: ",feats.shape )
                    inlet_layer_feats, inlet_layer_pts = self.inlet_conv[i](
                        in_features=feats,
                        in_pts=pts,
                        n_kernel_pts=self.n_centers,
                        output_points=None,
                    )
                else:
                    # print("inlet_{}_pts shape: ".format(i),inlet_layer_pts.shape)
                    # print("inlet_{}_features_0 shape: ".format(i),inlet_layer_feats.shape)
                    inlet_layer_feats, inlet_layer_pts = self.inlet_conv[i](
                        in_features=inlet_layer_feats,
                        in_pts=inlet_layer_pts,
                        n_kernel_pts=self.n_centers,
                        output_points=None,
                    )
        # print("Inlet Layer Points", inlet_layer_pts.shape[1])
        pts_list.append(inlet_layer_pts)

        # encoder
        stages = len(self.encoder_blocks)
        encoder_layers = []
        for i in range(stages):
            if i == 0:
                # print("down_sample_{} input shape pts: ".format(i),inlet_layer_pts.shape)
                # print("down_sample_{} input shape feats: ".format(i),inlet_layer_feats.shape)
                ds_i_feats, ds_i_pts = self.downsample[i](
                    in_features=inlet_layer_feats,
                    in_pts=inlet_layer_pts,
                    n_kernel_pts=self.n_centers,
                    output_points=self.n_pts_per_level_enc[
                        i
                    ], 
                )
                pts_list.append(ds_i_pts)
                # print("down_sample_{} output shape pts: ".format(i),ds_i_pts.shape)
                # print("down_sample_{} output shape feats: ".format(i),ds_i_feats.shape)

                enc_i_feats, enc_i_pts = self.encoder[i](
                    in_features=ds_i_feats,
                    in_pts=ds_i_pts,
                    n_kernel_pts=self.n_centers,
                )

                # print("encoder_{} output shape pts: ".format(i),enc_i_pts.shape)
                # print("encoder_{} output shape feats: ".format(i),enc_i_feats.shape)

                encoder_layers.append((enc_i_feats, enc_i_pts))
                # print("downsample",i,'inlet pts',inlet_layer_pts.shape[1], " outlet pts", ds_i_pts.shape[1])
            else:
                feats, pts = encoder_layers[i - 1]
                n_centers = self.n_centers if feats.shape[1] > self.n_centers else feats.shape[1]

                # print("down_sample_{} input shape pts: ".format(i),pts.shape)
                # print("down_sample_{} input shape feats: ".format(i),feats.shape)
                ds_i_feats, ds_i_pts = self.downsample[i](
                    in_features=feats,
                    in_pts=pts,
                    n_kernel_pts=n_centers,
                    output_points=self.n_pts_per_level_enc[i],
                )
                pts_list.append(ds_i_pts)
                # print("down_sample_{} output shape pts: ".format(i),ds_i_pts.shape)
                # print("down_sample_{} output shape feats: ".format(i),ds_i_feats.shape)
                n_centers = self.n_centers if ds_i_feats.shape[1] > self.n_centers else ds_i_feats.shape[1]
                # print(n_centers)
                enc_i_feats, enc_i_pts = self.encoder[i](
                    in_features=ds_i_feats,
                    in_pts=ds_i_pts,
                    n_kernel_pts=n_centers,
                )
                # print("encoder_{} output shape pts: ".format(i),enc_i_pts.shape)
                # print("encoder_{} output shape feats: ".format(i),enc_i_feats.shape)

                # print("downsample",i,'inlet pts',pts.shape[1], " outlet pts", ds_i_pts.shape[1])
                encoder_layers.append((enc_i_feats, enc_i_pts))


        # decoder
        pts_for_the_way_up = list(reversed(pts_list))
        decoder_layers = {"X": [], "Y": [], "Z": [], "P": []}
        for i in range(stages):
            upsample = {}
            if i == 0:
                enc_i_feats, enc_i_pts = encoder_layers[-1]
                n_centers = self.n_centers if enc_i_feats.shape[1] > self.n_centers else enc_i_feats.shape[1]
                # print("upsample_{} input shape pts: ".format(i),enc_i_pts.shape)
                # print("upsample_{} input shape feats: ".format(i),enc_i_feats.shape)
                for value in self.upsample:
                    upsample_i_feats, upsample_i_pts = self.upsample[value][i](
                        in_features=enc_i_feats,
                        in_pts=enc_i_pts,
                        n_kernel_pts=n_centers,
                        output_points=pts_for_the_way_up[i + 1],
                    )
                    upsample[value] = upsample_i_feats

                # print("upsample_{} output shape pts: ".format(i),upsample_i_pts.shape)
                # print("upsample_{} output shape feats: ".format(i),upsample_i_feats.shape)
                # print("upsample",i,"inlet_pts", enc_i_pts.shape[1], " outlet pts", upsample_i_pts.shape[1])

            else:
                dec_feats, dec_pts = decoder_layers[value][i - 1]
                n_centers = self.n_centers if dec_feats.shape[1] > self.n_centers else dec_feats.shape[1]

                # print("upsample_{} input shape pts: ".format(i),dec_pts.shape)
                # print("upsample_{} input shape feats: ".format(i),dec_feats.shape)
                for value in self.upsample:
                    upsample_i_feats, upsample_i_pts = self.upsample[value][i](
                        in_features=dec_feats,
                        in_pts=dec_pts,
                        n_kernel_pts=n_centers,
                        output_points=pts_for_the_way_up[i + 1],
                    )
                    upsample[value] = upsample_i_feats


                # print("upsample_{} output shape pts: ".format(i),upsample_i_pts.shape)
                # print("upsample_{} output shape feats: ".format(i),upsample_i_feats.shape)
                # print("upsample",i,"inlet_pts", dec_pts.shape[1], " outlet pts", upsample_i_pts.shape[1])

            # skip connections
            cat_upsample = {}
            if i < (self.stages - 1):
                for value in self.upsample:
                    # features
                    cat_upsample[value] = torch.cat(
                        (upsample[value], encoder_layers[-i - 2][0]), dim=2
                    )
                # print("upsample_{} concat shape feats: ".format(i),cat_upsample_feats.shape)
                # print("\tcat upsample: ", upsample.F.shape, encoder_layers[-i-2].F.shape, cat_upsample.F.shape)
            else:
                # print(upsample,encoder_layers[0])
                for value in self.upsample:
                    cat_upsample[value] = torch.cat(
                        (upsample[value], inlet_layer_feats), dim=2
                    )
                # print("upsample_{} concat shape feats: ".format(i),cat_upsample_feats.shape)
                # print("\tcat upsample: ", cat_upsample.F.shape)


            # print("decoder_{} input shape pts: ".format(i),upsample_i_pts.shape)
            # print("decoder_{} input shape feats: ".format(i),cat_upsample_feats.shape)

            n_centers = self.n_centers if cat_upsample[value].shape[1] > self.n_centers else cat_upsample[value].shape[1]
            
            for value in decoder_layers:
                decoder_feats, decoder_pts = self.decoder[value][i](
                    in_features=cat_upsample[value],
                    in_pts=upsample_i_pts,
                    n_kernel_pts=n_centers)

                decoder_layers[value].append((decoder_feats,decoder_pts))

        # header
        val_out = {}
        for value in decoder_layers:
            val_out[value] = self.header[value](decoder_layers[value][-1][0])
        
        # print("decoder_output_shape: ", decoder_output.shape)
        return torch.cat((val_out["X"], val_out["Y"], val_out["Z"], val_out["P"]),dim=2)

    def _init_weights(self):
        for m in self.modules():
            nl = "relu" if self.non_linearity == "ReLU" else "elu"
            # if isinstance(m, PtConv):
            #     torch.nn.init.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity=nl)

            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity=nl)


class ConvPointRelu(nn.Module):
    def __init__(
        self, in_features, out_features, n_centers, dim=3, nonlinearity="ReLU"
    ):
        super(ConvPointRelu, self).__init__()
        self.conv = PtConv(in_features, out_features, n_centers, dim, use_bias=True)
        self.activation = (
            nn.ELU(inplace=True) if nonlinearity == "ELU" else nn.ReLU(inplace=True)
        )

    def forward(self, in_features, in_pts, n_kernel_pts, output_points=None):
        # if output_n_pts === none then it is a stride 1 conv
        conv_feats_out, pts_out = self.conv(
            input=in_features,
            points=in_pts,
            K=n_kernel_pts,
            next_pts=output_points,
            normalize=False,
        )
        feats_out = self.activation(conv_feats_out)
        return feats_out, pts_out


class ConvPointResBlock(nn.Module):
    # Traditional ResNet Block
    """
    --  ----------- Identity -----------  +  ---- Act ->
      \                                  /
       --- Weight --- Act --- Weight ---

    """

    def __init__(
        self, in_features, out_features, n_centers, dim=3, nonlinearity="ReLU"
    ):
        super(ConvPointResBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_1 = (
            nn.ELU(inplace=True) if nonlinearity == "ELU" else nn.ReLU(inplace=True)
        )
        self.conv_1 = PtConv(in_features, out_features, n_centers, dim, use_bias=True)
        self.act_2 = (
            nn.ELU(inplace=True) if nonlinearity == "ELU" else nn.ReLU(inplace=True)
        )
        self.conv_2 = PtConv(out_features, out_features, n_centers, dim, use_bias=True)

        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features, bias=True)

    def forward(self, in_features, in_pts, n_kernel_pts):
        residual = in_features
        feats_1, pts_1 = self.conv_1(
            input=in_features,
            points=in_pts,
            K=n_kernel_pts,
            next_pts=None,
            normalize=False,
        )
        act_feats_1 = self.act_1(feats_1)
        conv_feats_2, pts_2 = self.conv_2(
            input=act_feats_1,
            points=pts_1,
            K=n_kernel_pts,
            next_pts=None,
            normalize=False,
        )

        if self.in_features != self.out_features:
            residual = self.projection(residual)

        resid_feats = conv_feats_2 + residual
        act_resid_feats = self.act_2(resid_feats)
        return act_resid_feats, pts_2


class ConvPointResidualBlocks(nn.Module):
    def __init__(
        self, in_features, out_features, n_centers, n_blocks, dim=3, nonlinearity="ReLU"
    ):
        super(ConvPointResidualBlocks, self).__init__()
        self.n_blocks = n_blocks

        channels = [in_features] + [out_features] * n_blocks
        self.res_blocks = nn.ModuleList(
            [
                ConvPointResBlock(
                    in_features=channels[i],
                    out_features=channels[i + 1],
                    n_centers=n_centers,
                    dim=3,
                    nonlinearity=nonlinearity,
                )
                for i in range(n_blocks)
            ]
        )

    def forward(self, in_features, in_pts, n_kernel_pts):
        for i in range(self.n_blocks):
            if i == 0:
                x_feats, x_pts = self.res_blocks[i](
                    in_features=in_features, in_pts=in_pts, n_kernel_pts=n_kernel_pts
                )
            else:
                x_feats, x_pts = self.res_blocks[i](
                    in_features=x_feats, in_pts=x_pts, n_kernel_pts=n_kernel_pts
                )

        return x_feats, x_pts
