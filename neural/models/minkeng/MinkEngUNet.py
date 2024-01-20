import torch.nn as nn
import torch
import MinkowskiEngine as ME


class MinkEngUNet(torch.nn.Module):
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
        non_linearity="ReLU",
    ):
        super(MinkEngUNet, self).__init__()
        self.non_linearity = non_linearity
        self.encoder_blocks = encoder_blocks
        self.encoder_channel = encoder_channel
        self.decoder_blocks = decoder_blocks
        self.decoder_channel = decoder_channel
        self.channel_in = channel_in
        self.first_conv_channel = first_conv_channel
        self.n_inlet_layers = n_inlet_layers
        self.n_outlet_layers = n_outlet_layers
        self.output_layer_channel = output_layer_channel
        self.nout = nout
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
        self.header = self.make_predict_module(self.decoder_channel[-1], nout)
        self._init_weights

    def create_downsampling_layers(self):
        self.downsample = nn.ModuleList([])
        for i in range(self.stages):
            if i == 0:
                self.downsample.extend(
                    [
                        MinkEngConvRelu(
                            dimension=3,
                            in_channels=self.first_conv_channel,
                            out_channels=self.encoder_channel[i],
                            kernel_size=3,
                            stride=2,
                            nonlinearity=self.non_linearity,
                        )
                    ]
                )

            else:
                self.downsample.extend(
                    [
                        MinkEngConvRelu(
                            dimension=3,
                            in_channels=self.encoder_channel[i - 1],
                            out_channels=self.encoder_channel[i],
                            kernel_size=3,
                            stride=2,
                            nonlinearity=self.non_linearity,
                        )
                    ]
                )

    def create_encoder_layers(self):
        self.encoder = nn.ModuleList([])
        for i in range(self.stages):
            self.encoder.extend(
                [
                    MinkEngResidualBlocks(
                        in_channels=self.encoder_channel[i],
                        out_channels=self.encoder_channel[i],
                        n_blocks=self.encoder_blocks[i],
                        nonlinearity=self.non_linearity,
                    )
                ]
            )

    def create_upsampling_layers(self):
        self.upsample = nn.ModuleList([])
        for i in range(self.stages):
            if i == 0:
                self.upsample.extend(
                    [
                        MinkEngDeConvRelu(
                            in_channels=self.encoder_channel[-1],
                            out_channels=self.decoder_channel[0],
                            nonlinearity=self.non_linearity,
                        )
                    ]
                )
            else:
                self.upsample.extend(
                    [
                        MinkEngDeConvRelu(
                            in_channels=self.decoder_channel[i - 1],
                            out_channels=self.decoder_channel[i],
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

        self.decoder = nn.ModuleList([])
        for i in range(self.stages):
            self.decoder.extend(
                [
                    MinkEngResidualBlocks(
                        in_channels=concat_in_channel[i],
                        out_channels=self.decoder_channel[i],
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
                    MinkEngConvRelu(
                        dimension=3,
                        in_channels=self.channel_in,
                        out_channels=self.first_conv_channel,
                        kernel_size=3,
                        stride=1,
                        nonlinearity=self.non_linearity,
                    )
                )
            else:
                module_list.append(
                    MinkEngConvRelu(
                        dimension=3,
                        in_channels=self.first_conv_channel,
                        out_channels=self.first_conv_channel,
                        kernel_size=3,
                        stride=1,
                        nonlinearity=self.non_linearity,
                    )
                )

        self.inlet_conv = module_list

    def make_predict_module(self, decoder_output_channels, out_channels=4):
        module_list = []

        if self.n_outlet_layers > 1:  # multi layer
            for i in range(self.n_outlet_layers):
                if i == 0:  # first layer
                    module_list.append(
                        MinkEngConvRelu(
                            dimension=3,
                            in_channels=decoder_output_channels,
                            out_channels=self.output_layer_channel,
                            kernel_size=1,
                            stride=1,
                            nonlinearity=self.non_linearity,
                        )
                    )

                else:
                    module_list.append(
                        MinkEngConvRelu(
                            dimension=3,
                            in_channels=self.output_layer_channel,
                            out_channels=self.output_layer_channel,
                            kernel_size=1,
                            stride=1,
                            nonlinearity=self.non_linearity,
                        )
                    )

            # output layer
            module_list.append(
                ME.MinkowskiConvolution(
                    in_channels=self.output_layer_channel,
                    out_channels=out_channels,
                    stride=1,
                    dilation=1,
                    kernel_size=1,
                    dimension=3,
                    bias=True,
                )
            )

            return torch.nn.Sequential(*module_list)

        else:  # single layer
            return torch.nn.Sequential(
                MinkEngConvRelu(
                    dimension=3,
                    in_channels=decoder_output_channels,
                    out_channels=self.output_layer_channel,
                    kernel_size=1,
                    stride=1,
                    nonlinearity=self.non_linearity,
                ),
                ME.MinkowskiConvolution(
                    in_channels=self.output_layer_channel,
                    out_channels=out_channels,
                    stride=1,
                    dilation=1,
                    kernel_size=1,
                    dimension=3,
                    bias=True,
                ),
            )

    def forward(self, x):
        # pass through the inlet
        if self.n_inlet_layers == 1:
            inlet_layer = self.inlet_conv[0](x)

        else:
            for i in range(self.n_inlet_layers):
                if i == 0:
                    inlet_layer = self.inlet_conv[i](x)
                else:
                    inlet_layer = self.inlet_conv[i](inlet_layer)

        # encoder
        stages = len(self.encoder_blocks)
        encoder_layers = []
        for i in range(stages):
            if i == 0:
                ds = self.downsample[i](inlet_layer)
                encoder_layers.extend([self.encoder[i](ds)])
            else:
                ds = self.downsample[i](encoder_layers[i - 1])
                encoder_layers.extend([self.encoder[i](ds)])

        # decoder
        decoder_layers = []
        # print(inlet_layer.shape,[i.shape for i in encoder_layers])
        for i in range(stages):
            if i == 0:
                upsample = self.upsample[i](encoder_layers[-1])
            else:
                upsample = self.upsample[i](decoder_layers[i - 1])

            # skip connections
            if i < (self.stages - 1):
                cat_upsample = ME.cat((upsample, encoder_layers[-i - 2]))
                # print("\tcat upsample: ", upsample.F.shape, encoder_layers[-i-2].F.shape, cat_upsample.F.shape)
            else:
                # print(upsample,encoder_layers[0])
                cat_upsample = ME.cat((upsample, inlet_layer))
                # print("\tcat upsample: ", cat_upsample.F.shape)

            decoder_layers.extend([self.decoder[i](cat_upsample)])

        # header
        y_t = self.header(decoder_layers[-1])
        return y_t

    def _init_weights(self):
        for m in self.modules():
            nl = "relu" if self.non_linearity == "ReLU" else "elu"
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity=nl)

            if isinstance(m, ME.MinkowskiLinear):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity=nl)


class MinkEngConvRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dimension=3,
        stride=1,
        dilation=1,
        kernel_size=3,
        nonlinearity="ReLU",
    ):
        super(MinkEngConvRelu, self).__init__()
        self.conv = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation,
            kernel_size=kernel_size,
            dimension=dimension,
            bias=True,
        )
        self.activation = (
            ME.MinkowskiELU(inplace=True)
            if nonlinearity == "ELU"
            else ME.MinkowskiReLU(inplace=True)
        )

    def forward(self, x):
        return self.activation(self.conv(x))


class MinkEngDeConvRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dimension=3,
        dilation=1,
        kernel_size=3,
        nonlinearity="ReLU",
    ):
        super(MinkEngDeConvRelu, self).__init__()
        self.conv = ME.MinkowskiConvolutionTranspose(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            dilation=dilation,
            kernel_size=kernel_size,
            dimension=dimension,
            bias=True,
        )
        self.activation = (
            ME.MinkowskiELU(inplace=True)
            if nonlinearity == "ELU"
            else ME.MinkowskiReLU(inplace=True)
        )

    def forward(self, x):
        return self.activation(self.conv(x))


class MinkEngResBlock(nn.Module):
    # Traditional ResNet Block
    """
    __  __________ Identity __________ + ____ Act
      \                               /
       \___ Weight __ Act ___ Weight /

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        dilation=1,
        dimension=3,
        nonlinearity="ReLU",
    ):
        super(MinkEngResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu1 = (
            ME.MinkowskiELU(inplace=True)
            if nonlinearity == "ELU"
            else ME.MinkowskiReLU(inplace=True)
        )
        self.conv1 = ME.MinkowskiConvolution(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            dimension=dimension,
            bias=True,
        )
        self.relu2 = (
            ME.MinkowskiELU(inplace=True)
            if nonlinearity == "ELU"
            else ME.MinkowskiReLU(inplace=True)
        )
        self.conv2 = ME.MinkowskiConvolution(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            dimension=dimension,
            bias=True,
        )
        if in_channels != out_channels:
            self.projection = ME.MinkowskiLinear(in_channels, out_channels, bias=True)

    def forward(self, x):
        residual = x
        x = self.conv2(self.relu1(self.conv1(x)))

        # print("In channel: ", self.in_channels, ", Out Channel: ", self.out_channels)
        # print("\t\t residual shape", residual.F.shape," output_shape: ", x.shape)
        if self.in_channels != self.out_channels:
            residual = self.projection(residual)
        return self.relu2(residual + x)


class MinkEngResidualBlocks(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_blocks,
        stride=1,
        dilation=1,
        dimension=3,
        nonlinearity="ReLU",
    ):
        super(MinkEngResidualBlocks, self).__init__()
        self.n_blocks = n_blocks

        channels = [in_channels] + [out_channels] * n_blocks
        self.res_blocks = nn.ModuleList(
            [
                MinkEngResBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    dimension=3,
                    stride=stride,
                    dilation=dilation,
                    nonlinearity=nonlinearity,
                )
                for i in range(n_blocks)
            ]
        )

    def forward(self, x):
        # print("\tResidual Blocks Input Shape", x.F.shape)
        for i in range(self.n_blocks):
            # print("\t\tBlock ",i,":")
            x = self.res_blocks[i](x)
        return x
