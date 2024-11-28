import torch
import torch.nn as nn


class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, decoder_layer_sizes,latent_size=256,
                 conditional=False, condition_size=2048):
        super().__init__()

        if conditional:
            assert condition_size > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list
        self.latent_size = encoder_layer_sizes[-1]  #保持encoder和decoder维度一致

        self.encoder = Encoder(
            encoder_layer_sizes, self.latent_size, conditional, condition_size)
        self.decoder = Decoder(
            decoder_layer_sizes, self.latent_size, conditional, condition_size)

    def forward(self, x, c=None):
        z = self.encoder(x, c)
        x_out = self.decoder(z, c)

        return x_out


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, condition_size):

        super().__init__()
        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += condition_size

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            '''new'''
            # self.MLP.add_module(name="BN{:d}".format(i), module= nn.BatchNorm1d(out_size))
            # self.MLP.add_module(name="D{:d}".format(i), module=nn.Dropout(p=0.1))  # 添加丢弃层
    def forward(self, x, c=None):
        if self.conditional:
            x = torch.cat((x, c), dim=-1)  # [B, 1024+61]
        # print('x size before MLP {}'.format(x.size()))
        x = self.MLP(x)
        return x


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, condition_size):
        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + condition_size
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
                '''new'''
                # self.MLP.add_module(name="BN{:d}".format(i), module= nn.BatchNorm1d(out_size))
                # self.MLP.add_module(name="D{:d}".format(i), module=nn.Dropout(p=0.1))  # 添加丢弃层

    def forward(self, z, c=None):
        if self.conditional:
            z = torch.cat((z, c), dim=-1)
            # print('z size {}'.format(z.size()))
        x = self.MLP(z)

        return x


if __name__ == '__main__':
    model = VAE(
        encoder_layer_sizes=[1024, 512, 256],
        latent_size=256,
        decoder_layer_sizes=[256, 256, 61],
        condition_size=1024)

    x = torch.randn((5, 1024))
    condition = torch.randn((5, 1024))
    print(x.size(), condition.size())
    print('params {}M'.format(sum(p.numel()
          for p in model.parameters()) / 1000000.0))
    # model.train()
    # recon, _, _, _ = model(x, condition)
    model.eval()
    recon = model(x, condition)
    print('recon size {}'.format(recon.size()))
