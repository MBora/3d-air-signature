import torch
import torch.nn as nn
from torchvision import models


class NewGRU(nn.Module):

    def __init__(
        self,
        num_columns,
        num_windows,
        num_descriptors,
        num_classes,
        dropout_rate,
        hidden_size_1,
        hidden_size_2,
        hidden_size_3,
        hidden_size_4,
        layer1_bidirectional,
        layer2_bidirectional,
    ):
        super(NewGRU, self).__init__()

        self.num_columns = num_columns
        self.num_windows = num_windows
        self.num_descriptors = num_descriptors

        self.layer1 = nn.ModuleList(
            [
                nn.GRU(
                    num_descriptors, hidden_size_1, bidirectional=layer1_bidirectional, dropout=0.5
                )
                for i in range(self.num_columns)
            ]
        )

        self.fc1 = nn.ModuleList(
            [
                nn.Linear(
                    in_features=hidden_size_1 * (int(layer1_bidirectional) + 1),
                    out_features=hidden_size_2,
                )
                for i in range(self.num_columns)
            ]
        )

        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout_rate)

        self.layer2 = nn.GRU(
            hidden_size_2 * self.num_columns,
            hidden_size_3,
            bidirectional=layer2_bidirectional,
            dropout=0.5
        )
        self.fc2 = nn.Linear(
            hidden_size_3 * (int(layer2_bidirectional) + 1), hidden_size_4
        )

        self.flatten = nn.Flatten(start_dim=1, end_dim=2)

        self.fc3 = nn.Linear(hidden_size_4 * num_windows, num_classes)

    def forward(self, x):

        layer1_out = []

        for i in range(self.num_columns):
            t = x[:, :, :, i]
            t, _ = self.layer1[i](t)
            t = self.fc1[i](t)
            layer1_out.append(t)

        concat_out = torch.cat(layer1_out, dim=2)
        x, _ = self.layer2(concat_out)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc3(x)

        return x


class OldGRU(nn.Module):

    def __init__(
        self,
        num_columns,
        num_windows,
        num_descriptors,
        num_classes,
        dropout_rate,
        hidden_size_1,
        hidden_size_2,
        hidden_size_3,
        hidden_size_4,
        layer1_bidirectional,
        layer2_bidirectional,
    ):
        super(OldGRU, self).__init__()

        self.num_columns = num_columns
        self.num_windows = num_windows
        self.num_descriptors = num_descriptors

        self.layer1 = nn.ModuleList(
            [
                nn.GRU(num_columns, hidden_size_1, bidirectional=layer1_bidirectional)
                for _ in range(self.num_descriptors)
            ]
        )

        self.fc1 = nn.ModuleList(
            [
                nn.Linear(
                    hidden_size_1 * (int(layer1_bidirectional) + 1), hidden_size_2
                )
                for _ in range(self.num_descriptors)
            ]
        )

        self.layer2 = nn.GRU(
            hidden_size_2 * self.num_descriptors,
            hidden_size_3,
            bidirectional=layer2_bidirectional,
        )
        self.fc2 = nn.Linear(
            hidden_size_3 * (int(layer2_bidirectional) + 1), hidden_size_4
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        self.fc3 = nn.Linear(hidden_size_4 * num_windows, num_classes)

    def forward(self, x):
        layer1_out = []

        for i in range(self.num_descriptors):
            t = x[:, :, i]
            t, _ = self.layer1[i](t)
            t = self.fc1[i](t)
            layer1_out.append(t)

        concat_out = torch.cat(layer1_out, dim=2)

        x, _ = self.layer2(concat_out)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc3(x)

        return x


class NewLSTM(nn.Module):

    def __init__(
        self,
        num_columns,
        num_windows,
        num_descriptors,
        num_classes,
        dropout_rate,
        hidden_size_1,
        hidden_size_2,
        hidden_size_3,
        hidden_size_4,
        layer1_bidirectional,
        layer2_bidirectional,
    ):
        super(NewLSTM, self).__init__()

        self.num_columns = num_columns
        self.num_windows = num_windows
        self.num_descriptors = num_descriptors

        self.layer1 = nn.ModuleList(
            [
                nn.LSTM(
                    num_descriptors, hidden_size_1, bidirectional=layer1_bidirectional
                )
                for i in range(self.num_columns)
            ]
        )

        self.fc1 = nn.ModuleList(
            [
                nn.Linear(
                    in_features=hidden_size_1 * (int(layer1_bidirectional) + 1),
                    out_features=hidden_size_2,
                )
                for i in range(self.num_columns)
            ]
        )

        self.layer2 = nn.LSTM(
            hidden_size_2 * num_columns,
            hidden_size_3,
            bidirectional=layer2_bidirectional,
        )
        self.fc2 = nn.Linear(
            hidden_size_3 * (int(layer2_bidirectional) + 1), hidden_size_4
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.flatten = nn.Flatten(start_dim=1, end_dim=2)

        self.fc3 = nn.Linear(hidden_size_4 * num_windows, num_classes)

    def forward(self, x):
        layer1_out = []

        for i in range(self.num_columns):
            t = x[:, :, :, i]
            t, _ = self.layer1[i](t)
            t = self.fc1[i](t)
            layer1_out.append(t)

        concat_out = torch.cat(layer1_out, dim=2)

        x, _ = self.layer2(concat_out)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc3(x)

        return x


class OldLSTM(nn.Module):

    def __init__(
        self,
        num_columns,
        num_windows,
        num_descriptors,
        num_classes,
        dropout_rate,
        hidden_size_1,
        hidden_size_2,
        hidden_size_3,
        hidden_size_4,
        layer1_bidirectional,
        layer2_bidirectional,
    ):
        super(OldLSTM, self).__init__()

        self.num_columns = num_columns
        self.num_windows = num_windows
        self.num_descriptors = num_descriptors

        self.layer1 = nn.ModuleList(
            [
                nn.LSTM(num_columns, hidden_size_1, bidirectional=layer1_bidirectional)
                for _ in range(self.num_descriptors)
            ]
        )

        self.fc1 = nn.ModuleList(
            [
                nn.Linear(
                    hidden_size_1 * (int(layer1_bidirectional) + 1), hidden_size_2
                )
                for _ in range(self.num_descriptors)
            ]
        )

        self.layer2 = nn.LSTM(
            hidden_size_2 * num_descriptors,
            hidden_size_3,
            bidirectional=layer2_bidirectional,
        )
        self.fc2 = nn.Linear(
            hidden_size_3 * (int(layer2_bidirectional) + 1), hidden_size_4
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        self.fc3 = nn.Linear(hidden_size_4 * num_windows, num_classes)

    def forward(self, x):
        layer1_out = []

        for i in range(self.num_descriptors):
            t = x[:, :, i]
            t, _ = self.layer1[i](t)
            t = self.fc1[i](t)
            layer1_out.append(t)

        concat_out = torch.cat(layer1_out, dim=2)

        x, _ = self.layer2(concat_out)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc3(x)

        return x


class CNN1D(nn.Module):

    def __init__(self, num_rows, num_columns, num_classes):
        super(CNN1D, self).__init__()

        self.num_columns = num_columns
        self.num_rows = num_rows

        self.conv1d_cnn_1 = nn.Sequential(
            nn.Conv1d(
                self.num_columns, self.num_columns, kernel_size=4, stride=2, padding=1
            ),
            nn.Dropout(0.2),
            nn.ReLU(),
        )

        self.conv1d_cnn_2 = nn.Sequential(
            nn.Conv1d(
                self.num_columns, self.num_columns, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
        )

        self.conv1d_cnn_3 = nn.Sequential(
            nn.Conv1d(
                self.num_columns, self.num_columns, kernel_size=4, stride=2, padding=1
            ),
            nn.Dropout(0.2),
            nn.ReLU(),
        )

        self.conv1d_cnn_4 = nn.Sequential(
            nn.Conv1d(
                self.num_columns, self.num_columns, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
        )

        self.conv1d_cnn_5 = nn.Conv1d(
            self.num_columns, self.num_columns, kernel_size=4, stride=2, padding=1
        )

        self.final_layer = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=2),
            nn.Linear(int(self.num_rows / 32) * self.num_columns, 128),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))

        x = self.conv1d_cnn_1(x)
        x = self.conv1d_cnn_2(x)
        x = self.conv1d_cnn_3(x)
        x = self.conv1d_cnn_4(x)
        x = self.conv1d_cnn_5(x)

        x = self.final_layer(x)

        return x


def VGG16(num_classes):

    num_classes = num_classes
    model = models.vgg16(pretrained=True)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    return model


class SimpleLSTM(nn.Module):

    def __init__(
        self,
        num_rows,
        num_columns,
        num_classes,
        hidden_size_1,
        hidden_size_2,
        bidirectional,
    ):
        super(SimpleLSTM, self).__init__()

        self.layer1 = nn.LSTM(num_columns, hidden_size_1, bidirectional=bidirectional, dropout=0.5)
        self.fc1 = nn.Linear(hidden_size_1 * (int(bidirectional) + 1), hidden_size_2)
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        self.fc2 = nn.Linear(num_rows * hidden_size_2, num_classes)

    def forward(self, x):

        x, _ = self.layer1(x)
        x = self.fc1(x)
        x = self.flatten(x)
        x = self.fc2(x)

        return x


class SimpleGRU(nn.Module):

    def __init__(
        self,
        num_rows,
        num_columns,
        num_classes,
        hidden_size_1,
        hidden_size_2,
        bidirectional,
    ):
        super(SimpleGRU, self).__init__()

        self.layer1 = nn.GRU(num_columns, hidden_size_1, bidirectional=bidirectional, dropout=0.5)
        self.fc1 = nn.Linear(hidden_size_1 * (int(bidirectional) + 1), hidden_size_2)
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        self.fc2 = nn.Linear(num_rows * hidden_size_2, num_classes)

    def forward(self, x):

        x, _ = self.layer1(x)
        x = self.fc1(x)
        x = self.flatten(x)
        x = self.fc2(x)

        return x


class CNN2D(nn.Module):

    def __init__(self, num_rows, num_columns, num_classes):
        super().__init__()

        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_classes = num_classes
        self.factor = int(self.num_rows / 512)

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(30*self.factor,self.num_columns),stride=1)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(8*self.factor,1),stride=1)

        self.norm = nn.LayerNorm([64,self.num_rows-(30*self.factor)+1,1])
        self.maxpool = nn.MaxPool2d(kernel_size=(2,1))
        self.lrelu = nn.LeakyReLU()
        self.flatten = nn.Flatten(start_dim=1,end_dim=2)

        self.fc1 = nn.Linear(in_features=128*int((self.num_rows+2-(38*self.factor))/2),out_features=128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=self.num_classes)

    def forward(self,x):
        x = torch.unsqueeze(x,dim=1)
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = torch.squeeze(x)
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SliTCNN2D(nn.Module):

    def __init__(self, num_rows, num_columns, num_classes):
        super().__init__()

        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_classes = num_classes

        # if not self.num_columns == 6:
        #         raise Exception("Invalid dataloader!")

        self.conv1 = nn.ModuleList([
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(30,3),stride=1)
            for i in range(2)
        ])

        self.conv2 = nn.ModuleList([
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(8,1),stride=1)
            for i in range(2)
        ])

        self.norm = nn.LayerNorm([64,self.num_rows-29,1])
        self.maxpool = nn.MaxPool2d(kernel_size=(2,1))
        self.lrelu = nn.LeakyReLU()
        self.flatten = nn.Flatten(start_dim=1,end_dim=2)

        self.fc1 = nn.ModuleList([
            nn.Linear(in_features=128*int((self.num_rows-36)/2),out_features=128)
            for i in range(2)
        ])

        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=256, out_features=self.num_classes)

    def forward(self,x):
        to_cat = []

        for i in range(2):
            t = x[:,:,3*i:3*(i+1)]
            t = torch.unsqueeze(t,dim=1)
            t = self.conv1[i](t)
            t = self.lrelu(t)
            t = self.norm(t)
            t = self.conv2[i](t)
            t = self.lrelu(t)
            t = self.maxpool(t)
            t = self.flatten(t)
            t = torch.squeeze(t)
            t = self.fc1[i](t)
            t = self.lrelu(t)
            t = self.dropout(t)
            to_cat.append(t)
        out = torch.cat(to_cat,dim=1)
        out = self.fc2(out)
        return out

class SliTCNN2DEncoderDecoder(nn.Module):

    def __init__(self, num_rows, num_columns, num_classes):
        super().__init__()

        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_classes = num_classes

        # if not self.num_columns == 6:
        #         raise Exception("Invalid dataloader!")

        self.conv1 = nn.ModuleList([
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(30,3),stride=1)
            for i in range(2)
        ])

        self.conv2 = nn.ModuleList([
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(8,1),stride=1)
            for i in range(2)
        ])

        self.norm = nn.LayerNorm([64,self.num_rows-29,1])
        self.maxpool = nn.MaxPool2d(kernel_size=(2,1))
        self.lrelu = nn.LeakyReLU()
        self.flatten = nn.Flatten(start_dim=1,end_dim=2)

        self.fc1 = nn.ModuleList([
            nn.Linear(in_features=128*int((self.num_rows-36)/2),out_features=128)
            for i in range(2)
        ])

        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=256, out_features=self.num_classes)

    def forward(self,x):
        to_cat = []

        for i in range(2):
            t = x[:,:,3*i:3*(i+1)]
            t = torch.unsqueeze(t,dim=1)
            t = self.conv1[i](t)
            t = self.lrelu(t)
            t = self.norm(t)
            t = self.conv2[i](t)
            t = self.lrelu(t)
            t = self.maxpool(t)
            t = self.flatten(t)
            t = torch.squeeze(t)
            t = self.fc1[i](t)
            t = self.lrelu(t)
            t = self.dropout(t)
            to_cat.append(t)
        out = torch.cat(to_cat,dim=1)
        out = self.fc2(out)
        return out

# https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
class SliTCNN1StreamEncoderDecoder(nn.Module):
    def __init__(self, num_rows, num_columns, num_classes, svc=False):
        super().__init__()
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_classes = num_classes

        if svc:
            self.conv1 = nn.ModuleList([
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(30, 2), stride=1)
                for i in range(1)
            ])
        else:
            self.conv1 = nn.ModuleList([
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(30, 3), stride=1)
                for i in range(1)
            ])

        self.conv2 = nn.ModuleList([
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(8, 1), stride=1)
            for i in range(1)
        ])

        self.norm = nn.LayerNorm([64, self.num_rows-29, 1])
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 1))
        self.lrelu = nn.LeakyReLU()
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        self.fc1 = nn.ModuleList([
            nn.Linear(in_features=128*int((self.num_rows-36)/2), out_features=128) # earlier 512
            for i in range(1)
        ])
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=self.num_classes)

        # Decoder layers
        self.fc3 = nn.Linear(in_features=512, out_features=128*int((self.num_rows-36)/2))
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, int((self.num_rows-36)/2), 1))
        self.upsample = nn.Upsample(scale_factor=(2, 1))
        self.deconv2 = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(8, 1), stride=1)
            for i in range(1)
        ])
        if svc:
            self.deconv1 = nn.ModuleList([
                nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(30, 2), stride=1)
                for i in range(1)
            ])
        else:
            self.deconv1 = nn.ModuleList([
                nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(30, 3), stride=1)
                for i in range(1)
            ])
        self.fc_mu = nn.Linear(in_features=512, out_features=128)
        self.fc_var = nn.Linear(in_features=512, out_features=128)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.ztoencode = nn.Linear(in_features=128, out_features=512)
   
    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale.clamp(min=-5, max=3))  # Clamp to avoid too large or too small values

        # Check for NaNs or Infs in mean and scale
        if torch.isnan(x_hat).any() or torch.isinf(x_hat).any():
            print("NaNs or Infs found in mean")
        if torch.isnan(scale).any() or torch.isinf(scale).any():
            print("NaNs or Infs found in scale")

        dist = torch.distributions.Normal(x_hat, scale)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))


    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl
         

    def forward(self, x, x_target, mode="train"):
        to_cat = []
        for i in range(1):
            t = x[:,:,0:3]
            # t = x[:,:,3*i:3*(i+1)]
            t = torch.unsqueeze(t, dim=1)
            t = self.conv1[i](t)
            t = self.lrelu(t)
            t = self.norm(t)
            t = self.conv2[i](t)
            t = self.lrelu(t)
            t = self.maxpool(t)
            t = self.flatten(t)
            t = torch.squeeze(t)
            t = self.fc1[i](t)
            encoded = self.lrelu(t)
            z = encoded
            # mu = self.fc_mu(encoded)
            # log_var = self.fc_var(encoded)
            # print(mu.shape, log_var.shape)
            # sample z from q
            # std = torch.exp(log_var / 2)
            # q = torch.distributions.Normal(mu, std)
            # z = q.rsample()
            if(mode == "val"):
                t = x_target[:,:,0:3]
                # t = x[:,:,3*i:3*(i+1)]
                t = torch.unsqueeze(t, dim=1)
                t = self.conv1[i](t)
                t = self.lrelu(t)
                t = self.norm(t)
                t = self.conv2[i](t)
                t = self.lrelu(t)
                t = self.maxpool(t)
                t = self.flatten(t)
                t = torch.squeeze(t)
                t = self.fc1[i](t)
                encoded = self.lrelu(t)
                z2 = encoded
                z = (z + z2)/2                       
            to_cat.append(z)
            # print(z.shape)
            # input()

        out = torch.cat(to_cat, dim=1)
        out = self.fc2(out)

        # Decoder path
        reconstructed = []
        for i in range(1):
            r = self.ztoencode(z)
            # print(r.shape)
            # input()
            r = self.fc3(r)
            # print(r.shape)
            # input()
            r = self.unflatten(r)
            r = self.upsample(r)
            r = self.lrelu(r)
            r = self.deconv2[i](r)
            r = self.lrelu(r)
            r = self.deconv1[i](r)
            reconstructed.append(r)

        reconstructed = torch.cat(reconstructed, dim=1)  # concatenate along the channel dimension
        x_selected = x[:, :, :3]  # Select only the first 3 columns
        x_selected = x_selected.unsqueeze(1)
        x_target = x_target.unsqueeze(1)
        recon_loss = self.gaussian_likelihood(reconstructed, self.log_scale, x_target)
        # kl
        # kl = self.kl_divergence(z, mu, std)
        # elbo
        # elbo = (kl - recon_loss)
        # print(recon_loss)
        # input()
        elbo = -recon_loss 
        elbo = elbo.mean()
        # print(reconstructed.shape, x.shape)
        # input()

        return out, reconstructed, elbo
    
class SliTCNN1Stream(nn.Module):

    def __init__(self, num_rows, num_columns, num_classes):
        super().__init__()

        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_classes = num_classes

        # if not self.num_columns == 6:
        #         raise Exception("Invalid dataloader!")

        self.conv1 = nn.ModuleList([
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(30,3),stride=1)
            for i in range(1)
        ])

        self.conv2 = nn.ModuleList([
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(8,1),stride=1)
            for i in range(1)
        ])

        self.norm = nn.LayerNorm([64,self.num_rows-29,1])
        self.maxpool = nn.MaxPool2d(kernel_size=(2,1))
        self.lrelu = nn.LeakyReLU()
        self.flatten = nn.Flatten(start_dim=1,end_dim=2)

        self.fc1 = nn.ModuleList([
            nn.Linear(in_features=128*int((self.num_rows-36)/2),out_features=128)
            for i in range(1)
        ])

        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=self.num_classes)

    def forward(self,x):
        to_cat = []

        for i in range(1):
            t = x[:,:,3*i:3*(i+1)]
            t = torch.unsqueeze(t,dim=1)
            t = self.conv1[i](t)
            t = self.lrelu(t)
            t = self.norm(t)
            t = self.conv2[i](t)
            t = self.lrelu(t)
            t = self.maxpool(t)
            t = self.flatten(t)
            t = torch.squeeze(t)
            t = self.fc1[i](t)
            t = self.lrelu(t)
            t = self.dropout(t)
            to_cat.append(t)
        out = torch.cat(to_cat,dim=1)
        out = self.fc2(out)
        return out

class SliTCNN1StreamLSTM(nn.Module):

    def __init__(self, num_rows, num_columns, num_classes, lstm_hidden_size, svc=False):
        super().__init__()

        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_classes = num_classes
        self.lstm_hidden_size = lstm_hidden_size
        print("lstm_hidden_size", self.lstm_hidden_size)

        if svc:
            self.conv1 = nn.ModuleList([
                nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(30,2),stride=1)
                for i in range(1)
            ])
        else:
            self.conv1 = nn.ModuleList([
                nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(30,3),stride=1)
                for i in range(1)
            ])

        self.conv2 = nn.ModuleList([
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(8,1),stride=1)
            for i in range(1)
        ])
        self.lstm = nn.ModuleList([ 
            nn.LSTM(input_size=128, hidden_size=self.lstm_hidden_size, num_layers=1, batch_first=True, dropout=0.5, bidirectional=True)
            for i in range(1)
        ])
        self.norm = nn.LayerNorm([64,self.num_rows-29,1])
        self.maxpool = nn.MaxPool2d(kernel_size=(2,1))
        self.lrelu = nn.LeakyReLU()
        self.flatten = nn.Flatten(start_dim=1,end_dim=2)

        self.fc1 = nn.ModuleList([
            nn.Linear(in_features=int((self.num_rows-36))*self.lstm_hidden_size*2,out_features=128)
            for i in range(1)
        ])

        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=self.num_classes)

    def forward(self,x):
        to_cat = []

        for i in range(1):
            t = x[:,:,3*i:3*(i+1)]
            t = torch.unsqueeze(t,dim=1)
            t = self.conv1[i](t)
            t = self.lrelu(t)
            t = self.norm(t)
            t = self.conv2[i](t)
            t = self.lrelu(t)
            t = t.squeeze(-1) # remove the last dim
            t = t.permute(0, 2, 1) # Change shape to [batch, seq_len, channels]
            t, (hn, cn) = self.lstm[i](t)
            # t = self.maxpool(t)
            # print("shape after lstm", t.shape)
            # input()
            t = self.flatten(t)
            # print(t.shape)
            t = self.fc1[i](t)
            t = self.lrelu(t)
            t = self.dropout(t)
            to_cat.append(t)
        out = torch.cat(to_cat,dim=1)
        out = self.fc2(out)
        return out

class SliTCNN2D_LSTM_ParallelTipTail(nn.Module):
    def __init__(self, num_rows, num_columns, num_classes, lstm_hidden_size):
        super().__init__()

        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_classes = num_classes
        self.lstm_hidden_size = lstm_hidden_size
        print("lstm_hidden_size", self.lstm_hidden_size)
        # if not self.num_columns == 6:
        #     raise Exception("Invalid dataloader!")

        self.conv1 = nn.ModuleList([
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(30,3),stride=1)
            for i in range(2)
        ])

        self.conv2 = nn.ModuleList([
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(8,1),stride=1)
            for i in range(2)
        ])
        self.lstm = nn.ModuleList([ 
            nn.LSTM(input_size=128, hidden_size=self.lstm_hidden_size, num_layers=1, batch_first=True, dropout=0.5, bidirectional=True)
            for i in range(2)
        ])
        self.norm = nn.LayerNorm([64,self.num_rows-29,1])
        self.maxpool = nn.MaxPool2d(kernel_size=(2,1))
        self.lrelu = nn.LeakyReLU()
        self.flatten = nn.Flatten(start_dim=1,end_dim=2)

        # print(int((self.num_rows-36)/2))
        # input()
        self.fc1 = nn.ModuleList([
            nn.Linear(in_features=int((self.num_rows-36))*self.lstm_hidden_size*2,out_features=128)
            for i in range(2)
        ])

        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=256, out_features=self.num_classes)

    def forward(self,x):
        to_cat = []

        for i in range(2):
            t = x[:,:,3*i:3*(i+1)]
            t = torch.unsqueeze(t,dim=1)
            t = self.conv1[i](t)
            t = self.lrelu(t)
            t = self.norm(t)
            t = self.conv2[i](t)
            t = self.lrelu(t)
            t = t.squeeze(-1) # remove the last dim
            t = t.permute(0, 2, 1) # Change shape to [batch, seq_len, channels]
            t, (hn, cn) = self.lstm[i](t)
            # t = self.maxpool(t)
            # print("shape after lstm", t.shape)
            # input()
            t = self.flatten(t)
            # print(t.shape)
            t = self.fc1[i](t)
            t = self.lrelu(t)
            t = self.dropout(t)
            to_cat.append(t)
        out = torch.cat(to_cat,dim=1)
        out = self.fc2(out)
        return out

class SliTCNN2D_LSTM_SequentialTipTail(nn.Module):

    def __init__(self, num_rows, num_columns, num_classes, lstm_hidden_size):
        super().__init__()

        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_classes = num_classes
        self.lstm_hidden_size = lstm_hidden_size

        if not self.num_columns == 6:
            raise Exception("Invalid dataloader!")

        self.conv1 = nn.ModuleList([
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(30,3),stride=1)
            for i in range(2)
        ])

        self.conv2 = nn.ModuleList([
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(8,1),stride=1)
            for i in range(2)
        ])
        
        self.lstm = nn.LSTM(input_size=128, hidden_size=self.lstm_hidden_size, num_layers=1, batch_first=True, dropout=0.5, bidirectional=True)

        self.norm = nn.LayerNorm([64,self.num_rows-29,1])
        self.maxpool = nn.MaxPool2d(kernel_size=(2,1))
        self.lrelu = nn.LeakyReLU()
        self.flatten = nn.Flatten(start_dim=1,end_dim=2)

        self.fc1 = nn.ModuleList([
            nn.Linear(in_features=476*2*lstm_hidden_size,out_features=128)
            for i in range(2)
        ])

        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=256, out_features=self.num_classes)

    def forward(self,x):
        to_cat = []

        for i in range(2):
            t = x[:,:,3*i:3*(i+1)]
            t = torch.unsqueeze(t,dim=1)
            t = self.conv1[i](t)
            t = self.lrelu(t)
            t = self.norm(t)
            t = self.conv2[i](t)
            t = self.lrelu(t)
            # t = self.maxpool(t) # remove maxpool
            t = t.squeeze(-1) # remove the last dim
            t = t.permute(0, 2, 1) # Change shape to [batch, seq_len, channels]
            t, (hn, cn) = self.lstm(t)
            # print("shape after lstm", t.shape)
            # input()
            t = self.flatten(t)
            # print(t.shape)
            t = self.fc1[i](t)
            t = self.lrelu(t)
            t = self.dropout(t)
            to_cat.append(t)
        out = torch.cat(to_cat,dim=1)
        out = self.fc2(out)
        return out