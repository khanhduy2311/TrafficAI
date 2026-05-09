import os, glob, itertools
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

MODE = 'train'
IMG_TRAIN_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3),
            nn.InstanceNorm2d(c),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3),
            nn.InstanceNorm2d(c)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        ]

        for _ in range(6):
            model += [ResBlock(256)]

        model += [
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        ]

        self.net = nn.Sequential(*model)

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.net(x)

class DayNightDataset(Dataset):
    def __init__(self):
        self.transform = T.Compose([
            T.Resize((IMG_TRAIN_SIZE, IMG_TRAIN_SIZE)),
            T.ToTensor(),
            T.Normalize((0.5,) * 3, (0.5,) * 3)
        ])

        self.day_imgs = glob.glob("dataset/day/*.*")
        self.night_imgs = glob.glob("dataset/night/*.*")

    def __len__(self):
        return max(len(self.day_imgs), len(self.night_imgs))

    def __getitem__(self, i):
        day = self.transform(
            Image.open(self.day_imgs[i % len(self.day_imgs)]).convert('RGB')
        )

        night = self.transform(
            Image.open(self.night_imgs[i % len(self.night_imgs)]).convert('RGB')
        )

        return day, night

G_Day2Night = Generator().to(DEVICE)
G_Night2Day = Generator().to(DEVICE)

D_Night = Discriminator().to(DEVICE)
D_Day = Discriminator().to(DEVICE)

if MODE == 'train':

    loader = DataLoader(
        DayNightDataset(),
        batch_size=1,
        shuffle=True
    )

    opt_G = torch.optim.Adam(
        itertools.chain(
            G_Day2Night.parameters(),
            G_Night2Day.parameters()
        ),
        lr=0.0002
    )

    opt_D = torch.optim.Adam(
        itertools.chain(
            D_Day.parameters(),
            D_Night.parameters()
        ),
        lr=0.0002
    )

    MSE_Loss = nn.MSELoss()
    L1_Loss = nn.L1Loss()

    os.makedirs("models", exist_ok=True)

    for epoch in range(100):

        for day, night in loader:

            day = day.to(DEVICE)
            night = night.to(DEVICE)

            valid = torch.ones(1, 1, 31, 31).to(DEVICE)
            fake = torch.zeros(1, 1, 31, 31).to(DEVICE)

            opt_G.zero_grad()

            fake_night = G_Day2Night(day)
            fake_day = G_Night2Day(night)

            loss_GAN = (
                MSE_Loss(D_Night(fake_night), valid) +
                MSE_Loss(D_Day(fake_day), valid)
            )

            loss_Cycle = (
                L1_Loss(G_Night2Day(fake_night), day) +
                L1_Loss(G_Day2Night(fake_day), night)
            )

            loss_G = loss_GAN + (10.0 * loss_Cycle)

            loss_G.backward()
            opt_G.step()

            opt_D.zero_grad()

            loss_D = (
                MSE_Loss(D_Day(day), valid) +
                MSE_Loss(D_Day(fake_day.detach()), fake) +
                MSE_Loss(D_Night(night), valid) +
                MSE_Loss(D_Night(fake_night.detach()), fake)
            ) / 2

            loss_D.backward()
            opt_D.step()

        print(
            f"Epoch {epoch}/100 | "
            f"Loss G: {loss_G.item():.4f} | "
            f"Loss D: {loss_D.item():.4f}"
        )

        torch.save(
            G_Day2Night.state_dict(),
            "models/Day2Night.pth"
        )

elif MODE == 'generate':

    os.makedirs("output", exist_ok=True)

    G_Day2Night.load_state_dict(
        torch.load(
            "models/Day2Night.pth",
            map_location=DEVICE
        )
    )

    G_Day2Night.eval()

    trans_infer = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,) * 3, (0.5,) * 3)
    ])

    for path in glob.glob("test_images/*.*"):

        name = os.path.basename(path)

        img = Image.open(path).convert('RGB')

        tensor_img = trans_infer(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out_tensor = G_Day2Night(tensor_img)

        out_img = (
            (
                out_tensor[0]
                .cpu()
                .numpy()
                .transpose(1, 2, 0) + 1
            ) / 2.0 * 255
        ).astype('uint8')

        Image.fromarray(out_img).save(
            f"output/night_{name}"
        )

        print(f"Đã tạo ảnh: night_{name}")
