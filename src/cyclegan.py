import os
import glob
import random
import itertools
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ==========================================
# 1. CẤU HÌNH (HYPERPARAMETERS)
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
LEARNING_RATE = 0.0002
NUM_EPOCHS = 100 # Chạy ít nhất 100-200 epochs để có kết quả tốt
IMG_SIZE = 256
CHANNELS = 3

# ==========================================
# 2. KIẾN TRÚC MẠNG (NETWORKS)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    # Dùng để chuyển đổi Day -> Night (và ngược lại)
    def __init__(self, num_residual_blocks=9):
        super(Generator, self).__init__()
        
        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, 3, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    # Dùng để phân biệt ảnh thật/giả
    def __init__(self):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize: layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(3, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

# ==========================================
# 3. XỬ LÝ DỮ LIỆU (DATASET)
# ==========================================
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob(os.path.join(root, "trainA" if mode=="train" else "testA") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "trainB" if mode=="train" else "testB") + "/*.*"))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert("RGB"))
        item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert("RGB"))
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

# ==========================================
# 4. HÀM TRAIN MÔ HÌNH
# ==========================================
def train_model():
    print(f"Đang dùng thiết bị: {DEVICE}")
    os.makedirs("saved_models", exist_ok=True)

    # Khởi tạo mạng
    G_AB = Generator().to(DEVICE) # Đổi Day -> Night
    G_BA = Generator().to(DEVICE) # Đổi Night -> Day
    D_A = Discriminator().to(DEVICE)
    D_B = Discriminator().to(DEVICE)

    # Loss và Optimizers
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    optimizer_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # Dataloader
    transform = [
        transforms.Resize(int(IMG_SIZE * 1.12), Image.BICUBIC),
        transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    dataloader = DataLoader(ImageDataset("dataset", transforms_=transform, mode="train"), 
                            batch_size=BATCH_SIZE, shuffle=True)

    # Vòng lặp Train
    for epoch in range(NUM_EPOCHS):
        for i, batch in enumerate(dataloader):
            real_A = batch["A"].to(DEVICE) # Day
            real_B = batch["B"].to(DEVICE) # Night

            valid = torch.ones((real_A.size(0), 1, 16, 16), device=DEVICE)
            fake = torch.zeros((real_A.size(0), 1, 16, 16), device=DEVICE)

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()

            # Identity loss (giữ màu sắc)
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A) # Sinh Night giả
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B) # Sinh Day giả
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss (Day -> Night -> Day phải giống Day cũ)
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + 10.0 * loss_cycle + 5.0 * loss_identity
            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminators
            # -----------------------
            # D_A
            optimizer_D_A.zero_grad()
            loss_real = criterion_GAN(D_A(real_A), valid)
            loss_fake = criterion_GAN(D_A(fake_A.detach()), fake)
            loss_D_A = (loss_real + loss_fake) / 2
            loss_D_A.backward()
            optimizer_D_A.step()

            # D_B
            optimizer_D_B.zero_grad()
            loss_real = criterion_GAN(D_B(real_B), valid)
            loss_fake = criterion_GAN(D_B(fake_B.detach()), fake)
            loss_D_B = (loss_real + loss_fake) / 2
            loss_D_B.backward()
            optimizer_D_B.step()

            if i % 50 == 0:
                print(f"[Epoch {epoch}/{NUM_EPOCHS}] [Batch {i}/{len(dataloader)}] [Loss G: {loss_G.item():.4f}]")

        # Lưu model sau mỗi 10 epoch
        if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
            torch.save(G_AB.state_dict(), f"saved_models/G_Day2Night_epoch_{epoch}.pth")
            print(f"Đã lưu model epoch {epoch}")

# ==========================================
# 5. HÀM TẠO DATA AUGMENTATION (INFERENCE)
# ==========================================
def generate_data(model_path, input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Load model
    G_AB = Generator().to(DEVICE)
    try:
        G_AB.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except Exception as e:
        print(f"Không tìm thấy hoặc lỗi load model: {e}")
        return
        
    G_AB.eval()

    # Cấu hình biến đổi ảnh
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image_paths = glob.glob(os.path.join(input_folder, "*.*"))
    print(f"Tìm thấy {len(image_paths)} ảnh để chuyển đổi...")

    for path in image_paths:
        img_name = os.path.basename(path)
        img = Image.open(path).convert('RGB')
        original_size = img.size # Lưu lại size gốc để resize lại (W, H)
        
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            fake_night_tensor = G_AB(img_tensor)

        # Đưa tensor về ảnh numpy/PIL
        img_array = fake_night_tensor[0].cpu().float().numpy()
        img_array = (np.transpose(img_array, (1, 2, 0)) + 1) / 2.0 * 255.0
        img_array = img_array.astype(np.uint8)
        
        result_img = Image.fromarray(img_array)
        result_img = result_img.resize(original_size, Image.BICUBIC) # Đưa về size gốc của ảnh
        
        save_path = os.path.join(output_folder, f"night_{img_name}")
        result_img.save(save_path)
        print(f"Đã tạo: {save_path}")

# ==========================================
# 6. ĐIỀU KHIỂN CHƯƠNG TRÌNH CHÍNH
# ==========================================
import numpy as np # import thêm np cho phần generate

if __name__ == '__main__':
    # -----------------------------------------------------------------
    # CHỌN CHẾ ĐỘ Ở ĐÂY:
    # Đổi thành 'train' nếu muốn huấn luyện mô hình.
    # Đổi thành 'generate' nếu đã train xong và muốn tạo ảnh data.
    # -----------------------------------------------------------------
    MODE = 'generate' 
    
    if MODE == 'train':
        print("BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN CYCLEGAN...")
        train_model()
        
    elif MODE == 'generate':
        print("BẮT ĐẦU QUÁ TRÌNH TẠO ẢNH BAN ĐÊM (AUGMENTATION)...")
        # Thay đổi file model_path dưới đây thành file epoch cao nhất bạn có
        model_weights_path = "saved_models/G_Day2Night_epoch_90.pth" 
        input_day_images = "test_images" 
        output_night_images = "output_images"
        
        generate_data(model_weights_path, input_day_images, output_night_images)
