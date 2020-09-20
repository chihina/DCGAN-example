# coding: utf-8
import torch
from torch import nn, optim
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, datasets
from tqdm import tqdm
from statistics import mean

from model.generator import Generator
from model.discriminator import Discriminator

import datetime 
from pytz import timezone
import sys

# === 1. データの読み込み ===
# datasetrの準備
dataset = datasets.ImageFolder("datasets_resized_dog/",
    transform=transforms.Compose([
        transforms.ToTensor()
]))

# バッチサイズ定義
batch_size = 64

# epoch数の定義
epoch_num = 10

# dataloaderの準備
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# モデル定義
model_G = Generator()
model_D = Discriminator()

cuda = torch.cuda.is_available()
if cuda:
    model_G.cuda()
    model_D.cuda()
    print('cuda is available!')

else:
    print('cuda is not available')

# パラメータ設定
# params_G = optim.Adam(model_G.parameters(),
#     lr=0.0002, betas=(0.5, 0.999))
# params_D = optim.Adam(model_D.parameters(),
#     lr=0.0002, betas=(0.5, 0.999))

params_G = optim.Adam(model_G.parameters(),
    lr=0.01)
params_D = optim.Adam(model_D.parameters(),
    lr=0.01)

# 潜在特徴100次元ベクトルz
nz = 100

# ロスを計算するときのラベル変数
if cuda:
    ones = torch.ones(batch_size).cuda() # 正例 1
    zeros = torch.zeros(batch_size).cuda() # 負例 0

loss_f = nn.BCEWithLogitsLoss()

# 途中結果の確認用の潜在特徴z
check_z = torch.randn(batch_size, nz, 1, 1).cuda()

data_length = len(data_loader)

# 訓練関数
def train_dcgan(model_G, model_D, params_G, params_D, data_loader):
    log_loss_G = []
    log_loss_D = []

    count = 1
    for real_img, _ in data_loader:
  
        # print("{}/{} iteration finished".format(count,data_length))
        count += 1

        batch_len = len(real_img)


        # == Generatorの訓練 ==
        # 偽画像を生成
        z = torch.randn(batch_len, nz, 1, 1).cuda()
        fake_img = model_G(z)

        # 偽画像の値を一時的に保存 => 注(１)
        fake_img_tensor = fake_img.detach()

        # 偽画像を実画像（ラベル１）と騙せるようにロスを計算
        out = model_D(fake_img)
        loss_G = loss_f(out, ones[: batch_len])
        log_loss_G.append(loss_G.item())

        # 微分計算・重み更新 => 注（２）
        model_D.zero_grad()
        model_G.zero_grad()
        loss_G.backward()
        params_G.step()


        # == Discriminatorの訓練 ==
        # sample_dataの実画像
        real_img = real_img.cuda()

        # 実画像を実画像（ラベル１）と識別できるようにロスを計算
        real_out = model_D(real_img)

        print(ones[:batch_len])
        print(ones[:batch_len].size())

        print(real_out)
        print(real_out.size())
        sys.exit()
        loss_D_real = loss_f(real_out, ones[: batch_len])

        # 計算省略 => 注（１）
        fake_img = fake_img_tensor

        # 偽画像を偽画像（ラベル０）と識別できるようにロスを計算
        fake_out = model_D(fake_img_tensor)
        loss_D_fake = loss_f(fake_out, zeros[: batch_len])

        # 実画像と偽画像のロスを合計
        loss_D = loss_D_real + loss_D_fake
        log_loss_D.append(loss_D.item())

        # 微分計算・重み更新 => 注（２）
        model_D.zero_grad()
        model_G.zero_grad()
        loss_D.backward()
        params_D.step()

    return mean(log_loss_G), mean(log_loss_D)

with open("loss_G.log", mode="w") as f:
    pass

with open("loss_D.log", mode="w") as f:
    pass

for epoch in tqdm(range(epoch_num)):
    log_loss_G, log_loss_D = train_dcgan(model_G, model_D, params_G, params_D, data_loader)

    print("{}/{} epoch finished".format(epoch + 1, epoch_num))

    with open("loss.log", mode="a") as f:
        f.write("{}/{} epoch G_loss : {}, D_loss : {} \n".format(epoch+1, epoch_num, log_loss_G, log_loss_D))
    print("G_loss : {}, D_loss : {}".format(log_loss_G, log_loss_D))

    with open("loss_G.log", mode="a") as f:
        f.write(str(log_loss_G) + "\n")

    with open("loss_D.log", mode="a") as f:
        f.write(str(log_loss_D) + "\n")   

    # 訓練途中のモデル・生成画像の保存
    if epoch % 3 == 0:
        torch.save(
            model_G.state_dict(),
            "Weight_Generator/" + datetime.datetime.now(timezone('Asia/Tokyo')).strftime("%Y_%m_%d_%H_%M_%S") + "_G_{:03d}.prm".format(epoch),
            pickle_protocol=4)
        torch.save(
            model_D.state_dict(),
            "Weight_Discriminator/" + datetime.datetime.now(timezone('Asia/Tokyo')).strftime("%Y_%m_%d_%H_%M_%S") + "_D_{:03d}.prm".format(epoch),
            pickle_protocol=4)

        generated_img = model_G(check_z)
        save_image(generated_img,
                   "Generated_Image/" + datetime.datetime.now(timezone('Asia/Tokyo')).strftime("%Y_%m_%d_%H_%M_%S") + "{:03d}.jpg".format(epoch))
