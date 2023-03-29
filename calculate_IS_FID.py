# from https://github.com/w86763777/pytorch-gan-metrics 

from pytorch_gan_metrics import get_inception_score_and_fid, ImageDataset
from torch.utils.data import DataLoader

data_dir = str(input("Enter path to generated images: "))
stats = str(input("Enter path to stats file (.npz ending): "))

# loading images
print("Loading images ...")
data = ImageDataset(root=data_dir)
dataloader = DataLoader(data, batch_size=50)

# calculate IS and FID
print("Calculate scores ...")
(IS, IS_std), FID = get_inception_score_and_fid(dataloader, stats)
print("IS mean = ", IS, " IS std = ", IS_std, " FID = ", FID)
