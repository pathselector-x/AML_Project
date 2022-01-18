from re import I
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from model.triplet_match.model import TripletMatch
from PIL import Image

# Data loader
class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, annotations, img_transformer=None):
        self.img_transformer = img_transformer
        self.images = images
        self.annotations = annotations

    def __getitem__(self, index):
        img, txt_path = self.images[index], self.annotations[index]
        img = Image.open(img).convert('RGB')
        img = self.img_transformer(img)
        txt = '' # textual description
        with open(txt_path, 'r') as f:
            txt = f.read()
        return img, txt

    def __len__(self):
        return len(self.images)

def get_transform():
    img_tr = [transforms.Resize(256),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)

# Implementation of the Metric Learning Approach
def main():
    # Positive
    img_path = 'PACS/kfold/art_painting/dog/pic_206.jpg'
    txt_path = 'datalabels/art_painting/dog/pic_206.jpg.txt'
    dataset = Dataset([img_path], [txt_path], get_transform())
    dataset_source = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    
    # Negative
    nimg_path = 'PACS/kfold/art_painting/house/pic_034.jpg'
    ntxt_path = 'datalabels/art_painting/house/pic_034.jpg.txt'
    ndataset = Dataset([nimg_path], [ntxt_path], get_transform())
    ndataset_source = torch.utils.data.DataLoader(ndataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    model = TripletMatch()
    model.cuda()
    model.eval()

    for (img, txt), (nimg, ntxt) in zip(dataset_source, ndataset_source):
        ie = model.img_encoder(img.cuda())
        te =  model.lang_encoder(txt)
        nie = model.img_encoder(nimg.cuda())
        nte =  model.lang_encoder(ntxt)

        positive_norm = torch.pow(torch.norm(ie - te, p=2), 2)
        ie_nte_norm = torch.pow(torch.norm(ie - nte), 2)
        nie_te_norm = torch.pow(torch.norm(nie - te), 2)
        print(1 + positive_norm - ie_nte_norm)
        print(1 + positive_norm - nie_te_norm)
        Lp = torch.max(torch.tensor(0, device='cuda:0'), 1 + positive_norm - ie_nte_norm)
        Li = torch.max(torch.tensor(0, device='cuda:0'), 1 + positive_norm - nie_te_norm)
        loss = Lp + Li #TODO loss = sum(Lp[i] + Li[i]) / len(batch) for i in range(len(batch))
        print(loss)

if __name__ == '__main__':
    main()