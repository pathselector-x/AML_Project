import os
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from model.triplet_match.model import TripletMatch
from PIL import Image
import matplotlib.pyplot as plt

#! Data Loading
class Dataset(torch.utils.data.Dataset):
    def __init__(self, images_path, annotations_path, img_transformer=None):
        self.img_transformer = img_transformer
        self.idx_artpainting = []
        self.idx_cartoon = []
        self.idx_photo = []
        self.idx_sketch = []
        self.images = []
        self.annotations = []
        idx = 0
        for root, dirs, files in os.walk(annotations_path):
            if len(files) > 0:
                for file in files:
                    full_path = os.path.join(root, file)
                    normalized = os.path.normpath(full_path).split('\\')
                    self.images.append(images_path + normalized[1] + '/' + normalized[2] + '/' + normalized[3].split('.txt')[0])
                    self.annotations.append(annotations_path + normalized[1] + '/' + normalized[2] + '/' + normalized[3])
                    if normalized[1] == 'art_painting': self.idx_artpainting.append(idx)
                    elif normalized[1] == 'cartoon': self.idx_cartoon.append(idx)
                    elif normalized[1] == 'photo': self.idx_photo.append(idx)
                    elif normalized[1] == 'sketch': self.idx_sketch.append(idx)
                    idx += 1
        self.idx_pos_neg = [] # I want "all" permutations of (img, ann, neg_img, neg_ann)
        # 3 ways of selecting negative:
        # - negative = different visual domain, same object
        # - negative = different visual domain, different object
        # - negative = different visual domain, same object and different object <-- I'll go with this
        for i in self.idx_artpainting:
            for j in self.idx_cartoon: self.idx_pos_neg.append((i, j))
            for j in self.idx_photo: self.idx_pos_neg.append((i, j))
            for j in self.idx_sketch: self.idx_pos_neg.append((i, j))
        for i in self.idx_cartoon:
            for j in self.idx_artpainting: self.idx_pos_neg.append((i, j))
            for j in self.idx_photo: self.idx_pos_neg.append((i, j))
            for j in self.idx_sketch: self.idx_pos_neg.append((i, j))
        for i in self.idx_photo:
            for j in self.idx_artpainting: self.idx_pos_neg.append((i, j))
            for j in self.idx_cartoon: self.idx_pos_neg.append((i, j))
            for j in self.idx_sketch: self.idx_pos_neg.append((i, j))
        for i in self.idx_sketch:
            for j in self.idx_artpainting: self.idx_pos_neg.append((i, j))
            for j in self.idx_cartoon: self.idx_pos_neg.append((i, j))
            for j in self.idx_photo: self.idx_pos_neg.append((i, j))
        # in this case we consider (I', P') also paired like (I, P)

    def __getitem__(self, index):
        idx_positive, idx_negative = self.idx_pos_neg[index]
        img_path, txt_path = self.images[idx_positive], self.annotations[idx_positive]
        neg_img_path, neg_txt_path = self.images[idx_negative], self.annotations[idx_negative]

        img = Image.open(img_path).convert('RGB')
        img = self.img_transformer(img)

        neg_img = Image.open(neg_img_path).convert('RGB')
        neg_img = self.img_transformer(neg_img)

        txt = ''
        with open(txt_path, 'r') as f: txt = f.read()

        neg_txt = ''
        with open(neg_txt_path, 'r') as f: neg_txt = f.read()

        return img, txt, neg_img, neg_txt

    def __len__(self):
        return len(self.idx_pos_neg)

def get_transform():
    img_tr = [transforms.Resize(256),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)

#TODO evaluation metric, LR scheduling/finetuning
#! Implementation of the Metric Learning Approach
def main():
    BATCH_SIZE = 1 #4
    LR = 0.0001
    model_path = 'metric_learning/weights.pth'
    writer = SummaryWriter()

    # Load data from path
    images_path = 'PACS/kfold/'
    annotations_path = 'datalabels/'

    dataset = Dataset(images_path, annotations_path, get_transform())
    dataset_source = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    
    # Init model for metric learning
    model = TripletMatch()
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
    elif not os.path.exists('./metric_learning'):
        os.mkdir('./metric_learning')

    for it, (img, txt, neg_img, neg_txt) in enumerate(tqdm(dataset_source)):
        # Compute loss
        ie = model.img_encoder(img.cuda())
        te =  model.lang_encoder(txt)
        nie = model.img_encoder(neg_img.cuda())
        nte =  model.lang_encoder(neg_txt)
        positive_norm = torch.pow(torch.norm(ie - te, p=2), 2)
        ie_nte_norm = torch.pow(torch.norm(ie - nte), 2)
        nie_te_norm = torch.pow(torch.norm(nie - te), 2)
        Lp = torch.max(torch.tensor(0, device='cuda:0'), 1 + positive_norm - ie_nte_norm)
        Li = torch.max(torch.tensor(0, device='cuda:0'), 1 + positive_norm - nie_te_norm)
        loss = torch.sum(Lp + Li) / BATCH_SIZE

        # Tensorboard
        writer.add_scalar('Loss/train', loss, it)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 50 == 0:
            torch.save(model.state_dict(), model_path)

    torch.save(model.state_dict(), model_path)

    writer.close()

if __name__ == '__main__':
    main()