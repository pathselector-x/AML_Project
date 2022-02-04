import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from model.triplet_match.model import TripletMatch
from PIL import Image
import time

def get_transform():
    img_tr = [transforms.Resize(256),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, sample_list, images_path='./PACS/kfold/', annotations_path='./datalabels/'):
        self.img_transform = get_transform()
        self.images = []
        self.descr = []
        self.visual_domain = []
        for path in sample_list:
            self.images.append(torch.unsqueeze(self.img_transform(Image.open(images_path + path.split('.txt')[0]).convert('RGB')), dim=0))
            with open(annotations_path + path, 'r') as f:
                self.descr.append(f.read())
            if 'art_painting' in path: self.visual_domain.append(0)
            elif 'cartoon' in path: self.visual_domain.append(1)
            elif 'photo' in path: self.visual_domain.append(2)
            elif 'sketch' in path: self.visual_domain.append(3)
        self.t_images = torch.Tensor(len(sample_list), 3, 224, 224)
        torch.cat(self.images, out=self.t_images)
        self.t_images

    def __len__(self):
        return len(self.images)

def MLTLoss(ie, te, nie, nte, BATCH_SIZE):
    positive_norm = torch.pow(torch.norm(ie - te, p=2, dim=1), 2)
    ie_nte_norm = torch.pow(torch.norm(ie - nte, p=2, dim=1), 2)
    nie_te_norm = torch.pow(torch.norm(nie - te, p=2, dim=1), 2)
    Lp = torch.maximum(torch.zeros(BATCH_SIZE, device='cuda:0'), torch.ones(BATCH_SIZE, device='cuda:0') + positive_norm - ie_nte_norm)
    Li = torch.maximum(torch.zeros(BATCH_SIZE, device='cuda:0'), torch.ones(BATCH_SIZE, device='cuda:0') + positive_norm - nie_te_norm)
    return torch.sum(Lp + Li) / BATCH_SIZE

def train_val_test_split(annotations_path, train_pctg=0.6, val_pctg=0.2, test_pctg=0.2):
    assert np.isclose(train_pctg + val_pctg + test_pctg, 1.0)
    train_txt, val_txt, test_txt = [], [], []
    _, visual_domains, _ = next(os.walk(annotations_path))
    for domain in visual_domains:
        _, categories, _ = next(os.walk(annotations_path + domain))
        for cat in categories:
            _, _, files = next(os.walk(annotations_path + domain + '/' + cat))
            N = len(files)
            Ntrain, Nval = np.floor(N * train_pctg), np.ceil(N * val_pctg)
            Nval += Ntrain
            for i, file in enumerate(files):
                if i < Ntrain: train_txt.append(domain + '/' + cat + '/' + file)
                elif i < Nval: val_txt.append(domain + '/' + cat + '/' + file)
                else: test_txt.append(domain + '/' + cat + '/' + file)
    return train_txt, val_txt, test_txt

def generate_minibatch(model, trainset, batch_size, mode='hard'):
    assert mode in ['hard', 'semihard']
    # Semi-hard negative mining
    #TODO
    if mode == 'semihard':
        pass

    # Online hard mining
    elif mode == 'hard':
        with torch.no_grad():
            out_img = model.img_encoder(trainset.t_images.cuda()).cpu()
            out_txt = model.lang_encoder(trainset.descr).cpu()
            # the smaller, the harder
            pos, pos_indices = torch.norm(out_img - out_txt, p=2, dim=1).sort()
            pos_indices = pos_indices[-batch_size:]

            pos_images = []
            pos_phrase = []
            neg_images = []
            neg_phrase = []
            for idx in pos_indices:
                pos_images.append(trainset.images[idx])
                pos_phrase.append(trainset.descr[idx])
                f_pos_i = out_img[idx]
                f_pos_p = out_txt[idx]
                
                pos_vd = trainset.visual_domain[idx]
                pos_i_neg_p = []
                pos_p_neg_i = []
                for i in range(len(trainset)):
                    if trainset.visual_domain[i] == pos_vd: 
                        pos_i_neg_p.append(np.inf)
                        pos_p_neg_i.append(np.inf)
                        continue
                    pos_i_neg_p.append(np.linalg.norm((f_pos_i - out_txt[i]).numpy(), ord=2))
                    pos_p_neg_i.append(np.linalg.norm((out_img[i] - f_pos_p).numpy(), ord=2))

                neg_p_idx = [x for _, x in sorted(zip(pos_i_neg_p, range(len(pos_i_neg_p))))][0]
                neg_i_idx = [x for _, x in sorted(zip(pos_p_neg_i, range(len(pos_p_neg_i))))][0]
                neg_images.append(trainset.images[neg_i_idx])
                neg_phrase.append(trainset.descr[neg_p_idx])
            
            t_pos_images = torch.Tensor(batch_size, 3, 224, 224)
            t_neg_images = torch.Tensor(batch_size, 3, 224, 224)
            torch.cat(pos_images, out=t_pos_images)
            torch.cat(neg_images, out=t_neg_images)
            return t_pos_images, pos_phrase, t_neg_images, neg_phrase


                    
            



def train(batch_size=16, mode='hard'):
    model_path = 'metric_learning/weights.pth'
    train_list, val_list, _ = train_val_test_split('datalabels/')
    
    trainset = TrainDataset(train_list)

    model = TripletMatch()
    model.cuda()

    pos_images, pos_phrase, neg_images, neg_phrase = generate_minibatch(model, trainset, batch_size, mode)
    

if __name__ == '__main__':
    train()