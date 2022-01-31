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
import random

def cosine_sim(a,b):
    return ((np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))+1)

#! Data Loading
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, annotations_path, annotations, model):
        self.model = model
        self.img_transformer = get_transform()
        self.indexes = {'art_painting': [], 'cartoon': [], 'photo': [], 'sketch': []}
        self.images = []
        self.descr = []
        idx = 0
        for path in annotations:
            self.images.append(self.img_transformer(Image.open(images_path + path).convert('RGB')))
            with open(annotations_path + path, 'r') as f:
                self.descr.append(f.read())
            if 'art_painting' in path: self.indexes['art_painting'].append(idx)
            elif 'cartoon' in path: self.indexes['cartoon'].append(idx)
            elif 'photo' in path: self.indexes['photo'].append(idx)
            elif 'sketch' in path: self.indexes['sketch'].append(idx)
            idx += 1
    
    def __getitem__(self, index):
        # Positive
        index = index % 4 # 0: art 1: cartoon 2: photo 3: sketch
        key = ['art_painting', 'cartoon', 'photo', 'sketch'][index]
        idx_positive = random.choice(self.indexes[key])
        pos_img, pos_txt = self.images[idx_positive], self.descr[idx_positive] # (I+, P+)

        # Negative
        other_visual_domains = list(set(['art_painting', 'cartoon', 'photo', 'sketch']) - set([key]))
        neg_img = None
        max_sim_img = None
        neg_txt = None
        max_sim_txt = None
        with torch.no_grad():
            for domain in other_visual_domains:
                for idx in self.indexes[domain]:
                    sim_img = cosine_sim(self.model.img_encoder(pos_img), self.model.img_encoder(self.images[idx]))
                    if max_sim_img is None or sim_img > max_sim_img:
                        neg_img = self.images[idx]
                        max_sim_img = sim_img
                    sim_txt = cosine_sim(self.model.lang_encoder(pos_txt), self.model.lang_encoder(self.descr[idx]))
                    if max_sim_txt is None or sim_txt > max_sim_txt:
                        neg_txt = self.descr[idx]
                        max_sim_txt = sim_txt
        return (pos_img, pos_txt, neg_img, neg_txt, key)

    def __len__(self):
        return len(self.images)
    
class EvalDataset(torch.utils.data.Dataset): # both for validation, test -> return (pos_img, pos_txt)
    def __init__(self, images_path, annotations_path, annotations):
        self.img_transformer = get_transform()
        self.indexes = {'art_painting': [], 'cartoon': [], 'photo': [], 'sketch': []}
        self.images = []
        self.descr = []
        idx = 0
        for path in annotations:
            self.images.append(self.img_transformer(Image.open(images_path + path).convert('RGB')))
            with open(annotations_path + path, 'r') as f:
                self.descr.append(f.read())
            if 'art_painting' in path: self.indexes['art_painting'].append(idx)
            elif 'cartoon' in path: self.indexes['cartoon'].append(idx)
            elif 'photo' in path: self.indexes['photo'].append(idx)
            elif 'sketch' in path: self.indexes['sketch'].append(idx)
            idx += 1

    def __getitem__(self, index):
        img, txt = self.images[index], self.descr[index]
        if index in self.indexes['art_painting']: label = 'art_painting'
        elif index in self.indexes['cartoon']: label = 'cartoon'
        elif index in self.indexes['photo']: label = 'photo'
        elif index in self.indexes['sketch']: label = 'sketch'
        return (img, txt, label)
        
    def __len__(self):
        return len(self.images)

def get_transform():
    img_tr = [transforms.Resize(256),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])

def compute_mAP(match_scores, gt_matrix, mode='i2p'):
    """
    INPUT:
    - match_scores: [img_num x phrase_num], match_scores[i,j] = cosine_sim(emb(img_i), emb(phrase_j)) (shows how well img_i and phrase_j matches)
    - gt_matrix: [img_num x phrase_num], gt_matrix[i,j] shows which img_i corresp. to which phrase_j (1 if they corresp., 0 otherwise)
    - mode: 'i2p' = retrieve images given phrases, 'p2i' = retrieve phrases given images
    """
    img_num = gt_matrix.shape[0]
    phrase_num = gt_matrix.shape[1]

    if mode == 'i2p':
        # each row is prediction for one image. phrase sorted by pred scores. values are whether the phrase is correct
        i2p_correct = np.zeros_like(gt_matrix, dtype=bool)  # img_num x phrase_num
        i2p_phrase_idxs = np.zeros_like(i2p_correct, dtype=int)
        for img_i in range(img_num):
            phrase_idx_sorted = np.argsort(-match_scores[img_i, :])
            i2p_phrase_idxs[img_i] = phrase_idx_sorted
            i2p_correct[img_i] = gt_matrix[img_i, phrase_idx_sorted]
        retrieve_binary_lists = i2p_correct
    elif mode == 'p2i':
        # each row is prediction for one prhase. images sorted by pred scores. values are whether the image is correct
        p2i_correct = np.zeros_like(gt_matrix, dtype=bool).transpose()  # class_num x img_num
        p2i_img_idxs = np.zeros_like(p2i_correct, dtype=int)
        for pi in range(phrase_num):
            img_idx_sorted = np.argsort(-match_scores[:, pi])
            p2i_img_idxs[pi] = img_idx_sorted
            p2i_correct[pi] = gt_matrix[img_idx_sorted, pi]
        retrieve_binary_lists = p2i_correct
    else:
        raise NotImplementedError
    
    # calculate mAP
    return mean_average_precision(retrieve_binary_lists)

def plot_clouds():
    BATCH_SIZE = 1
    model_path = 'metric_learning/weights.pth'

    # Load data from path
    images_path = 'PACS/kfold/'
    annotations_path = 'datalabels/'

    print('Starting to load data...')
    dataset = Dataset(images_path, annotations_path, get_transform())
    dataset_source = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    
    # Init model for metric learning
    print('Loading model...')
    model = TripletMatch()
    model.cuda()

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    xs, ys, lbls = [], [], []

    print('Start evaluation...')
    for it, (img, txt, neg_img, neg_txt, lbl) in enumerate(tqdm(dataset_source)):
        ie = model.img_encoder(img.cuda())
        te =  model.lang_encoder(txt)
        x = torch.linalg.svdvals(ie).detach().cpu().numpy()[0]
        y = torch.linalg.svdvals(te).detach().cpu().numpy()[0]
        xs.append(x)
        ys.append(y)
        lbls.append(lbl)

        if it == 2000: break
    
    xs, ys, lbls = np.array(xs), np.array(ys), np.array(lbls)

    plt.scatter(xs[lbls == 0], ys[lbls == 0], c='r')
    plt.scatter(xs[lbls == 1], ys[lbls == 1], c='g')
    plt.scatter(xs[lbls == 2], ys[lbls == 2], c='c')
    plt.scatter(xs[lbls == 3], ys[lbls == 3], c='b')
    plt.grid()
    plt.show()

def MLTLoss(ie, te, nie, nte, BATCH_SIZE):
    positive_norm = torch.pow(torch.norm(ie - te, p=2), 2)
    ie_nte_norm = torch.pow(torch.norm(ie - nte), 2)
    nie_te_norm = torch.pow(torch.norm(nie - te), 2)
    Lp = torch.max(torch.tensor(0, device='cuda:0'), 1 + positive_norm - ie_nte_norm)
    Li = torch.max(torch.tensor(0, device='cuda:0'), 1 + positive_norm - nie_te_norm)
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

#! Implementation of the Metric Learning Approach
def train(eps_improving=0.0005, save_every=50, use_tensorboard=True):
    BATCH_SIZE = 4
    LR = 0.0001
    model_path = 'metric_learning/weights.pth'
    if use_tensorboard: writer = SummaryWriter()

    # Load data from path
    images_path = 'PACS/kfold/'
    annotations_path = 'datalabels/'

    train_txt, val_txt, test_txt = train_val_test_split('datalabels/')

    # Init model for metric learning
    print('Loading model...')
    model = TripletMatch()
    model.cuda()

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
    elif not os.path.exists('./metric_learning'):
        os.mkdir('./metric_learning')

    print('Starting to load data...')
    trainset = TrainDataset(images_path, annotations_path, train_txt, model)
    trainset_source = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

    valset = EvalDataset(images_path, annotations_path, val_txt)
    valset_source = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

    #testset = EvalDataset(images_path, annotations_path, test_txt)
    #testset_source = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

    print('Start training...')
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for it, (img, txt, neg_img, neg_txt, _) in enumerate(tqdm(trainset_source)):
        # Compute loss
        ie = model.img_encoder(img.cuda())
        te =  model.lang_encoder(txt)
        nie = model.img_encoder(neg_img.cuda())
        nte =  model.lang_encoder(neg_txt)

        loss = MLTLoss(ie, te, nie, nte, BATCH_SIZE)

        # Tensorboard
        if use_tensorboard: writer.add_scalar('Loss/train', loss, it)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % save_every == 0:
            mAP_img = 0.0 #TODO: implement mAP calc
            mAP_txt = 0.0 #TODO: implement mAP calc
            # for img, phr in validation_split:
            # ...
            # ... compute_mAP(match_scores, gt_matrix, mode='i2p')

            if (mAP_img + mAP_txt) / 2 > eps_improving:
                torch.save(model.state_dict(), model_path)

            if use_tensorboard:
                writer.add_scalar('Improvement', (mAP_img + mAP_txt) / 2, it)
                writer.add_scalar('Improvement', eps_improving, it)
                writer.add_scalar('mAP_img/val', mAP_img, it)
                writer.add_scalar('mAP_txt/val', mAP_txt, it)

    if use_tensorboard: writer.close()

if __name__ == '__main__':
    train()
    #plot_clouds()