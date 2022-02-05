import os
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import numpy as np
from model.triplet_match.model import TripletMatch
from PIL import Image
import random

def cosine_sim(a,b):
    return ((np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))+1)

def build_transforms(is_train=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, sample_list, images_path='./PACS/kfold/', annotations_path='./datalabels/'):
        self.img_transform = build_transforms()
        self.images = []
        self.descr = []
        self.visual_domain = []
        for path in sample_list:
            self.images.append(Image.open(images_path + path.split('.txt')[0]).convert('RGB'))
            with open(annotations_path + path, 'r') as f:
                self.descr.append(f.read())
            if 'art_painting' in path: self.visual_domain.append(0)
            elif 'cartoon' in path: self.visual_domain.append(1)
            elif 'photo' in path: self.visual_domain.append(2)
            elif 'sketch' in path: self.visual_domain.append(3)

    def __len__(self):
        return len(self.images)

class EvalDataset(torch.utils.data.Dataset): # both for validation, test -> return (pos_img, pos_txt)
    def __init__(self, sample_list, images_path='./PACS/kfold/', annotations_path='./datalabels/'):
        self.img_transformer = build_transforms(is_train=False)
        self.indexes = {'art_painting': [], 'cartoon': [], 'photo': [], 'sketch': []}
        self.images = []
        self.descr = []
        idx = 0
        for path in sample_list:
            self.images.append(torch.unsqueeze(self.img_transformer(Image.open(images_path + path.split('.txt')[0]).convert('RGB')), dim=0))
            with open(annotations_path + path, 'r') as f:
                self.descr.append(f.read())
            if 'art_painting' in path: self.indexes['art_painting'].append(idx)
            elif 'cartoon' in path: self.indexes['cartoon'].append(idx)
            elif 'photo' in path: self.indexes['photo'].append(idx)
            elif 'sketch' in path: self.indexes['sketch'].append(idx)
            idx += 1
        self.t_images = torch.Tensor(len(sample_list), 3, 224, 224)
        torch.cat(self.images, out=self.t_images)

    def __getitem__(self, index):
        img, txt = self.images[index], self.descr[index]
        if index in self.indexes['art_painting']: label = 'art_painting'
        elif index in self.indexes['cartoon']: label = 'cartoon'
        elif index in self.indexes['photo']: label = 'photo'
        elif index in self.indexes['sketch']: label = 'sketch'
        return (img, txt, label)

    def __len__(self):
        return len(self.images)

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

static_idx_counter = 0
def generate_minibatch(model, trainset, batch_size, mode='batch_hard'):
    global static_idx_counter
    # Random selection
    if mode == 'random':
        imgs = []
        for img in trainset.images:
            imgs.append(torch.unsqueeze(trainset.img_transform(img), dim=0))
        t_images = torch.Tensor(len(trainset), 3, 224, 224)
        torch.cat(imgs, out=t_images)
        pos_indices = [(x + static_idx_counter) % len(trainset) for x in range(batch_size)]
        static_idx_counter = (static_idx_counter + batch_size) % len(trainset)
        pos_images = []
        pos_phrase = []
        neg_images = []
        neg_phrase = []
        for idx in pos_indices:
            pos_images.append(torch.unsqueeze(t_images[idx], dim=0).cpu())
            pos_phrase.append(trainset.descr[idx])
            pos_vd = trainset.visual_domain[idx]
            i = None
            while i is None or trainset.visual_domain[i] == pos_vd:
                i = random.choice(range(len(trainset)))
            neg_images.append(torch.unsqueeze(t_images[i], dim=0).cpu())
            i = None
            while i is None or trainset.visual_domain[i] == pos_vd:
                i = random.choice(range(len(trainset)))
            neg_phrase.append(trainset.descr[i])
        
        t_pos_images = torch.Tensor(batch_size, 3, 224, 224)
        t_neg_images = torch.Tensor(batch_size, 3, 224, 224)
        torch.cat(pos_images, out=t_pos_images)
        torch.cat(neg_images, out=t_neg_images)
        return t_pos_images, pos_phrase, t_neg_images, neg_phrase

    # Online hard-negative mining
    elif mode == 'hard_negative':
        with torch.no_grad():
            imgs = []
            for img in trainset.images:
                imgs.append(torch.unsqueeze(trainset.img_transform(img), dim=0))
            t_images = torch.Tensor(len(trainset), 3, 224, 224)
            torch.cat(imgs, out=t_images)

            out_img = model.img_encoder(t_images.cuda()).cpu()
            out_txt = model.lang_encoder(trainset.descr).cpu()
            
            pos_indices = [(x + static_idx_counter) % len(trainset) for x in range(batch_size)]
            static_idx_counter = (static_idx_counter + batch_size) % len(trainset)

            pos_images = []
            pos_phrase = []
            neg_images = []
            neg_phrase = []
            for idx in pos_indices:
                pos_images.append(torch.unsqueeze(t_images[idx], dim=0).cpu())
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
                neg_images.append(torch.unsqueeze(t_images[neg_i_idx], dim=0).cpu())
                neg_phrase.append(trainset.descr[neg_p_idx])
            
            t_pos_images = torch.Tensor(batch_size, 3, 224, 224)
            t_neg_images = torch.Tensor(batch_size, 3, 224, 224)
            torch.cat(pos_images, out=t_pos_images)
            torch.cat(neg_images, out=t_neg_images)
            return t_pos_images, pos_phrase, t_neg_images, neg_phrase

    # Batch hard mining
    elif mode == 'batch_hard':
        with torch.no_grad():
            imgs = []
            for img in trainset.images:
                imgs.append(torch.unsqueeze(trainset.img_transform(img), dim=0))
            t_images = torch.Tensor(len(trainset), 3, 224, 224)
            torch.cat(imgs, out=t_images)
            
            out_img = model.img_encoder(t_images.cuda()).cpu()
            out_txt = model.lang_encoder(trainset.descr).cpu()
            
            pos, pos_indices = torch.norm(out_img - out_txt, p=2, dim=1).sort()
            pos_indices = pos_indices[-batch_size:]

            pos_images = []
            pos_phrase = []
            neg_images = []
            neg_phrase = []
            for idx in pos_indices:
                pos_images.append(torch.unsqueeze(t_images[idx], dim=0).cpu())
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
                neg_images.append(torch.unsqueeze(t_images[neg_i_idx], dim=0).cpu())
                neg_phrase.append(trainset.descr[neg_p_idx])
            
            t_pos_images = torch.Tensor(batch_size, 3, 224, 224)
            t_neg_images = torch.Tensor(batch_size, 3, 224, 224)
            torch.cat(pos_images, out=t_pos_images)
            torch.cat(neg_images, out=t_neg_images)
            return t_pos_images, pos_phrase, t_neg_images, neg_phrase
    else:
        raise NotImplementedError

def train(use_tensorboard=True):
    batch_size=16
    mode='batch_hard'
    val_every=20

    #init_lr=0.000002 # for hard-negative
    init_lr=0.00001 # for batch-hard
    lr_decay_gamma = 0.1
    lr_decay_eval_count = 10

    weight_decay = 1e-6
    alpha = 0.8
    beta = 0.999
    epsilon = 1e-8

    train_list, val_list, _ = train_val_test_split('datalabels/')
    if use_tensorboard: writer = SummaryWriter()

    trainset = TrainDataset(train_list)
    valset = EvalDataset(val_list)

    model = TripletMatch()
    model.cuda()

    if os.path.exists('metric_learning/LAST_checkpoint.pth'):
        model.load_state_dict(torch.load('metric_learning/LAST_checkpoint.pth'), strict=False)
    elif not os.path.exists('./metric_learning'):
        os.mkdir('./metric_learning')

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay, betas=(alpha, beta), eps=epsilon)

    best_eval_metric = 0
    best_eval_count = 0
    it = 0
    try:
        while True:
            # Train
            pos_images, pos_phrase, neg_images, neg_phrase = generate_minibatch(model, trainset, batch_size, mode)

            neg_img_loss, neg_sent_loss = model(pos_images.cuda(), pos_phrase, neg_images.cuda(), neg_phrase)
            loss = neg_img_loss + neg_sent_loss

            if use_tensorboard: writer.add_scalar('Loss/train', loss, it)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation
            if it % val_every == val_every-1:
                with torch.no_grad():
                    out_img = model.img_encoder(valset.t_images.cuda()).cpu().numpy()
                    out_txt = model.lang_encoder(valset.descr).cpu().numpy()
                    match_scores = np.zeros((len(valset), len(valset)))
                    gt_matrix = np.eye(len(valset))
                    for i, img in enumerate(out_img):
                        for j, phr in enumerate(out_txt):
                            match_scores[i,j] = - np.sum(np.power(img - phr, 2)) # l2_s
                    
                    mAP_i2p = compute_mAP(match_scores, gt_matrix, mode='i2p')
                    mAP_p2i = compute_mAP(match_scores, gt_matrix, mode='p2i') 

                    eval_metric = mAP_i2p + mAP_p2i
                    if eval_metric > best_eval_metric:
                        best_eval_metric = eval_metric
                        best_eval_count = 0
                        torch.save(model.state_dict(), 'metric_learning/BEST_checkpoint.pth')
                    else:
                        best_eval_count += 1
                        torch.save(model.state_dict(), 'metric_learning/LAST_checkpoint.pth')

                    if best_eval_count % lr_decay_eval_count == 0 and best_eval_count > 0:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= lr_decay_gamma
                            if use_tensorboard: writer.add_scalar('LR/train', param_group['lr'], it)
                    
                    if use_tensorboard: writer.add_scalar('Improvement/train', eval_metric, it)
            it += 1

    except KeyboardInterrupt:
        writer.close()

def test():
    _, _, test_list = train_val_test_split('datalabels/')
    testset = EvalDataset(test_list)

    model = TripletMatch()
    model.cuda()
    model.eval()

    if os.path.exists('metric_learning/549_BEST_checkpoint.pth'):
        model.load_state_dict(torch.load('metric_learning/549_BEST_checkpoint.pth'), strict=False)

    with torch.no_grad():
        out_img = model.img_encoder(testset.t_images.cuda()).cpu().numpy()
        out_txt = model.lang_encoder(testset.descr).cpu().numpy()
        match_scores = np.zeros((len(testset), len(testset)))
        gt_matrix = np.eye(len(testset))
        for i, img in enumerate(out_img):
            for j, phr in enumerate(out_txt):
                match_scores[i,j] = - np.sum(np.power(img - phr, 2)) # l2_s

        mAP_i2p = compute_mAP(match_scores, gt_matrix, mode='i2p')
        mAP_p2i = compute_mAP(match_scores, gt_matrix, mode='p2i') 

        eval_metric = mAP_i2p + mAP_p2i
        print(f'mAP on test set: {eval_metric:.3f}')

if __name__ == '__main__':
    #train()
    test()