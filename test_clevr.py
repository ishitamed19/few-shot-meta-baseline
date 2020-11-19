import argparse
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
from tqdm import tqdm
from torch.utils.data import DataLoader
import random
import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler

class DoublePool_O():
    def __init__(self, pool_size):
        self.pool_size = pool_size
   
        random.seed(125)
        if self.pool_size > 0:
            self.num = 0
            self.embeds = []

    def fetch(self):
        return self.embeds, self.images
            
    def is_full(self):
        full = self.num==self.pool_size
        # print 'num = %d; full = %s' % (self.num, full)
        return full
            
    def update(self, embeds, images):
        # embeds is B x ... x C
        # images is B x ... x 3
        assert embeds.shape[0]==images.shape[0]

        for embed, image in zip(embeds, images):
            if self.num < self.pool_size:
                # the pool is not full, so let's add this in
                self.num = self.num + 1
            else:
                # the pool is full
                # pop from the front
                self.embeds.pop(0)
                self.images.pop(0)

            # add to the back
            self.embeds.append(embed)
            self.images.append(image)



def plot_query_retrieval(imgs_retrieval):
#     st()
    n_retrieval = len(imgs_retrieval)
    fig = plt.figure(figsize=(20, 4))
    for idx in range(n_retrieval):
        for im in range(0, 2):
            ax = fig.add_subplot(n_retrieval, 2, 2*idx+im+1,xticks=[], yticks=[])
            if im==0: 
                ax.set_title('Query')
            else:
                ax.set_title('Top_'+str(im))
            ax.imshow(imgs_retrieval[idx][im].permute(1,2,0).detach().cpu().numpy())
    plt.close(fig)
    return fig
    
    
model = models.load(torch.load('/home/mprabhud/ishita/few-shot-meta-baseline/im800-resnet50.pth?dl=0'))


model.eval()
utils.log('num params: {}'.format(utils.compute_n_params(model)))


##################################################
dataset = # INSERT DATASET HERE
##################################################

loader = DataLoader(dataset,num_workers=8, pin_memory=True)

pool_e = DoublePool_O(pool_size=1000)

test_epochs = 1
np.random.seed(0)

for epoch in range(1, test_epochs + 1):
    for iter_num, inp_imgs in enumerate(loader):
        
        # inp_images: Bx3xHxW
        
        with torch.no_grad():
            img_shape = data.shape[-3:]
            inp_embeds = model.encoder(data.view(-1, *img_shape)) #.mean(dim=1, keepdim=True)
            #p = F.normalize(feats, dim=-1)
            
            if iter_num % 10 == 0 and pool_e.pool_size==1000:
                
                B = inp_embeds.shape[0]
                feat_dim = inp_embeds.shape[1]

                imgs_to_vis = []
                
                for i in range(B):
                    pool_embeds, pool_imgs = pool_e.fetch()
                    pool_embeds = torch.from_numpy(np.asarray(pool_embeds.detach().cpu().numpy()))
                    query_embed = inp_embeds[i].reshape(feat_dim, 1)
                    dot_prod = torch.mm(pool_embeds, query_embed)
                    top1_index = torch.argmax(dot_prod)
                    imgs_to_vis.append([inp_imgs[i], pool_imgs[top1_index]])
                    
                fig_to_plot = plot_query_retrieval(imgs_to_vis)
            
                tb_logger.add_figure('Top1 Retrieval', fig_to_plot, iter_num)

            pool_e.update(inp_embeds, inp_imgs)
            
            
            
    