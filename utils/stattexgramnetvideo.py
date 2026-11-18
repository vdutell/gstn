import torch
import torch.nn as nn
#import utils.BNPyramids as bnp
import utils.argmax_diff as amxd
import re
import numpy as np

import utils.stpyramidbuilder as stpb
import utils.stattexpyramid as stp

import matplotlib.pyplot as plt

#sys.path.append('../PooledStatisticsMetamers/poolstatmetamer/')

#import poolingregions as pyrpool
#import metamersolver as pyrsolver
#from .stpyramid_builder import stpyramid_builder as stp


from itertools import combinations_with_replacement

from torchvision.transforms.functional import normalize
#from deeptextures.utils import MEAN, STD

class StatTexGramNetVideo(nn.Module):
    def __init__(self, movie_size, 
                 scales = [2,4,8,16],
                 oris = [0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4],
                 heights = [2,4,8],
                 batch_size=100, 
                 bottleneck_size=100, 
                 device=torch.device('cuda:0'),
                 compress=True,
                 color=False,
                 add_marginals=True,
                 hist_matching=False,
                 hist_bins=50,
                 hist_weight=10000,
                 filter_location=None,
                 use_pooling=False,
                 pooling_params = None
                ):
        super(StatTexGramNetVideo, self).__init__()
        
        #self.onehot = onehot #use a hard onehot constraint
        #self.entropy = entropy #use entropy instead of L1 weight loss
        #image params
        self.hist_weight = hist_weight
        self.compress = compress
        self.insize_y, self.insize_x, self.insize_t = movie_size        
        self.batch_size = batch_size
        self.bottleneck_size = bottleneck_size
        self.device = device
        self.add_marginals = add_marginals
        self.color = color
        self.oris = oris
        self.heights = heights
        self.scales = scales
        self.hist_matching = hist_matching
        self.num_hist_bins = hist_bins
        if(self.color):
            self.color_channels = 3
        else:
            self.color_channels = 1 #color channels
            
        self.dummy_mov = torch.zeros(self.batch_size,
                                     self.color_channels,
                                     self.insize_t,
                                     self.insize_y,
                                     self.insize_x).to(device).contiguous()
        
        #Cacluate number of pyramids
        #nedge_pyrs = 3 # 1 each for real, imag, and mag
        #npyrs = 1 + len(params.lowpass_range()) + oris*nedge_pyrs*len(params.edge_range())
        print(len(scales),len(oris),len(heights))
        npyrs = len(scales) * len(oris) * len(heights) * 2 + 4 #4 lowpass filters 
        print(f'There are {npyrs} pyramids per color channel')
        self.marginal_names = ['mean','var','std']
        self.marginal_size = len(self.marginal_names)

        #Spatiotemporal Pyramid Encodder Layer
        #if color image
        if(self.color):
            self.stpyr = stp.StatTexPyramidColor(dummy_movie=self.dummy_mov,filter_location=filter_location,device=device).to(self.device)
            self.npyrs = npyrs*3 #23 before adding imag and real images
            self.gram_size = int((self.npyrs*(self.npyrs+1))//2)
            # self.marginal_names = [m+'_C'+c for c in ['1','2','3'] for m in self.marginal_names]
            # self.marginal_size = self.marginal_size*3 #one per color channel
        else:
            self.stpyr = stp.StatTexPyramid(dummy_movie=self.dummy_mov,filter_location=filter_location,device=device).to(device)
            self.npyrs = npyrs #23 before adding imag and real images
        self.gram_size = int((self.npyrs*(self.npyrs+1))//2)
         
        #Statistics Labels for Gram Matrix
        #ordering of gram matrix combinations
        # self.stat_names = self.bruceNet.stat_names #this is going ot 
        # self.stat_names_gram_label = list(combinations_with_replacement(self.bruceNet.stat_names,r=2))
        # self.stat_names_gram_label = [pair[0]+','+pair[1] for pair in self.stat_names_gram_label]
                
        # #if including marginals and/or powerspectrum
        self.gram_size = int((self.npyrs*(self.npyrs+1))//2)
        self.stats_size = self.gram_size #should be equal to length of stat names gram label
        if(self.add_marginals):
            self.stats_size = self.stats_size + self.marginal_size
            #self.stat_names_gram_label = self.stat_names_gram_label + self.marginal_names
        # if(self.add_pspec):
        #     self.stats_size = self.stats_size + self.pspec_size
        #     self.stat_names_gram_label = self.stat_names_gram_label + self.pspec_names           

        # #statisitcs labels
        # self.stat_groups_gram_label = self.assign_gram_groups(self.stat_names_gram_label)
        
        #pyramid_encoder
        self.pyr_encoder = nn.Sequential(
            self.stpyr
        )
        if(self.compress):
            print(f'Compressing from {self.stats_size} to {self.bottleneck_size}')
            self._w = nn.Parameter(torch.rand((self.stats_size, self.bottleneck_size), device=self.device)).contiguous()
            #self._w.requires_grad(True) #set weights to use gradient
        self.triu_idxes = torch.triu(torch.ones((self.npyrs,self.npyrs)),diagonal=0).to(bool)

    def calcgramm(self, tnsr: torch.Tensor, triu_idxes: torch.Tensor) -> torch.Tensor:
        ###adapted from pytorch implementation of Leon Gatys Deep Textures
        #https://github.com/trsvchn/deep-textures
        #print('calculating gramm')
        
        """Computes Gram matrix for the input batch tensor.

        Args:
            tnsr (torch.Tensor): input tensor of the Size([B, C, T, H, W]).

        Returns:
            G (torch.Tensor): output tensor of the Size([B, C, C]).
        """
        #print(tnsr.shape)
        N = tnsr.size(1)  # Number of filters (size of feature maps) = C
        M = tnsr.size(-3) * tnsr.size(-2) * tnsr.size(-1) # size of the feature maps = T * H * W
        # Feature map: BCTHW->BNM (formerly B, C, F, H, W ->  B, N, M)
        #print(N,M)
        f = tnsr.view(tnsr.size(0), N, M)  # Size([B, N, M])
        #print('f',f.shape)
        # Gram matrix: B, N, M -> B, N, N
        #Try normalzing before calculation
        f = f-f.mean(dim=(1,2),keepdim=True)
        f = f / f.std(dim=(1,2),keepdim=True)
        #print('f',f.shape)
        G = torch.bmm(f, f.transpose(1, 2))  # Size([B, N, N])
        #print('g',G.shape)
        #print(triu_idxes.shape)
        
        #try normalizing the gramm 
        # print(torch.mean(G))
        # print(torch.std(G))
        #G = G / G.abs().sum(dim=1, keepdim=True)
        G = G /  torch.mean(abs(G))

       #print(G.shape)
        #print(triu_idxes.shape)
        

        #VD Remove for now for debugging
        #G = G.flatten(start_dim=1)
        G = G[:,triu_idxes]
        # plt.hist(G.cpu().detach().flatten(),bins=100)
        # plt.title('G')
        # plt.show()
        #print('G',G.shape)

        return(G)

#     def calchist(self, input_tensor, num_bins=50, min_val=0, max_val=1.0):
#         """
#         Compute a differentiable Kernel Density Estimate (KDE) with batch dimension support.
        
#         Args:
#             input_tensor (torch.Tensor): Tensor of shape (batch_size, num_points) or (batch_size, channels, frames, height, width)
#             num_bins (int): Number of histogram bins.
#             min_val (float): Minimum value of the histogram range.
#             max_val (float): Maximum value of the histogram range.
#             bandwidth (float): Bandwidth (standard deviation) of the Gaussian kernel for KDE.
        
#         Returns:
#             kde_histogram (torch.Tensor): Soft histogram (KDE-based) of shape (batch_size, num_bins)
#         """

#         #bandwidth = 1.0/(num_bins*0.5)
#         bandwidth = 0.9 * input_tensor.std() * input_tensor.numel() ** (-1 / 5)
# # Get the batch size and flatten everything else into one dimension
#         batch_size = input_tensor.shape[0]
#         input_tensor = input_tensor.view(batch_size, -1)  # Shape (batch_size, num_points)
    
#         # Generate bin centers based on num_bins, evenly spaced between min_val and max_val
#         bin_centers = torch.linspace(min_val, max_val, steps=num_bins, device=input_tensor.device)  # Shape (num_bins,)
#         bin_centers = bin_centers.view(1, 1, -1)  # Reshape for broadcasting, (1, 1, num_bins)
    
#         # Reshape input_tensor to (batch_size, num_points, 1) to compute KDE
#         input_tensor = input_tensor.unsqueeze(-1)  # Shape (batch_size, num_points, 1)
    
#         # Compute the Gaussian kernel for each input point with respect to each bin center
#         # Gaussian kernel formula: exp(-((x - mu)^2) / (2 * bandwidth^2))
#         diff = input_tensor - bin_centers  # Shape (batch_size, num_points, num_bins)
#         kernel_values = torch.exp(-0.5 * (diff / bandwidth) ** 2)  # ShaSha Shape (batch_size, num_points, num_bins)
#         #kernel_values = 1/(np.pi * bandwidth) * 1/(1 + (diff / bandwidth)**2)
#         # Sum over all input points to compute the KDE for each batch
#         kde_histogram = kernel_values.sum(dim=1)  # Shape (batch_size, num_bins)
    
#         # Normalize the KDE so that it sums to 1 per batch
#         kde_histogram = kde_histogram / kde_histogram.sum(dim=-1, keepdim=True)
#         #kde_histogram = kde_histogram / torch.max(kde_histogram)
#         #kde_histogram = kde_histogram / num_bins
#         # Add a small offset so we have a gradient
#         kde_histogram = kde_histogram + 1e-8
#         # print(kde_histogram)
#         # plt.plot(kde_histogram.cpu().detach().numpy())
#         # plt.show()
#         #print(kde_histogram)


#         # # Plotting
#         # kde_histogram_np = kde_histogram.cpu().detach().numpy().squeeze()  # Convert to NumPy for plotting
#         # bin_centers = torch.linspace(0, 1, steps=num_bins).cpu().numpy()  # Same bin centers used in calchist
        
#         # plt.figure(figsize=(8, 6))
#         # plt.plot(bin_centers, kde_histogram_np, label="KDE Histogram")
#         # plt.xlabel('Value')
#         # plt.ylabel('Density')
#         # plt.title('KDE-Based Histogram')
#         # plt.legend()
#         # plt.show()
    
#         return kde_histogram

    
    # def calchist(self, t: torch.Tensor) -> torch.Tensor:
    #     def singlechannelhist(t):
    #         hist = torch.histc(t,bins=100).unsqueeze(0)
    #         hist = hist / torch.amax(hist,1,keepdim=True)
    #         plt.hist(hist.cpu().detach().numpy().flatten())
    #         plt.show()
    #         # print('mmm',t.shape, torch.min(t),torch.max(t))
    #         #hist = torch.cat([torch.histc(x, min=-1, max=1, bins=100).view(1,-1) for x in t],dim=0)
    #         #print(torch.amax(hist,1,keepdim=True).shape)
    #         #hist = hist/torch.amax(hist,1,keepdim=True)
    #         #print(hist.shape)
    #         return(hist)
    #     if(self.color):
    #         hist = torch.cat([singlechannelhist(t.narrow(1,0,1)),
    #                           singlechannelhist(t.narrow(1,1,1)),
    #                           singlechannelhist(t.narrow(1,2,1))],dim=1)
    #     else:
    #         hist = singlechannelhist(t)
    #     return(hist)
    
    def calcmarginals(self, rawim: torch.Tensor) -> torch.Tensor:
        '''
        include mean, var,std, skew kurtosis
        '''
        def singlechannel(im):
            mean = im.view(im.size(0), -1).mean(1) #mean but skip batch dim
            mean = mean[:, None, None, None]#expand_dims
            diffs = im - mean
            var = torch.pow(diffs, 2.0)
            var = var.view(var.size(0),-1).mean(1)
            var = var[:, None, None, None] #expand_dims
            std = torch.pow(var, 0.5)
            return(torch.cat([mean,var,std],dim=1).squeeze(dim=(-2,-1)))
            
        if(self.color):
            #ORDER IS: [mean_c1,var_c1,std_c1,mean_c2,var_c2,std_c2,mean_c3,var_c3,std_c3]
            #print('rawim',rawim.shape)
            marginals = torch.cat([singlechannel(rawim.narrow(1,0,1)),
                                   singlechannel(rawim.narrow(1,1,1)),
                                   singlechannel(rawim.narrow(1,2,1))],dim=1)
            #print('marg',marginals.shape)
        else:
            marginals = singlechannel(rawim)
            
        return(marginals)

    

    def check_same_type(self,label,stattype):
        '''
        Check stat is of a consisten type (orietation, color channel, level, r/i/m) by checking if stat appears twice in the label
        stattype is one of: 'level','ori', 'pyr', 'color'
        '''
    
        if(stattype == 'level'):
            statvals = ['L0','L1','L2','L3','L4','L5','L6','L7','L8','L9'] #assume no more than 10 levels (using 6 right now)
        elif(stattype == 'ori'):
            statvals = ['O0','O1','O2','O3','O4','O5','O6','O7'] #assume no more than 8 orientations (using 4 now)
        elif(stattype == 'color'):
            statvals = ['C1','C2','C3'] #assume no more than 3 color channels
        elif(stattype == 'pyr'):
            statvals = ['real','imag','magn'] #assume no more than 3 color channels
        
        if(any([label.count(statstring)==2 for statstring in statvals])):
            return(True)
        else:
            return(False)

    def assign_gram_groups(self, gramlabels):
        '''
        Start by labeling only stats that are orientated pyramids crosssed with other oriented pyramids that vary in a single dimension (level,orientation,real/imag/mag,color). 
        Name other oriented pyramids that vary in multiple dimensions ori_Xmulti, and anthing crossed with a non-oriented pyramid as highpass,lowpass.
        Marginals are not crossed with anything
    
        TODO: Label cross-scale oriented pyramids as ori_Xscale_d1 if they are neighboring scales.
        TODO: Add power spectrum stats labels
        '''
        
        #labels:
        #'highpass','lowpass','pass_multi','marginal',
        #'ori_Xcolor','ori_Xori','ori_Xlevel','ori_Xri','ori_Xrm','ori_Xim','ori_Xmulti'
        gram_groups = []
        #assign each label to a group
        for gramstatlabel in gramlabels:
            #check for high/low/band pass
            if('highpass' in gramstatlabel):
                #see if it's only highpass, or if it's a highpass cross
                if(gramstatlabel.count('highpass')==2):
                    gram_groups.append('highpass')
                else:
                    gram_groups.append('pass_multi')
            elif('lowpass' in gramstatlabel):
                #see if it's only lowpass, or if it's a lowpass cross
                if(gramstatlabel.count('lowpass')==2):
                    gram_groups.append('lowpass')
                else:
                    gram_groups.append('pass_multi')
            #check marginals *note this will only label marginals x marginals
            elif(any(x in gramstatlabel for x in ['mean','var','std'])):
                gram_groups.append('marginal')

            #check for power spectrum
            elif(any(x in gramstatlabel for x in ['PS_'])):
                gram_groups.append('pspec')
            
            else:
                #now check oriented pyramids.
                #first double check these are only oriented pyaramids 
                assert any(x in gramstatlabel for x in ['real','imag','magn']), f'GramStatLabel {gramstatlabel} should be real, imag, or magn pyramid!'
                #first check for cross color (level, orientation, and r/im/mag must be consistent)
                if(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'pyr')])):
                    gram_groups.append('sub_Xcolor')
                #check for cross orientation (level, color, and r/im/mag must be consistent)
                elif(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'color'),self.check_same_type(gramstatlabel,'pyr')])):
                    gram_groups.append('sub_Xori')
                #check for cross level (orientation, color, and r/im/mag must be consistent)
                #TODO: Add orientation cross neighbor support (if levels are only one level apart) 'ori_Xlevel_neighbor'
                elif(all([self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'color'),self.check_same_type(gramstatlabel,'pyr')])):
                    gram_groups.append('sub_Xlevel')
                #REAL X REAL
                elif(gramstatlabel.count('real')==2):
                    #label these all realx multi for now
                    gram_groups.append('sub_real_Xmulti')
                #IMAG X IMAG                        
                elif(gramstatlabel.count('imag')==2):
                    gram_groups.append('sub_imag_Xmulti')
                #MAGN X MAGN
                elif(gramstatlabel.count('magn')==2):
                    gram_groups.append('sub_magn_Xmulti')

                #first cross real/imag
                elif('real' in gramstatlabel and 'imag' in gramstatlabel):
                    if(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'color')])):
                        gram_groups.append('sub_Xri_eq')
                    # elif(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'color')])):
                    #     gram_groups.append('sub_Xri_Xori')
                    # elif(all([self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'color')])):
                    #     gram_groups.append('sub_Xri_Xlevel')
                    # elif(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'ori')])):
                    #     gram_groups.append('sub_Xri_Xcolor')
                    else:
                        gram_groups.append('sub_Xmulti')
                    #gram_groups.append('sub_Xri_Xmulti')
                #next cross real/mag
                elif('real' in gramstatlabel and 'magn' in gramstatlabel):
                    # #print('real','mag: ',gramstatlabel)
                    if(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'color')])):
                        gram_groups.append('sub_Xrm_eq')
                    # elif(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'color')])):
                    #     gram_groups.append('sub_Xrm_Xori')
                    #     #print('real,mag,cross_ori',gramstatlabel)
                    # elif(all([self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'color')])):
                    #     gram_groups.append('sub_Xrm_Xlevel')
                    # elif(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'ori')])):
                    #     gram_groups.append('sub_Xrm_Xcolor')
                    else:
                        gram_groups.append('sub_Xmulti')
                    #gram_groups.append('sub_Xrm_Xmulti')
                #next cross imag/mag
                elif('imag' in gramstatlabel and 'magn' in gramstatlabel):
                    if(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'color')])):
                        gram_groups.append('sub_Xim_eq')
                    # elif(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'color')])):
                    #     gram_groups.append('sub_Xim_Xori')
                    # elif(all([self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'color')])):
                    #     gram_groups.append('sub_Xim_Xlevel')
                    # elif(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'ori')])):
                    #     gram_groups.append('sub_Xim_Xcolor')
                    else:
                        gram_groups.append('sub_Xmulti')
                        #gram_groups.append('sub_Xim_Xmulti')
                #otherwise, assume we're crossing mulitple stat types
                else:
                    gram_groups.append('garbagecan')
        return(gram_groups)

    def normalize_tensor(self, tensor):
        #tensor = tensor + 1
        #return(tensor)
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val)


    #encoder creates pyramid and calculates gramm matrix
    def encoder(self,vid):    
        #vid = self.normalize_tensor(vid)
        #use a tanh because filters are centered around 0.
        #vid = torch.clip(vid, min=0,max=1)
        #vid_norm = torch.nn.functional.sigmoid(vid)
        #vid = torch.nn.functional.sigmoid(vid)
        #vid = 0.5 + (vid-0.5)**3
        #vid_norm = torch.nn.functional.tanh(vid)
        #vid_norm = vid - torch.mean(vid)
        x = self.pyr_encoder(vid)
        #print('x before gramm',x.shape)
        x = self.calcgramm(x,triu_idxes=self.triu_idxes)
        #print('gramm',x.shape)
        #x = torch.zeros((1,1),device=self.device)
        #x = torch.ones_like(x)
        #print(x.shape)
        #if(self.add_marginals):
            #print(x.shape,self.calcmarginals(vid).shape)
        #    x = torch.cat([x,self.calcmarginals(vid)],dim=1)
        #if(self.hist_matching):
            #print('histmatching')
            #print(x.shape,self.calchist(vid).shape)
            #normalize like will be for image [0-1]
            #vid = (vid - torch.min(vid)) / (torch.max(vid) - torch.min(vid) + 1e-7)
            #plt.hist(vid.cpu().detach().numpy().flatten())
            #plt.show()
            #x = torch.cat([x,self.hist_weight*self.calchist(vid,num_bins=self.num_hist_bins)],dim=1)
        return(x)

    #compressor reduces dimensionality (learning)
    def compressor(self,x):
        self.w = self._w
        x = torch.matmul(x,self.w)
        return(x)
    
    #combine encoder and compressor
    def forward(self,x):
        #print(x.shape)
        x = self.encoder(x)
        # if(self.compress):
        #     x = self.compressor(x)
        return(x)


# The paper calls this model GSTN / GramStatTexNetVideo. Alias it under that
# name so either spelling works; StatTexGramNetVideo remains the definition.
GramStatTexNetVideo = StatTexGramNetVideo
