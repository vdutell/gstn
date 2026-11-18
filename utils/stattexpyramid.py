import utils.stpyramidbuilder as stpyr
import utils.color_utils as coloru
import pickle
import torch
import torch.nn as nn

class StatTexPyramid(nn.Module):
    'Pyramid Analysis for Videos. Analagous to BNPyramids, but self-implemented and for 3D input'
    def __init__(self, dummy_movie, filter_location=None, filterbank = None, device='cpu'):
        super().__init__()
        #self.params = SPyramidParams(4,boundary_mode='wrap')
        self.filter_names = ['edge_real',
                             'edge_imag',
                             'low_pass']
                             #[#'low_pass_image',
                             #'edge_magnitude_images',
                             #'high_pass_image']
        self.filter_location = filter_location
        self.filter_size_xy = dummy_movie.shape[3]
        self.filter_size_t = dummy_movie.shape[2]

        # load in the filters here instaed of in stpyramidbuilder so we don't have to load them in once per color channel
        def _load_filterbank(filter_location):
            #acces filter dictionary with: filters['spatiotemporal']['re']['1-0-1']
            with open(filter_location, 'rb') as f:
                return(pickle.load(f))
                
        if filter_location == None:
            filter_location = f'./filter_banks_normFalse_{self.filter_size_xy}px_{self.filter_size_t}fr/filters_torch.pkl'
        if filterbank == None:
            self.filterbank = _load_filterbank(filter_location)
            print('loaded filters at: ',filter_location)
        else:
            self.filterbank = filterbank
            


        self.builder = stpyr.STPyramidBuilder(dummy_movie, self.filterbank, device=device, filter_location=filter_location)
        #self.params = pyramid_params
    
        #TODO
        # self.stat_names = [ff'lowpass_{i}' for i in self.params.lowpass_range()] + \
        #                     [f'real_L{i}_O{l}' for i in self.params.edge_range() for l in range(self.params.orientations)] + \
        #                     [f'imag_L{i}_O{l}' for i in self.params.edge_range() for l in range(self.params.orientations)] + \
        #                     [f'magn_L{i}_O{l}' for i in self.params.edge_range() for l in range(self.params.orientations)] + \
        #                     [f'highpass']        
    
    def vectorize_pyramid(self, stpyramid):
        '''
        go through images and vectorize them all
        '''

        #print(stpyramid.edge_real_movies()[0].shape)
        #print('pyrs',torch.cat(stpyramid.edge_real_movies(),dim=1).shape)
        
        #create a stack treating pyramid images as channels
        return(torch.cat([torch.cat(stpyramid.edge_real_movies(),dim=1),
                          torch.cat(stpyramid.edge_imag_movies(),dim=1),
                          torch.cat(stpyramid.lowpass_movies(),dim=1)],dim=1))
    
        #return(torch.cat([
            #stpyramid.high_pass_image(),
            #torch.cat([stpyramid.low_pass_image(i) for i in stpyramid.params().lowpass_range()],dim=1),
            #torch.cat([stpyramid.edge_real_images(i) for i in stpyramid.params().edge_range()],dim=1),
            #torch.cat([stpyramid.edge_imag_images(i) for i in stpyramid.params().edge_range()],dim=1),
            #torch.cat([stpyramid.edge_magnitude_images(i) for i in stpyramid.params().edge_range()],dim=1)
        #    ]))

    def forward(self, x):
        x = self.builder.build_stpyramid(x,colorname='grey')
        x = self.vectorize_pyramid(x)
        return(x)


class StatTexPyramidColor(nn.Module):
    'wrapper for stattexpyramid to support color'
    def __init__(self, dummy_movie,filter_location,device='cpu'):
        super().__init__()

        self.colorspace = coloru.RGBToOpponentConeTransform()
        self.channel_names = self.colorspace.channel_names()
        self.filter_size_xy = dummy_movie.shape[3]
        self.filter_size_t = dummy_movie.shape[2]
        
        self.filter_location = filter_location
        # load in the filters here instaed of in stpyramidbuilder so we don't have to load them in once per color channel
        def _load_filterbank(filter_location):
            #acces filter dictionary with: filters['spatiotemporal']['re']['1-0-1']
            with open(filter_location, 'rb') as f:
                return(pickle.load(f))
        if filter_location == None:
            filter_location = f'./filter_banks_normFalse_{self.filter_size_xy}px_{self.filter_size_t}fr/filters_torch.pkl'
        self.filterbank = _load_filterbank(filter_location)
        
        #leave out cross color stats for now - individual channels only
        #we will get cross color stats from gramm matrix calculation downstream
        self.builder_1 = stpyr.STPyramidBuilder(dummy_movie[:,0,:,:,:],filterbank=self.filterbank,device=device, filter_location=filter_location)
        self.builder_2 = stpyr.STPyramidBuilder(dummy_movie[:,0,:,:,:],filterbank=self.filterbank,device=device, filter_location=filter_location)
        self.builder_3 = stpyr.STPyramidBuilder(dummy_movie[:,0,:,:,:],filterbank=self.filterbank,device=device, filter_location=filter_location)
        #self.params = pyramid_params
    
        #TODO
        #self.stat_names = [s+'_C1' for s in self.single_channel_stat_names] + \
        #                   [s+'_C2' for s in self.single_channel_stat_names] + \
        #                   [s+'_C3' for s in self.single_channel_stat_names]
    
    def vectorize_pyramid(self, stpyramid):
        '''
        go through images and vectorize them all
        '''
        # print('pyrshapes')
        # print((torch.cat(stpyramid.edge_real_movies(),dim=1)).shape)
        # print((torch.cat(stpyramid.edge_imag_movies(),dim=1)).shape)
        # print((torch.cat(stpyramid.lowpass_movies(),dim=1)).shape)
        
        #print(stpyramid.edge_real_movies()[0].shape)
        
        #create a stack treating pyramid images as channels
        return(torch.cat([torch.cat(stpyramid.edge_real_movies(),dim=1),
                          torch.cat(stpyramid.edge_imag_movies(),dim=1),
                          torch.cat(stpyramid.lowpass_movies(),dim=1)],dim=1))
    

    def forward(self, x):
        #print('x in fw',x.shape)
        #x = torch.stack([self.colorspace(x[0,:,i,:,:]) for i in range(x.shape[2])],dim=2)
        #print(x.shape)
        #print(x.squeeze(0).permute(1,0,2,3).shape)
        #print(x.squeeze(0).permute(1,0,2,3).permute(1,0,2,3).unsqueeze(0).shape)
        #x = self.colorspace(x.squeeze(0).permute(1,0,2,3)).permute(1,0,2,3).unsqueeze(0) #treat the frames as the batch dim
        #x = torch.einsum('bcfxy -> bcyx', self.colorspace(x))
        x = torch.cat([self.vectorize_pyramid(self.builder_1.build_stpyramid(x.narrow(1,0,1),colorname=self.channel_names[0])),
                       self.vectorize_pyramid(self.builder_2.build_stpyramid(x.narrow(1,1,1),colorname=self.channel_names[1])),
                       self.vectorize_pyramid(self.builder_3.build_stpyramid(x.narrow(1,2,1),colorname=self.channel_names[2]))],
                       dim=1)
        #print('postvector',x.shape)
        return(x)
