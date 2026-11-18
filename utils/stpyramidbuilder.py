import torch
import torch.nn.functional as F
import pickle

import matplotlib.pyplot as plt

## Object for a "Spatiotemporal Steerable" Pyramid augmented with complex edge images. The pyramids are created by the builders below.  They are created on-demand and thus not persistent or modules. This is the spatiotemporal version of BruceNet's spyramid & spyramidbuilder classes from PooledStatitcsMetamers
class STSPyramid(torch.nn.Module):
    def __init__(self,dummy_movie,lowpasslist,highpasslist,edgereallist,edgeimaglist):
        #super().__init__()
        self.movie = dummy_movie
        self.lowpass = []       # low-pass images for each level (or none if not present in a level)
        # edge images have four channels for orientations  (ie 1x4xHeightxWidth)
        self.edge_real = []     # edge images, real part  (its convolution kernel is odd and its edge response is even)
        self.edge_imag = []     # edge images, imaginary part  (this is not currently used except to construct magn and phase doubled)
        self.edge_magn = []     # edge images, magnitude
        # Now that we have defined the instance variables, let's really initialize them
        # Make a copy of the input lists and pad with None if needed to ensure the required matching lengths
        def _fixlist(a):
            a = list(a)
            #if len(a) < self.max_stop: a.extend([None]*(self.max_stop-len(a)))
            return a
        self.lowpass = _fixlist(lowpasslist)
        self.edge_real = _fixlist(edgereallist)
        self.edge_imag = _fixlist(edgeimaglist)
        
        # Generate derived edge images
        for (re,im) in zip(self.edge_real,self.edge_imag):
            mag = None
            if re is not None:
                # We add a tiny epsilon to ensure argument to sqrt() is >0,  (sqrt(0) produces NaNs in its gradient calculatons and if used as a denominator) 
                mag = torch.sqrt(re*re + im*im + torch.finfo(re.dtype).tiny)
            self.edge_magn.append(mag)

    def original_movie(self):
        return self.movie
        
    def high_pass_movie(self):
        return self.highpass
        
    def lowpass_movies(self):
        return self.lowpass #[level]
    
    def edge_real_movies(self): #real-edge-component has an *odd* convolution kernel and the corresponding edge response in images is *even*
        return self.edge_real
    
    def edge_imag_movies(self): #imaginary-edge-component has an *even* convolution kernel and the corresponding edge response in images is *odd*
        return self.edge_imag
    
    def edge_magnitude_movies(self):
        return self.edge_magn

# def pad_filter(filter, intput):
#     filter_shape = filter.size()
#     input_shape = intput.size()
#     #print(filter_shape, input_shape)
    
#     # Calculate padding
#     pad_depth = (input_shape[2] - filter_shape[2]) // 2
#     pad_height = (input_shape[3] - filter_shape[3]) // 2
#     pad_width = (input_shape[4] - filter_shape[4]) // 2
    
#     # If the difference is odd, add the extra padding to one side
#     pad_front = pad_depth
#     pad_back = pad_depth + (input_shape[2] - filter_shape[2]) % 2
#     pad_top = pad_height
#     pad_bottom = pad_height + (input_shape[3] - filter_shape[3]) % 2
#     pad_left = pad_width
#     pad_right = pad_width + (input_shape[4] - filter_shape[4]) % 2

#     filter_padded = F.pad(filter, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back), "constant", 0)
#     return(filter_padded)


# Object that can build steerable pyramids for images using FFT transforms and fourier-space filters  
class STPyramidBuilder(torch.nn.Module):
    
    # This method needs to know the size of the images so it can prebuild its fourier filters
    def __init__(self, movie, filterbank,
                 filter_location=None,
                device='cpu'):
        super().__init__()
        self.original_movie = movie
        #TODO: Implement high and low pass images
        
        self.filter_size_xy=movie.shape[3]
        self.filter_size_t=movie.shape[2]

        self.filterbank = filterbank
            
        self.filters_edge_real_names = self.filterbank['fourier_pad']['ori']['re'].keys()
        self.filters_edge_imag_names = self.filterbank['fourier_pad']['ori']['im'].keys()
        #print(self.filters_edge_real_names)

        self.filters_edge_real = [torch.real(self.filterbank['fourier_pad']['ori']['re'][k].unsqueeze(0).unsqueeze(0)).to(device) for k in self.filters_edge_real_names]
        self.filters_edge_imag = [torch.real(self.filterbank['fourier_pad']['ori']['im'][k].unsqueeze(0).unsqueeze(0)).to(device) for k in self.filters_edge_imag_names] 
        
        #TODO: Implement magnitude filters
        self.filters_edge_magn = None   
            
        self.filters_lowpass_spatial_real = torch.real(self.filterbank['fourier_pad']['lowpass_spatial']['re'].unsqueeze(0)).to(device)
        self.filters_lowpass_spatial_imag = torch.real(self.filterbank['fourier_pad']['lowpass_spatial']['im'].unsqueeze(0)).to(device)
        self.filters_lowpass_temporal_real = torch.real(self.filterbank['fourier_pad']['lowpass_temporal']['re'].unsqueeze(0)).to(device)
        self.filters_lowpass_temporal_imag = torch.real(self.filterbank['fourier_pad']['lowpass_temporal']['im'].unsqueeze(0)).to(device)
    
        self.filters_highpass = None
    
    
    def build_stpyramid(self, movie, colorname):

        # print('movieprepad')
        # print(movie.shape)
        # plt.imshow(movie[0][0][10].cpu().numpy())
        # plt.colorbar()
        # plt.show()       

        def inverse_ft(filt, dims=(2,3,4)):
            filt = torch.fft.ifftshift(filt, dim=dims)
            filt = torch.fft.ifftn(filt, dim=dims) 
            filt = torch.fft.fftshift(filt, dim=dims)

            #filt = torch.fft.ifftn(torch.fft.ifftshift(filt, dim=dims), dim=dims)# Shasha
            return filt

        def forward_ft(filt, dims=(2,3,4)):
            filt = torch.fft.fftshift(filt, dim=dims)
            filt = torch.fft.fftn(filt, dim=dims) 
            filt = torch.fft.ifftshift(filt, dim=dims)

            #filt = torch.fft.ifftn(torch.fft.ifftshift(filt, dim=dims), dim=dims)# Shasha
            return filt
    
        def get_even_padding(size):
            return size if size % 2 == 0 else size + 1
    
        # Mirror the movie to avoid edge effects
        pad_size = (get_even_padding(movie.shape[-1]//2), get_even_padding(movie.shape[-1]//2),
                    get_even_padding(movie.shape[-2]//2), get_even_padding(movie.shape[-2]//2),
                    get_even_padding(movie.shape[-3]//2), get_even_padding(movie.shape[-3]//2))
    
        # The order for F.pad is (right_d, left_d, right_h, left_h, right_w, left_w)
        movie_padded = F.pad(movie, pad=(pad_size[1], pad_size[0], pad_size[3], pad_size[2], pad_size[5], pad_size[4]), mode='constant')

        # print('paddedmovie')
        # plt.imshow(movie_padded[0][0][10].cpu().numpy())
        # plt.colorbar()
        # plt.show()  
    
        # Fourier transform of the padded movie
        #freq_movie = torch.fft.fftshift(torch.fft.fftn(movie_padded, dim=(2,3,4)), dim=(2,3,4)) #Shasha
        freq_movie = forward_ft(movie_padded)
        # print('freq_movie')
        # print(freq_movie.shape)
        # plt.imshow(torch.real(freq_movie)[0][0][20].cpu().numpy())
        # plt.colorbar()
        # plt.show()
        # Real part of filters
        edgereals = [None] * len(self.filters_edge_real)
        for i, f in enumerate(self.filters_edge_real):
            # print(f.shape)
            # print(freq_movie.shape)
            # print('f')
            # print(f.shape)
            # plt.imshow(torch.real(f)[0][0][20].cpu().numpy())
            # plt.colorbar()
            # plt.show()
            edgereal_padded = torch.real(inverse_ft(freq_movie * f))

            # print('edgereal_padded')
            # print(edgereal_padded.shape)
            # plt.imshow(edgereal_padded[0][0][10].cpu().numpy())
            # plt.colorbar()
            # plt.show()  
            # assert False
            # # Crop the output back to the original movie size
            edgereals[i] = edgereal_padded[:, :, pad_size[-1]:-pad_size[-1], pad_size[-3]:-pad_size[-3], pad_size[-5]:-pad_size[-5]]
        
        # Imaginary part of filters
        edgeimags = [None] * len(self.filters_edge_imag)
        #print('lenimag',len(self.filters_edge_imag))
        for i, f in enumerate(self.filters_edge_imag):
            edgeimag_padded = torch.real(inverse_ft(freq_movie * f))
            # Crop the output back to the original movie size
            edgeimags[i] = edgeimag_padded[:, :, pad_size[-1]:-pad_size[-1], pad_size[-3]:-pad_size[-3], pad_size[-5]:-pad_size[-5]]

        #TODO: Magnitude, lowpass and highpass
        #lowpass = freq_movie * lowpass_filt
        lowpass_spatial_real=torch.real(inverse_ft(freq_movie*self.filters_lowpass_spatial_real))
        lowpass_spatial_imag=torch.real(inverse_ft(freq_movie*self.filters_lowpass_spatial_imag))
        lowpass_temporal_real=torch.real(inverse_ft(freq_movie*self.filters_lowpass_temporal_real))
        lowpass_temporal_imag=torch.real(inverse_ft(freq_movie*self.filters_lowpass_temporal_imag))

        lowpass_spatial_real = lowpass_spatial_real[:, :, pad_size[-1]:-pad_size[-1], pad_size[-3]:-pad_size[-3], pad_size[-5]:-pad_size[-5]]
        lowpass_spatial_imag = lowpass_spatial_imag[:, :, pad_size[-1]:-pad_size[-1], pad_size[-3]:-pad_size[-3], pad_size[-5]:-pad_size[-5]]
        lowpass_temporal_real = lowpass_temporal_real[:, :, pad_size[-1]:-pad_size[-1], pad_size[-3]:-pad_size[-3], pad_size[-5]:-pad_size[-5]]
        lowpass_temporal_imag = lowpass_temporal_imag[:, :, pad_size[-1]:-pad_size[-1], pad_size[-3]:-pad_size[-3], pad_size[-5]:-pad_size[-5]]
    
        lowpass = [lowpass_spatial_real,lowpass_spatial_imag,lowpass_temporal_real,lowpass_temporal_imag]    
    
        highpass=None

        
        # print('filter')
        # print(f.shape)
        # plt.imshow(torch.real(inverse_ft(f))[0][0][10].cpu().numpy())
        # plt.colorbar()
        # plt.show()

        # print('movie')
        # print(freq_movie.shape)
        # plt.imshow(torch.real(inverse_ft(freq_movie))[0][0][10].cpu().numpy())
        # plt.colorbar()
        # plt.show()         
        
        # edgereal_padded = torch.real(inverse_ft(freq_movie * f))

        # print('edgereal_padded')
        # print(edgereal_padded.shape)
        # plt.imshow(edgereal_padded[0][0][10].cpu().numpy())
        # plt.colorbar()
        # plt.show()   

        # edgerealcrop = edgereal_padded[:, :, pad_size[-1]:-pad_size[-1], pad_size[-3]:-pad_size[-3], pad_size[-5]:-pad_size[-5]]

        
        # print('edgereal_crop')
        # print(edgerealcrop.shape)
        # plt.imshow(edgerealcrop[0][0][10].cpu().numpy())
        # plt.colorbar()
        # plt.show()   

        # print('edgereal_padded')
        # print(edgereal_padded.shape)
        # plt.imshow(edgereals[60][0][0][10].cpu().numpy())
        # plt.colorbar()
        # plt.show()   
        
        # print('edgeimag_padded')
        # print(edgeimag_padded.shape)
        # plt.imshow(edgeimags[60][0][0][10].cpu().numpy())
        # plt.colorbar()
        # plt.show()   

        # print('build stpyrmamid filter')
        # print(len(edgereal))
        # plt.imshow(torch.real(inverse_ft(f))[0][0][10].cpu().numpy())
        # plt.colorbar()
        # plt.show()

        # print('build stpyrmamid movie')
        # print(len(edgereal))
        # plt.imshow(torch.real(inverse_ft(freq_movie))[0][0][10].cpu().numpy())
        # plt.colorbar()
        # plt.show()
        
        # print('build stpyrmamid output')
        # print(len(edgereal))
        # plt.imshow(edgereal[i][0][0][10].detach().cpu().numpy())
        # plt.colorbar()
        # plt.show()
        return(STSPyramid(movie,lowpass,highpass,edgereals,edgeimags))
        
    #funtions to return images and pyramids
    def original_movie(self):
        return self.original_movie   

    def high_pass_filter(self):
        return self.filters_highpass

    def low_pass_filters(self):
        return self.filters_lowpass
    
    def edge_real_filters(self):
        return self.filters_edge_real

    def edge_imag_filters(self):
        return self.filters_edge_imag

    def edge_magn_filters(self):
        return self.filters_edge_magn
        