import pandas as pd
import numpy as np
from skimage import io, filters
from skimage.color import rgb2gray
from sklearn.preprocessing import normalize
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)

train_df = pd.read_json('train.json')
print('.json lido para dataframe pandas')

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75)
					for band in train_df["band_1"]])
						
image = x_band1[1]
print (len(x_band1))
'''filtered_images = []
for band in x_band1:
	filtered_images.append(filters.gaussian(band, preserve_range=True))
print (len(filtered_images))
for i in range (9):
	io.imshow(filtered_images[i])
	io.show()

'''
io.imshow(image)
io.show()

normalized_image = normalize(image)
io.imshow(normalized_image)
io.show()

denoised_image = denoise_tv_chambolle(image, weight=0.1, multichannel=True)
io.imshow(denoised_image)
io.show()

gaussian_image = filters.gaussian(image, preserve_range=True)
io.imshow(gaussian_image)
io.show()
'''
gaussian_image = filters.gaussian(normalized_image)
io.imshow(gaussian_image)
io.show()'''

gray_gaussian = rgb2gray(gaussian_image)
io.imshow(gray_gaussian)
io.show()

threshold = filters.threshold_otsu(gaussian_image)
mask = gaussian_image < threshold
io.imshow(mask)
io.show()

thresholded_gaussian = filters.threshold_sauvola(gray_gaussian)
io.imshow(thresholded_gaussian)
io.show()

median_image = filters.gaussian(normalized_image)
io.imshow(median_image)
io.show()
