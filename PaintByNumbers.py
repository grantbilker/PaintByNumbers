""" 
Grant Lewis Bilker -20210219

    This project is meant to take any image (preferably higher contrast)
    and turn it into a semi-vectorized template for use as a paint by numbers
    template. The products from this should include: the template image in the
    specified dimensions, the template's color palette (the number specified),
    potentially a configuration panel for absolute color choices (GUI),
    potentially an ability to vectorize as opposed to reimaging, potentially
    the ratios of certain paints to mix each of the colors in the palette

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from time import time
from PIL import Image


def quantize_image(filename, number_of_colors=24):
    
    
    n_colors = number_of_colors

    # Load the image
    original_image = Image.open(filename)

    # Convert to floats instead of the default 8-bit integer coding
    # plt.imshow() requires rgb values [0,1]
    original_image = np.array(original_image, dtype=np.float64) / 255

    # Transform the image array into a 2D numpy array.
    w, h, d = original_shape = tuple(original_image.shape)
    assert d == 3
    image_array = np.reshape(original_image, (w * h, d))
    
    # Grab a shuffled sample of pixels from the image
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    
    # Fit the pixels and generate cluster centers from the sample data
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

    # Get labels for all points
    # Predict color indices on the full image (k-means)
    labels = kmeans.predict(image_array)
    
    
    # Recreate the (compressed) image from the cluster centers & labels
    
    # Create a "blank" image array
    quantized_image_array = np.zeros((w, h, d))
    
    label_idx = 0
    
    for i in range(w):
        for j in range(h):
            
            # Fill image with cluster center label index search
            quantized_image_array[i][j] = kmeans.cluster_centers_[labels[label_idx]]
            
            label_idx += 1
            
    return quantized_image_array



def imsave(image_array, filename):
    
    my_image = Image.fromarray(np.array(image_array*255, dtype=np.uint8), mode='RGB')
    my_image.save(filename)
    
    pass



def color_count(countee):
    
    if type(countee) == str:
        
        # Load the image
        original_image = Image.open(countee)
        
        # Turn image into array
        original_image = np.array(original_image, dtype=np.float64)
    
    else:
        original_image = countee
    
    # Transform the image array into a 2D numpy array.
    w, h, d = original_shape = tuple(original_image.shape)
    assert d == 3
    image_array = np.reshape(original_image, (w * h, d))
    
    cc = set()
    
    for idx in range(w*h):
            
        pixel = tuple(image_array[idx])
        cc.add(pixel)
    
    return len(cc)



def show(image):
    
    plt.figure(1)
    plt.clf()
    plt.axis('off')
    
    if type(image) == str:
    
        imagetype = 'Original'
        image = Image.open(image)
        image = np.array(image, dtype=np.float64) / 255
        
    else:
        imagetype = 'Quantized'
    
    plt.title(imagetype + ' image w/ ' + str(color_count(image)) + ' colors')
    plt.imshow(image)
    
    pass