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
#from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
#from time import time
from PIL import Image
import scipy.ndimage as ndimage 



class PixelGroup():
    
    
    def __init__(self, uid, color_array):
        
        self.id = uid
        self.color = color_array
        self.members = []
        self.neighbors = set()
        pass
    
    
    def add_member(self, i, j):
        
        self.members.append((i,j))
        pass
    
    
    def size(self):
        
        return len(self.members)
    
    
    def calculate_neighbors(self, id_array):
        
        # Neighbors will be defined as above, below, left, and right
        # This allows for more accurate absorption
        
        for member in self.members:
            
            try:
                above = id_array[member[0]+1][member[1]]
                if above != self.id:
                    self.neighbors.add(above)
                    continue
            except:
                pass
            try:
                left = id_array[member[0]][member[1]-1]
                if left != self.id:
                    self.neighbors.add(left)
                    continue
            except:
                pass
            try:
                right = id_array[member[0]][member[1]+1]
                if right != self.id:
                    self.neighbors.add(right)
                    continue
            except:
                pass
            try:
                below = id_array[member[0]-1][member[1]]
                if below != self.id:
                    self.neighbors.add(below)
                    continue
            except:
                pass
            
            
    def absorb(self, pgroup):
        
        self.members.extend(pgroup.members)
        pass
    
    
    



def quantize_image(filename, number_of_colors=24):
    
    
    n_colors = number_of_colors

    # Load the image
    original_image = Image.open(filename)

    # Convert to floats instead of the default 8-bit integer coding
    # plt.imshow() requires rgb values [0,1]
    original_image = np.array(original_image, dtype=np.float64) / 255

    # Transform the image array into a 2D numpy array.
    h, w, d = tuple(original_image.shape)
    assert d == 3
    image_array = np.reshape(original_image, (h * w, d))
    
    # Grab a shuffled sample of pixels from the image
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    
    # Fit the pixels and generate cluster centers from the sample data
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

    # Get labels for all points
    # Predict color indices on the full image (k-means)
    labels = kmeans.predict(image_array)
    
    
    # Recreate the (compressed) image from the cluster centers & labels
    
    # Create a "blank" image array
    quantized_image_array = np.zeros((h, w, d))
    
    label_idx = 0
    
    for i in range(h):
        for j in range(w):
            
            # Fill image with cluster center label index search
            quantized_image_array[i][j] = kmeans.cluster_centers_[labels[label_idx]]
            
            label_idx += 1
            
    return quantized_image_array



def gaussian_blur(image_array, sigma=3):
    
    return ndimage.gaussian_filter(image_array, (int(sigma),int(sigma),0))



def requantize_image_array(blurred_image_array, number_of_colors=24):
    
    n_colors = number_of_colors
    
    # Transform the image array into a 2D numpy array.
    h, w, d = tuple(blurred_image_array.shape)
    assert d == 3
    image_array = np.reshape(blurred_image_array, (h * w, d))
    
    # Grab a shuffled sample of pixels from the image
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    
    # Fit the pixels and generate cluster centers from the sample data
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

    # Get labels for all points
    # Predict color indices on the full image (k-means)
    labels = kmeans.predict(image_array)
    
    
    # Recreate the (compressed) image from the cluster centers & labels
    
    # Create a "blank" image array (one is white)
    requantized_image_array = np.ones((h, w, d))
    
    label_idx = 0
    
    for i in range(h):
        for j in range(w):
            
            # Fill image with cluster center label index search
            requantized_image_array[i][j] = kmeans.cluster_centers_[labels[label_idx]]
            
            label_idx += 1
            
    return requantized_image_array



def trace(image_array):
    
    # Get the image_array's dimension
    h, w, d = tuple(image_array.shape)
    
    # Create new "blank" image array that is the correct dimension
    trace_array = np.ones((h, w, d), dtype=np.float64)
    
    # Create black pixel array
    black = np.array([0, 0, 0], dtype=np.float64)
    
    # Color the top and bottom edges black
    for i in range(w):
        trace_array[0][i] = black
        trace_array[h-1][i] = black
    
    # Color the left and right edges black
    for j in range(h):
        trace_array[j][0] = black
        trace_array[j][w-1] = black
    
    # Traverse the image_array's height
    for i in range(1,h-1):
        # Traverse the image_array's width
        for j in range(1,w-1):
            
            current_pixel = image_array[i][j]
            
            # 8-point check for different colors
            # X is the current pixel (current_pixel)
            #
            #  1  2  3
            #  4  X  5
            #  6  7  8
            
            # Check against pixel 1
            if np.array_equal(current_pixel, image_array[i-1][j-1]) == False:
                # Color the corresponding pixel on the trace to black
                trace_array[i][j] = black
                # Advance inner loop
                continue
            
            # Check against pixel 2
            if np.array_equal(current_pixel, image_array[i-1][j]) == False:
                # Color the corresponding pixel on the trace to black
                trace_array[i][j] = black
                # Advance inner loop
                continue
            
            # Check against pixel 3
            if np.array_equal(current_pixel, image_array[i-1][j+1]) == False:
                # Color the corresponding pixel on the trace to black
                trace_array[i][j] = black
                # Advance inner loop
                continue
            
            # Check against pixel 4
            if np.array_equal(current_pixel, image_array[i][j-1]) == False:
                # Color the corresponding pixel on the trace to black
                trace_array[i][j] = black
                # Advance inner loop
                continue
            
            # Check against pixel 5
            if np.array_equal(current_pixel, image_array[i][j+1]) == False:
                # Color the corresponding pixel on the trace to black
                trace_array[i][j] = black
                # Advance inner loop
                continue
            
            # Check against pixel 6
            if np.array_equal(current_pixel, image_array[i+1][j-1]) == False:
                # Color the corresponding pixel on the trace to black
                trace_array[i][j] = black
                # Advance inner loop
                continue
            
            # Check against pixel 7
            if np.array_equal(current_pixel, image_array[i+1][j]) == False:
                # Color the corresponding pixel on the trace to black
                trace_array[i][j] = black
                # Advance inner loop
                continue
            
            # Check against pixel 8
            if np.array_equal(current_pixel, image_array[i+1][j+1]) == False:
                # Color the corresponding pixel on the trace to black
                trace_array[i][j] = black
                # Advance inner loop
                continue
            
    return trace_array



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
    h, w, d = tuple(original_image.shape)
    assert d == 3
    image_array = np.reshape(original_image, (h * w, d))
    
    cc = set()
    
    for idx in range(h*w):
            
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