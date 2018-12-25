# encoding: utf-8

"""
Here we have the code that works on the README.md file

"""

from clustering.SOM import SOM
import numpy as np
from matplotlib import pyplot as plt

"""
Example of SOM with a simple color-example
"""

r_color = [0, 0, 0, 0.125, 0.33, 0.6, 0, 1, 0, 1, 1, 1, 0.33, 0.5, 0.66]
g_color = [0, 0, 0, 0.529, 0.4, 0.5, 1, 0, 1, 0, 1, 1, 0.33, 0.5, 0.66]
b_color = [0, 1, 0.5, 1, 0.67, 1, 0, 0, 1, 1, 0, 1, 0.33, 0.5, 0.66]
# Training inputs for RGB-colors
colors = list(map(list, list(zip(r_color, g_color, b_color))))

color_names = ['black', 'blue', 'darkblue', 'skyblue', 'greyblue', 'lilac', 'green', 'red', 'cyan', 'violet',
               'yellow', 'white', 'darkgrey', 'mediumgrey', 'lightgrey']

# Train a 20x30 SOM with 400 iterations
som = SOM(m=20, n=30, dim=3, n_iterations=400)
som.train(colors)

# Get output grid
image_grid = som.get_centroids()
# Map colours to their closest neurons
mapped = som.map_vects(colors)

# Plot
plt.figure('simple_SOM')
plt.imshow(image_grid)
plt.title('Color SOM')
for i, m in enumerate(mapped):
    plt.text(m[1], m[0], color_names[i], ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.5, lw=0))

plt.figure('simple_SOM')
plt.savefig('./Plot/Clustering/simple_SOM.png')

"""
Example of SOM with a more difficult problem.

Because colors are easy to plot.. we would also like to showcase what happens when the weights dont have some nice 
3-D like structure.. 

Therefore we get some other dummy data... we will shuffle the colors above and add them,  so we get some 6 
dimensional data. With this example you can see that if you shift through the layers you see that the layers 0:2 and 
3:6 show the proposed colors, while the layers in between show a mixture.
"""

# Create a permutation of the original columns
shuffle_index = np.random.choice(range(len(colors)), size=15, replace=False)
shuffle_colors = np.array(colors)[shuffle_index]
shuffle_color_names = np.array(color_names)[shuffle_index]

combined_colors = [list(x) + colors[i] for i, x in enumerate(shuffle_colors)]

m_SOM = 20
n_SOM = 30
n_dim = 6
# Setup the SOM object
som = SOM(m=m_SOM, n=n_SOM, dim=n_dim, n_iterations=400)
# Train on the new colors
som.train(combined_colors)

# Get output grid
image_grid = som.get_centroids()
# Map colours to their closest neurons
mapped = som.map_vects(combined_colors)


# Now that we have the trained SOM, we are going to extract the
s_min = 0
s_max = 5
s_size = 3
init_slice = np.array(np.arange(s_min, s_min+s_size))
max_slice = s_max-s_min-s_size+2
slicing_array = [init_slice + x for x in range(max_slice)]

# Because the data is now 6-dimensional, we cannot plot it immediately,
# hence we need to slice it down to several 3-D plots.
list_image_grid_sel = []
for i_slice in slicing_array:
    dum = [x[i_slice] for b in image_grid for x in b]
    list_image_grid_sel.append(np.reshape(dum, (m_SOM, n_SOM, 3)))

for i_plot, i_data in enumerate(list_image_grid_sel):
    plt.figure(i_plot)
    plt.clf()  # Make sure that we have nothing in it before plotting.
    plt.imshow(i_data)
    if i_plot == 0:
        for i, m in enumerate(mapped):
            plt.text(m[1], m[0], shuffle_color_names[i], ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.5, lw=0))
    if i_plot == (len(slicing_array)-1):
        for i, m in enumerate(mapped):
            plt.text(m[1], m[0], color_names[i], ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.5, lw=0))
    plt.savefig('./Plot/Clustering/mutli_layer_SOM_' + str(i_plot) + '.png')
