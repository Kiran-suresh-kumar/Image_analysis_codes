#The function performs K-Means clustering on the input images using the KMeans class from scikit-learn. 
#The fit method is called on the images array to fit the K-Means model to the data.
#kmeans.labels_ is then used to obtain the lables of each image.
#An empty dictionary called image_labels_mapping to store the mapping between image filenames (or indices) and their corresponding cluster labels.

import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import glob
from tqdm import tqdm

#Loading images

def load_images(folder_path):
    images = []
    for file in glob.glob(os.path.join(folder_path, "*.tif")):
        img = cv2.imread(file)
        images.append(img)
    return np.array(images

#Pre-processing images 
def preprocess_images(images):
    processed_images = []
    for img in images:
        resized_img = cv2.resize(img, (100, 100))
        flattened_img = resized_img.reshape((-1,))  # Flatten image into a vector
        processed_images.append(flattened_img)
    return np.array(processed_images)

#Classifying images
def classify_images(images, image_filenames, num_clusters=4):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(images)
    # Get cluster labels
    cluster_labels = kmeans.labels_

    # Associate image filenames or indices with cluster labels
    image_labels_mapping = {}  # Dictionary to store image labels mapping

    # Assuming 'image_filenames' contains filenames of input images
    for i, filename in enumerate(image_filenames):
        image_labels_mapping[filename] = cluster_labels[i]
    return kmeans.labels_,image_labels_mapping



# Specify the path to the folder containing the images
folder_path = "\path to the folder\"

# Define the pattern to match all files
all_files_pattern = os.path.join(folder_path, "*")

# Use glob to get a list of all file names in the folder
all_files = glob.glob(all_files_pattern)

# Extract only the filenames without the folder path
all_filenames = [os.path.basename(file) for file in all_files]

# Print the list of file names
#print("All filenames in the folder:")
#for filename in all_filenames:
#    print(filename)


images = load_images(folder_path)

# Preprocess images
processed_images = preprocess_images(images)


labels,image_labels_mapping = classify_images(processed_images,all_filenames)

#to check which cluster 'i' does the images belong to 
#for filename, label in image_labels_mapping.items():
#    if label == i:
#        print(f"Image {filename} belongs to cluster i")


#Plotting 3 images belonging to class 0 together
import matplotlib.pyplot as plt
from skimage import io

# Load the images
img_1_path = 'image_1_class0.tif'
img_1 = io.imread(img_1_path)

img_2_path = 'image_2_class_0.tif'
img_2 = io.imread(img_2_path)

img_3_path = 'image_3_class_0.tif'
img_3 = io.imread(img_3_path)


# Display the first image
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.imshow(img_1)
plt.title('Image 1')
plt.axis('off')

# Display the second image
plt.subplot(1, 3, 2)
plt.imshow(img_2)
plt.title('Image 2')
plt.axis('off')

# Display the third image
plt.subplot(1, 3, 3)
plt.imshow(img_3)
plt.title('Image 3')
plt.axis('off')

plt.show()
