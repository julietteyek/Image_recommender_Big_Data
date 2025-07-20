import cv2
import psycopg2
import pandas as pd
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt

from tkinter import filedialog
from os.path import join

from resnet_extraction import ResNet_Feature_Extractor
from phashes import perceptual_hashes
from histograms import hist
from scipy.spatial.distance import euclidean, hamming


# data_cluster.pkl
# ----> .pk unbedingt in .pkl!
class Recommender_NC:

    def __init__(self, methods):
        self.methods = methods

    def filedialog(self):
        root = tk.Tk()
        root.withdraw()
        source_image_paths = filedialog.askopenfilenames(title="upload your image(s)")
        if not source_image_paths:
            return None

        # turn image path(s) into arrays
        source_images = []
        for path in source_image_paths:
            img = cv2.imread(path)
            if img is not None:
                source_images.append(img)
            else:
                print(f"Failed to load image: {path}")

        return source_images

    def recommend(self):
        # open filedialog to select image(s):
        source_images = self.filedialog()

        if not source_images:
            print("No image selected.")
            return

        combined_results = []

        for method in self.methods:
            print(f"processing with method: {method}")

            # extract features from the uploaded image(s):
            features = [self.extract_features(img, method) for img in source_images]
            uploaded_feature = np.mean(features, axis=0)

            if uploaded_feature is not None:
                # load the features from pickle
                pickle_path = "pickle/data.pkl"
                dataset = pd.read_pickle(pickle_path)
                nearest_neighbors = self.find_nearest_neighbors(
                    uploaded_feature, dataset, method, k=5
                )
                combined_results.append(
                    (method, nearest_neighbors)
                )  # combine top-k-images from each method
            else:
                print("Failed to extract features from the upload.")
        # display the combined results
        self.show_results(combined_results)

    def extract_features(self, img, method):
        if method == "embeddings":
            resnet_extractor = ResNet_Feature_Extractor()
            return resnet_extractor.extract_features(img)
        elif method == "hashes":
            return perceptual_hashes(img)
        elif method == "histogram":
            return hist(img)
        else:
            raise ValueError(f"Unknown method: {method}")

    def find_nearest_neighbors(self, uploaded_feature, dataset, method, k=5):
        distances = []
        if method == "embeddings":
            method_column = "Embeddings"
        elif method == "hashes":
            method_column = "Perceptual_Hash"
        elif method == "histogram":
            method_column = "RGB_Histogram"
        uploaded_feature = np.ravel(
            uploaded_feature
        )  # Convert uploaded_feature to a 1D array

        for idx, feature in dataset[method_column].items():
            feature = np.ravel(np.array(feature))

            if method == "embeddings":
                dist = euclidean(uploaded_feature, feature)
            elif method == "hashes":
                dist = hamming(uploaded_feature, feature) * len(feature)
            elif method == "histogram":
                dist = self.chi_square_distance(uploaded_feature, feature)
            else:
                raise ValueError(f"Unknown method: {method}")

            distances.append((dist, idx))  # store distance & index

        # sort distances by the computed distance
        distances.sort(key=lambda x: x[0])
        top_k = distances[:k]
        top_images = []

        conn = psycopg2.connect(
            host="localhost", database="imagerec", user="postgres", password="meep"
        )
        cursor = conn.cursor()

        for _, image_id in top_k:
            cursor.execute("SELECT name FROM images_into_db WHERE id = %s", (image_id,))
            result = cursor.fetchone()
            if result:
                image_path = result[0]  # get image path from the result
                img = cv2.imread(image_path)
                if img is not None:
                    top_images.append(img)

        conn.close()
        #print(top_k, top_images)

        return top_k, top_images

    def chi_square_distance(self, histA, histB, eps=1e-10):
        return 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))

    def show_results(self, combined_results):
        if not combined_results:
            print("No neighbors found.")
            return

        # total number of images that need to be displayed:
        total_images = sum(len(top_images) for _, (_, top_images) in combined_results)

        cols = 5  # bc 5 recommended images are demanded
        rows = (total_images + cols - 1) // cols
        plt.figure(figsize=(15, 5 * rows))
        plt.suptitle("We thought you might also like the following:", fontsize=16)

        img_idx = 1

        for method, (top_k, top_images) in combined_results:
            for i, (dist, img) in enumerate(zip(top_k, top_images)):
                ax = plt.subplot(rows, cols, img_idx)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # vonvert BGR to RGB
                ax.set_title(f"{method}: Dist: {dist[0]:.2f}", fontsize=10)
                plt.axis("off")
                img_idx += 1

        # plt.tight_layout(h_pad=0.2, w_pad=0.3)  # reduce the space between rows (vertically)
        # plt.subplots_adjust(hspace=0.1)  # decreases space between rows
        plt.show()


# recommender = Recommender(method="resnet_embedding")
# recommender.recommend()