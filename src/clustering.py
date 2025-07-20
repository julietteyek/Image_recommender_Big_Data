# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 23:45:53 2024

@author: marko
"""
"""
wählen kmeans, da so alle Bilde reinem cluster zugeordnet werden,
 egal wie gut sie in dieses reinpassen (also werden immer dem zugeordnet was noch am besten passt)
 """
from histograms import hist
from phashes import perceptual_hashes
import cv2
from os.path import join, isfile
from pathlib import Path
import os
import argparse
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np


def fit_cluster(n_clusters=10):
    """
    Returns three kmeans models based on 1. RGB histograms, 2. Resnet embeddings, 3. Perceptual Hashes

    Adds columns to data.pk ('cluster_histogram''cluster_embedding''cluster_phash') which include the cluster each image gets sorted to

            Parameters:
                    n_clusters (int): number of clusters; default; n_clusters = 10
            Returns:
                    data.pk (file): pickle file with all of the image data including clusters

                    kmeans_model_histograms.pkl (file): includes kmeans model

                    kmeans_model_embeddings.pkl (file): includes kmeans model

                    kmeans_model_perceptualhashes.pkl (file): includes kmeans model

            use-example: fit_cluster()


    """
    print("Collecting data")

    # loading pickle file with data
    # file = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/data.pk"
    # getting current path
    current_path = os.getcwd()

    # path to pickle file
    pk_file = "pickle/data.pkl"

    pk_path = join(current_path, pk_file)
    data = pd.read_pickle(pk_path)
    
    # getting data for embedding clusteting
    #emb_col = pd.DataFrame(data["Embeddings"])
    emb_col = np.stack(data['Embeddings'].values)
    # getting data for histograms clustering
    histograms_col = pd.DataFrame(data["RGB_Histogram"].tolist())

    # getting data for perceptual hash clustering
    phash_col = pd.DataFrame(data["Perceptual_Hash"].tolist())


    """
    #getting data for all 3 combined
    all_33 = 
    """
    """
    #scaling of data
    scaler = StandardScaler()
    
    scaler.fit(histograms_col)
    histograms_col = pd.DataFrame(scaler.transform(histograms_col),
                               columns= histograms_col.columns )
    
    scaler.fit(phash_col)
    histograms_col = pd.DataFrame(scaler.transform(phash_col),
                               columns= phash_col.columns )
    
    scaler.fit(emb_col)
    histograms_col = pd.DataFrame(scaler.transform(emb_col),
                               columns= emb_col.columns )
    
    """

    # initializing clustering model
    def initializing_kmeans(n_clusters):
        print(f"\ninitializing kmeans-models with {n_clusters} clusters")

        kmeans_histo = KMeans(n_clusters=n_clusters, random_state=42)

        kmeans_phash = KMeans(n_clusters=n_clusters, random_state=42)

        kmeans_emb = KMeans(n_clusters=(n_clusters*3), random_state=42)

        """
        kmeans_all = KMeans( n_clusters = n_clusters, random_state = 0)
        """

        return kmeans_emb, kmeans_histo, kmeans_phash  # , kmeans_all

    # fitting model to methods
    def fit_and_save_kmeans(n_clusters):
        print("\nfitting and saving kmeans-models\n")
        kmeans_emb, kmeans_histo, kmeans_phash = initializing_kmeans(n_clusters)

        # fitting kmeans models
        kmeans_histo.fit(histograms_col)
        kmeans_phash.fit(phash_col)
        kmeans_emb.fit(emb_col)

        # path to pickle file
        modelfile_histo = "pickle/kmeans_models/kmeans_model_histograms.pkl"
        modelfile_phash = "pickle/kmeans_models/kmeans_model_perceptualhashes.pkl"
        modelfile_emb = "pickle/kmeans_models/kmeans_model_embeddings.pkl"

        modelfile_histo = join(current_path, modelfile_histo)
        modelfile_phash = join(current_path, modelfile_phash)
        modelfile_emb = join(current_path, modelfile_emb)
        
        print(modelfile_histo)
        # saving the models
        with open(modelfile_histo, "wb") as modelfile_histo:
            pickle.dump(kmeans_histo, modelfile_histo)

        with open(modelfile_phash, "wb") as modelfile_phash:
            pickle.dump(kmeans_phash, modelfile_phash)
            
        with open(modelfile_emb, "wb") as modelfile_emb:
            pickle.dump(kmeans_emb, modelfile_emb)

        data_clustered = data.copy()
        # adding clusters to data.pk
        data_clustered["cluster_histogram"] = kmeans_histo.labels_
        data_clustered["cluster_embedding"] = kmeans_emb.labels_
        data_clustered["cluster_phash"] = kmeans_phash.labels_

        # saving into a new pickle (copy of data + clusters)
        clustered_data_path = "pickle/data_clustered.pkl"
        clustered_data_path = join(current_path, clustered_data_path)
        # saving file
        data_clustered.to_pickle(clustered_data_path)



    print("Starting clustering")
    fit_and_save_kmeans(n_clusters)

    print("clustering finished!")


def predict_cluster(img_path, method, data):  # , histogram, embedding, phash):
    """
    Returns the cluster (int) the uploaded image belongs to


            Parameters:
                    n_clusters (int): number of clusters
                    img: loaded image
                    img_path: oath of the image
                    method: method of comparison, choose between  ['histogram', 'embeddings', 'hashes']
                    data: data we need for comparing the image, based on which method we choose (histogram of the image, embedding of the image, perceptual hash of the image)
            Returns:
                    cluster (int): number of the cluster the image belongs to

            use-example: cluster = predict_cluster(img, image_path, method = 'histogram', data = histogram)

    """

    current_path = os.getcwd()

    if method == "histogram":
        modelfile = "pickle/kmeans_models/kmeans_model_histograms.pkl"
        modelfile = join(current_path, modelfile)
        # modelfile = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_models/kmeans_model_histograms.pkl"

        # data = histogram

    elif method == "embeddings":
        modelfile = "pickle/kmeans_models/kmeans_model_embeddings.pkl"
        modelfile = join(current_path, modelfile)
        # data = embedding

    elif method == "hashes":
        modelfile = "pickle/kmeans_models/kmeans_model_perceptualhashes.pkl"
        modelfile = join(current_path, modelfile)
        # data = phash_vector

    with open(modelfile, "rb") as file:
        kmeans = pickle.load(file)

    if method == "histogram":
        new_data = pd.DataFrame({"Name": img_path, "RGB_Histogram": [data]})

        # Konvertiere die neue Liste des Histograms in einen DataFrame
        new_data_df = pd.DataFrame(new_data["RGB_Histogram"].tolist())

        """
        #scaling data
        scaler = StandardScaler()

        scaler.fit(new_data_df)
        new_data_df = pd.DataFrame(scaler.transform(new_data_df),
                                   columns= new_data_df.columns )
        """
    elif method == "embeddings":
        new_data = pd.DataFrame({"Name": img_path, "Embeddings": [data]})
        
        new_data_df = np.stack(new_data["Embeddings"].values) 
        """
        #scaling data
        scaler = StandardScaler()

        scaler.fit(new_data_df)
        new_data_df = pd.DataFrame(scaler.transform(new_data_df),
                                   columns= new_data_df.columns )
        """
    elif method == "hashes":
        new_data = pd.DataFrame({"Name": img_path, "Perceptual_Hash": [data]})
        # Konvertiere die neue Liste der perceptual hashes in ein DataFrame
        new_data_df = pd.DataFrame(new_data["Perceptual_Hash"].tolist())

        """
        #scaling data
        scaler = StandardScaler()

        scaler.fit(new_data_df)
        new_data_df = pd.DataFrame(scaler.transform(new_data_df),
                                   columns= new_data_df.columns )
        """

    # Weise den neuen Daten Cluster zu, ohne das Modell neu zu trainieren
    new_data["cluster"] = kmeans.predict(new_data_df)

    return new_data.iloc[0]["cluster"]
