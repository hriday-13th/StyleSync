import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import cv2
import PIL
import PIL.Image
import pandas as pd
import numpy as np
from flask import url_for
import random

df = pd.read_csv('/home/hprad/Projects/capstone/data/styles_images_path.csv', nrows=5000)

df['text'] = (df['gender'] + ' ' + df['gender'] + ' ' + df['gender'] + ' ' + df['gender'] + ' ' + 
              df['subCategory'] + ' ' + df['subCategory'] + ' ' + df['subCategory'] + ' ' + df['subCategory'] + ' ' + 
              df['baseColour'] + ' ' + df['baseColour'] + ' ' + df['baseColour'] + ' ' + df['baseColour'] + ' ' + 
              df['season'] + ' ' + df['season'] + ' ' + df['season'] + ' ' + df['season'] + ' ' + 
              df['usage'] + ' ' + df['productDisplayName'] + ' ' + df['productDisplayName'])

text_df = pd.DataFrame(df['text'])
text_df.fillna("", inplace=True)


model = SentenceTransformer('bert-base-nli-mean-tokens')
embeddings = model.encode(text_df['text'].tolist())
embedding_df = pd.DataFrame(embeddings)
embedding_df.to_csv('textual_embeddings_data.csv', index=False)

cosine_similarities = cosine_similarity(embedding_df)

n = 10
similar_products = {}
for i, row in enumerate(cosine_similarities):
    similar_indices = np.argsort(-row)[1:n+1]
    similar_ids = embedding_df.iloc[similar_indices].index.tolist()
    similar_products[embedding_df.index[i]] = similar_ids

similar_products[0]
df.iloc[similar_products[0]]
df.iloc[similar_products[1]]

def load_image(img_path, resized_fac = 0.1):
    img_object = plt.imread(img_path)
    w, h, c = img_object.shape
    resized = cv2.resize(img_object, (int(h*resized_fac), int(w*resized_fac)))
    return resized

def plot_image(image_id, styles_df):
    plt.imshow(load_image(styles_df.iloc[image_id]['image']))
    plt.title(styles_df.iloc[image_id]['productDisplayName'])

def similar(image_id, gender, subCategory, color, season):
    new_text = (gender + ' ' + gender + ' ' + gender + ' ' + gender + ' ' + 
                subCategory + ' ' + subCategory + ' ' + subCategory + ' ' + subCategory + ' ' + 
                color + ' ' + color + ' ' + color + ' ' + color + ' ' + 
                season + ' ' + season + ' ' + season + ' ' + season + ' ' + 
                df.loc[image_id, 'usage'] + ' ' + df.loc[image_id, 'productDisplayName'] + ' ' + 
                df.loc[image_id, 'productDisplayName'])

    new_embedding = model.encode([new_text])[0]
    embeddings = pd.read_csv('textual_embeddings_data.csv')
    existing_embeddings = embeddings.values
    
    similarity = cosine_similarity([new_embedding], existing_embeddings).flatten()
    similar_indices = similarity.argsort()[-n:][::-1]
    random_indices = random.sample(list(similar_indices), min(5, len(similar_indices)))

    image_paths = []
    for idx in random_indices:
        full_path = df.iloc[idx]['image']
        relative_path = os.path.join("images", os.path.basename(full_path))
        image_paths.append(url_for('static', filename=relative_path))
    return image_paths

def plot_similar_images(query_id, similarity_dict=similar_products, styles_df=df):
    similar_products = similarity_dict[query_id]
    if len(similar_products) > 5:
        similar_products = random.sample(similar_products, 5)
    image_paths = []
    for idx in similar_products:
        full_path = styles_df.iloc[idx]['image']
        relative_path = os.path.join("images", os.path.basename(full_path))
        image_paths.append(url_for('static', filename=relative_path))
    return image_paths