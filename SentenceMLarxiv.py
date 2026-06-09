
#%%
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np
from collections import Counter
import pandas as pd
import plotly.express as px

#%%
#How many arxiv papers
liste = []
with open("arxiv_metadata.json") as f:
    for i, line in enumerate(f):
        liste.append(i)
    
    print(len(liste))



#%%
#Retrieve text data from arxiv metadata
papers = []
batch_size = 5000
with open("arxiv_metadata.json") as f:
    for i, line in enumerate(f):
        if i >= batch_size:
            break
        data = json.loads(line)
        data["title"] = data["title"].replace('\n', ' ')
        data["abstract"] = data["abstract"].replace('\n', ' ')
        papers.append(data["title"].strip() + " " + data["abstract"].strip())

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(papers, normalize_embeddings=True)
print(embeddings.shape)


#%%
#Useful to display more data about arxiv paper
papers_matrix = []
data_cat = []
with open("arxiv_metadata.json") as f:
    for i, line in enumerate(f):
        if i >= batch_size:
            break
        data = json.loads(line)
        papers_matrix.append({"title" : data["title"].strip() , "abstract" : data["abstract"].strip() , "categories" : data["categories"], "authors" : data["authors"]})
        data_cat.append(json.loads(line))

category = [paper['categories'].split()[0] for paper in data_cat]
unique_cats = list(set(category))
cat_to_int = {cat: i for i, cat in enumerate(unique_cats)}
color_ids = [cat_to_int[cat] for cat in category]






#%%   To search nearest paper from to a specific set of words

query = "qunatum effect history"
human_query_embedding = model.encode(query, normalize_embeddings=True)

similarities = embeddings @ human_query_embedding
nearest_indices = np.argsort(similarities)[::-1]
for idx in nearest_indices[:10]:
    print()
    print("title:", papers_matrix[idx]["title"])
    print("categories:", papers_matrix[idx]["categories"], ", similarities:", similarities[idx])
    print("authors:", papers_matrix[idx]["authors"])






#%%   To search nearest papers to an actual other one
query_index = 76
query_embedding = embeddings[query_index]

similarities = embeddings @ query_embedding
nearest_indices = np.argsort(similarities)[::-1]
for idx in nearest_indices[:10]:
    print()
    print("title:", papers_matrix[idx]["title"])
    print("categories:", papers_matrix[idx]["categories"], ", similarities:", similarities[idx])
    print("authors:", papers_matrix[idx]["authors"])





#%%   some imports (umap and plots)
import umap
import matplotlib.pyplot as plt
import seaborn as sns



#%%   the umap fit_transform
n_neighbors=25
min_dist=0.5
n_components=3
metric='cosine'

fit = umap.UMAP(n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric)

u = fit.fit_transform(embeddings)




#%%   Display plots of the closeness of the data

#scatter = plt.scatter(u[:, 0], u[:, 1], c=color_ids, cmap='tab10', s=5, alpha=0.7)
#plt.title('UMAP of ArXiv Papers by Category')

fig = plt.figure(figsize=(20, 15))
if n_components == 1:
    ax = fig.add_subplot(111)
    ax.scatter(u[:,0], range(len(u)), c=color_ids)
if n_components == 2:
    ax = fig.add_subplot(111)
    ax.scatter(u[:,0], u[:,1], c=color_ids, cmap='tab10', s=5, alpha=0.5, linewidths=0)
if n_components == 3:
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(u[:,0], u[:,1], u[:,2], c=color_ids, cmap='tab10', s=10, alpha=0.8, linewidths=0)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=20, azim=30)
title = 'UMAP of ArXiv Papers by Category on ' + str(n_components) + ' components'
plt.title(title, fontsize = 10)



# %%   Helps for displaying color associated the most redundant categories
index_most_common = 25
category_counts = Counter(category)
top_cats = [cat for cat, _ in category_counts.most_common(index_most_common)]
categories_filtered = [cat if cat in top_cats else 'other' for cat in category]

colors = px.colors.qualitative.Alphabet

legend_data = []
for i, cat in enumerate(top_cats):
    hex_color = colors[i % len(colors)]
    legend_data.append({'Category': cat, 'Color': hex_color, 'Count': category_counts[cat]})

legend_df = pd.DataFrame(legend_data).sort_values('Count', ascending=False)

def color_cell(row):
    return ['', f'background-color: {row.Color}; color: {row.Color}', '']

legend_df.style.apply(color_cell, axis=1)



# %%
color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(top_cats)}
color_map['other'] = '#d3d3d3'  # light grey for 'other'

df = pd.DataFrame({
    'x': u[:, 0],
    'y': u[:, 1],
    'z': u[:, 2],
    'category': categories_filtered,
    'title': [paper['title'] for paper in papers_matrix]
})
fig = px.scatter_3d(
    df, x='x', y='y', z='z',
    color='category',
    color_discrete_map=color_map,
    hover_data={'title': True, 'x': False, 'y': False, 'z': False, 'category':True},
    opacity=0.5,
    size_max=2,
    title='UMAP 3D - ArXiv Papers by Category'
)

fig.update_traces(marker=dict(size=2))
fig.update_layout(width=700, height=600)
fig.show()
# %%
