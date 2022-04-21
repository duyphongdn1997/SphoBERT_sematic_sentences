import numpy as np
import pandas as pd
import plotly.express as px
from ml.sentence_transformers.SentenceTranformer import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

model = SentenceTransformer('/home/phongdtd/phongdtd/SphoBERT_sematic_sentences/ml/'
                            'model/phobert_base_mean_tokens_NLI_STS')

corpus = ['Cô giáo đang ăn kem.',
          'Cô giáo đang ăn bánh mì.',
          'Chị gái đang thử món thịt dê.',
          'Một anh trai đang cưỡi ngựa.',
          'Kị binh đang đi tuần.',
          'Một vụ tai nạn thảm khốc vừa xảy ra.',
          'Đã có ít nhất hai người chết trong vụ xe khách rơi xuống vực.',
          'Một con khỉ đang diễn xiếc.',
          'Một người đàn ông trong bộ đồ tinh tinh đang làm trò.',
          'Lũ quét đột ngột gây thiệt hại cho ít nhất 3 tỉnh miền núi']

corpus_embeddings = model.encode(corpus)
print(np.array(corpus_embeddings).shape)

num_clusters = 5

clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_
X = np.array(corpus_embeddings)

pca = PCA(n_components=3)
result = pca.fit_transform(X)

# create dataframe to feed to
df = pd.DataFrame({
    'sent': corpus,
    'cluster': cluster_assignment.astype(str),
    'x': result[:, 0],
    'y': result[:, 1],
    'z': result[:, 2]
})
fig = px.scatter_3d(df, x='x', y='y', z='z',
                    color='cluster', hover_name='sent',
                    range_x=[df.x.min() - 1, df.x.max() + 1],
                    range_y=[df.y.min() - 1, df.y.max() + 1],
                    range_z=[df.z.min() - 1, df.z.max() + 1])

fig.update_traces(hovertemplate='<b>%{hovertext}</b>')
fig.show()
