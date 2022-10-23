from string import punctuation
import pandas as pd
from fuzzywuzzy import process

VERSION='05'
filename = f'df_listings_v{VERSION}.csv'
#remote_pathname = f'https://raw.githubusercontent.com/jayportfolio/capstone_streamlit/main/data/final/{filename}'


df_pathname_raw = f'../../data/source/{filename}'
#df_pathname_tidy = f'../../data/final/{filename}'

df = pd.read_csv(df_pathname_raw, on_bad_lines='error', index_col=0)

df['keyFeatures'].to_numpy()
df['keyFeatures2'] = df['keyFeatures'].str.split(',')
df[['keyFeatures','keyFeatures2']]

feature_list = []

for feature_raw in df['keyFeatures']:
    feature = feature_raw.strip("][").split("\', \'")
    feature = [s.strip(punctuation).lower().strip() for s in feature]
    feature_list.extend(feature)

df['keyFeatures']

#xxxnice_feature_list = feature_list.unique()
#matches = process.extract('italian',feature_list, limit=len(feature_list))

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=5, token_pattern='(\S+)')
tf_idf_matrix = tfidf_vectorizer.fit_transform(feature_list)


def get_matches_df(sparse_matrix, name_vector, top=100):
    non_zeros = sparse_matrix.nonzero()

    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size

    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)

    for index in range(0, nr_matches):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]

    return pd.DataFrame({'TITLE': left_side,
                         'SIMILAR_TITLE': right_side,
                         'similairity_score': similairity})


matches_df = pd.DataFrame()
matches_df = get_matches_df(matches, df['TITLE'], top=10000)
# Remove all exact matches
matches_df = matches_df[matches_df['similairity_score'] < 0.99999]
matches_df.sample(10)


print('end')

