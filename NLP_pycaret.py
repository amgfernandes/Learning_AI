# %%
from pycaret.datasets import get_data


# %%
import spacy

# %%
data = get_data('kiva')
#check the shape of data
data.shape
# %%
# sampling the data to select only 1000 documents
data = data.sample(1000, random_state=786).reset_index(drop=True)
data.shape
# %%
from pycaret.nlp import *
exp_nlp101 = setup(data = data, target = 'en', session_id = 123, log_experiment=False)
# %%
lda = create_model('lda')
print(lda)
# %%
lda_results = assign_model(lda)
lda_results.head()
# %%
plot_model()
# %%
plot_model(plot = 'bigram')

# %%
plot_model(lda, plot = 'frequency', topic_num = 'Topic 0')
# %%
plot_model(lda, plot = 'frequency', topic_num = 'Topic 1')
# %%
plot_model(lda, plot = 'frequency', topic_num = 'Topic 2')

# %%
plot_model(lda, plot = 'frequency', topic_num = 'Topic 3')
# %%
plot_model(lda, plot = 'topic_distribution')
# %%
plot_model(lda, plot = 'umap')
# %%
evaluate_model(lda)
# %%
save_model(lda, 'Final LDA model'