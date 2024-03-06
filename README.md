# Embedding-based Reference-Free Evaluation Metrics for Topic Segmentation

The repository includes the metrics and experiments used in the paper "When Cohesion Lies in the Embedding Space: New Framework and Methodologies for Embedding-Based Reference-Free Metrics for Topic Segmentation".

From this repository both the experiments presented in the paper and the embedding-based metrics on your own segmentations can be used.

# Requirements

To start, create a conda environment

```
conda create -n refree_topseg
```

and install all the dependencies

```
pip install -r requirements.txt
```

# Run the experiments

To run the experiments presented in the original paper, first you need to download the individual datasets in the data folder. All the datasets but SBBC-RadioNews are publicly available and can be found here:

- Wikisection Datasets:  https://github.com/sebastianarnold/WikiSection

- QMSum Dataset:  https://github.com/Yale-LILY/QMSum

Specifically, download all the json files from the Wikisection dataset into data/wikisection and clone the original QMSum repository into the data folder, as well.
Ince done that, the experiments are executed by the main.py function as following:

```
main.py -d dataset_name -e model_name
```

Where dataset_name should be one of "QMSUM", "wikisection_en_city" or "wikisection_en_disease", while model_name should be one of "roberta-base", "falcon" or "MPNET".

All the results will be stored as figures (for the synthetic experiments) in "output_figures" folder and as tables (for the real world experiments) in the "output_correlations" folder.

## Running Experiments With Falcon Embeddings
As Falcon is a very computationally expensive model, we have divided the process of extracting embeddings from the model in a separate script that should be run before using the main.py script with falcon as an encoder.

For each of the three datasets described above you should run then run the following:

```
extract_embeddings_with_falcon.py dataset_name
```

with dataset_name being one of "wikisection_en_city", "wikisection_en_disease" or "QMSUM".

# Run Our Embedding-based metrics on your own segmentations
You can use the embedding-based scores for reference-free evaluation on your own segmentations as well.

To use ARP method, for example, you can use our ARP class by first instantiating:

```
from embedding_metrics import ARPMetric

metric = ARPMetric(score_function, encoder)
```

Where score_function is one of "pairwise_cosine", "standard_deviation" or "average_cosine", as described in the paper, while encoder should be either a sentence-transformer model or a viable huggingface model (transformer encoders like BERT and RoBERTa are the only ones supported for now). In both cases, the encoder argument should be the name of the model as it appears on its official model card on huggingface hub. For example:

```
metric = ARPMetric(score_function="pairwise_cosine", encoder="all-mpnet-base-v2")
```

which is also the default configuration and, therefore, equivalent to:
```
metric = ARPMetric()
```

At this point, we need a corpus and a segmentation for it in the following form:
```
corpus = [["document 1 sentence 1", "document 1 sentence 2 (end of topic segment)", "document 1 sentence 3"], ["document 2 sentence 1", "document 2 sentence 2"]]

segmentation = [[0, 1, 0], [0, 0]]
```

In the above minimal example, the corpus is a list of 2 documents each including sentences which are separated in a list. The segmentation, then it is also a list of lists, where each top-level list corresponds to the document having the same position in the corpus and includes a list of binary integers, 1 if the sentence at the given document and at the given index in the document ends a topic segment and 0 otherwise. This is also the same format described in the popular [nltk](https://www.nltk.org/api/nltk.metrics.segmentation.html) library.

Having our corpus and hypothesised segmentation, then, we can obtain the ARP score for the corpus by running the following:
```
from embedding_metrics import ARPMetric

metric = ARPMetric() # using default settings

corpus = {your corpus}

segmentation = {your hypothesised segmentation}

ARP_score = metric.evaluate_segmentation(corpus, segmentation)
```

Additionally, we can pass custom embeddings instead of using our encoder to extract them in the following way:
```
ARP_score = metric.evaluate_segmentation(corpus, segmentation, embeddings=custom_embeddings)
```
Where custom_embeddings should be a list of numpy arrays (embeddings) corresponding to the same document-sentence structure (e.g. in the previous example it would be a list of 2 embedding arrays, one of size 3xembedding_dimension and the second of size 2xembedding_dimension).

Finally, if you want to return all the scores for each input document together with the aggregate average of such scores:
```
aggregated_average_score, document_level_scores = metric.evaluate_segmentation(corpus, segmentation, output_all_scores=True)
```

To use SegReFree, just change the instantiation to:
```
metric = SegReFreeMetric()
```
And equally, for Silhouette:
```
metric = SilhouetteMetric()
```
More specific parameters for these two metrics can be found in the code. Use them as described for ARP metric.
