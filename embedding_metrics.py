from itertools import combinations
from joblib import Parallel, delayed

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import distance_metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import check_random_state

from sentence_encoders import *

def cosine_dispersion(segments):
  mean = np.nanmean(segments, axis = 0).reshape(1, -1)
  return 1-np.nanmean(cosine_similarity(mean, segments))

def average_variance(segments):
  return np.linalg.norm(np.nanstd(segments, axis = 0))

def average_pairwise_similarity(segments):
  x = cosine_similarity(segments)
  x = np.triu(x, k=1)
  x[x==0] = np.nan
  return 1-np.nanmean(x)

def SegReFreeScore(segments, correction_factor=True, negative=False, bounded=False):
  sign=-1 if negative else 1
  mean = np.nanmean(segments, axis = 0).reshape(1, -1)
  if len(segments)>1 or not correction_factor:
    if correction_factor:
      return mean, np.nanmean(euclidean_distances(mean, segments))/(1-(1/len(segments)**0.5))*sign
    else:
      return mean, np.nanmean(euclidean_distances(mean, segments))*sign
  else:
    return mean, None

def SegReFree(segments, verbose=False, max_choice = True,
              correction_factor=True, negative=False,
              bounded=False, return_all_scores_and_means=False,
              default_cap=10, no_backoff=False):
  """
    Code to calculate the SegReFree score as described in https://openreview.net/forum?id=NVM6eCm69eu.

    Input:
        segments <- List: list of list containing the embeddings grouped for each ground truth segment
        max_choice <- bool: if True, it includes the max pooling between the two adjacent topic segments, else it uses just the following for comparison (same as ARP).
        correction_factor <- bool: if True, it applies a correction factor for de-biasing the metric with respect to short segments. If False, the metric is the same as the Davies-Bouldin Index.
        negative <- bool: If True, outputs the negative SegReFree score. Use just for debugging.
        bounded <- bool: If True, normalise the scores with sigmoid, so that the metric is bounded between 0 and 1
        return_all_scores_and_means <- bool: If True, return all the R scores and the segments' centroids. Use for debugging.
        default_cap <- int|"auto": a positive integer or the string "auto" determining a default value to assign to the document-level score in the cases of metrics' failure (i.e. no topic boundary or each segment has length 1). If "auto" is used, then it automatically assign to these cases the highest score among the ones in the corpus (use just if having more than one document).
        no_backoff <- bool: If True, the metric will assign R=0 to each segment of length 1, as in the original paper. This is not recommended as it yields results that are stongly skewed towards favoring segments of length 1.
    Output:
        if return_all_scores_and_means=False:
          float: the SegReFree score.
        else:
          tuple: (float, List of tuples). Where the first element is the SegReFree Score and the second is a list including R score and centroid for each segment.

    """
  all_scores = []
  if return_all_scores_and_means:
    all_scores_and_means = []
  single_scores = 0
  all_one_scores = 0
  sign=-1 if negative else 1
  use_max_r_as_cap = False
  if default_cap=="auto":
    use_max_r_as_cap = True
  for doc_index, doc in enumerate(segments):
        if len(doc)<2:
          if verbose:
            print(f"Warning: document no segmentation found for document number {doc_index}: score for the document defaulting to 0")
          single_scores += 1
          continue
        segment_S = []
        one_sentence_segment_indeces = []
        defined_S = 0
        for index, seg in enumerate(doc):
          S = SegReFreeScore(seg, correction_factor, negative, bounded)
          if S[1] is None:
            one_sentence_segment_indeces.append(index)
          else:
            defined_S += 1

          segment_S.append(S)

        if defined_S:
          if no_backoff:
            mean_S = 0
          else:
            mean_S = np.mean([s[1] for s in segment_S if s[1] is not None])
        else:
          all_one_scores+=1

        for idx in one_sentence_segment_indeces:
          segment_S[idx]=(segment_S[idx][0], mean_S)

        if return_all_scores_and_means:
          all_scores_and_means.append(([s[1] for s in segment_S], [s[0] for s in segment_S]))

        for index, (mean_S, S) in enumerate(segment_S):
          if max_choice:
            if index>0 and index<len(segment_S)-1:
              if bounded:
                  R_prev = (S+segment_S[index-1][1])/(euclidean_distances(mean_S, segment_S[index-1][0])+(S+segment_S[index-1][1]))
                  R_fol = (S+segment_S[index+1][1])/(euclidean_distances(mean_S, segment_S[index+1][0])+(S+segment_S[index+1][1]))
              else:
                  R_prev = (S+segment_S[index-1][1])/euclidean_distances(mean_S, segment_S[index-1][0])
                  R_fol = (S+segment_S[index+1][1])/euclidean_distances(mean_S, segment_S[index+1][0])
              R = max(R_prev[0][0], R_fol[0][0])
            elif index:
              if bounded:
                R = [(S+segment_S[index-1][1])/(euclidean_distances(mean_S, segment_S[index-1][0])+(S+segment_S[index-1][1]))][0][0][0]
              else:
                R = [(S+segment_S[index-1][1])/euclidean_distances(mean_S, segment_S[index-1][0])][0][0][0]
            else:
              if bounded:
                R = (S+segment_S[index+1][1])/(euclidean_distances(mean_S, segment_S[index+1][0])+(S+segment_S[index+1][1]))[0][0][0]
              else:
                R =  [(S+segment_S[index+1][1])/euclidean_distances(mean_S, segment_S[index+1][0])][0][0][0]
          else:
            try:
              if bounded:
                R = [(S+segment_S[index+1][1])/(euclidean_distances(mean_S, segment_S[index+1][0])+(S+segment_S[index+1][1]))][0][0][0]
              else:
                R =  [(S+segment_S[index+1][1])/euclidean_distances(mean_S, segment_S[index+1][0])][0][0][0]
            except IndexError:
              pass

          all_scores.append(R.item())
          
  if all_scores:
    pass
  else:
    # no segmentations in the corpus
    return float(default_cap)*sign
  if use_max_r_as_cap:
    default_cap_max = max(all_scores)
    default_cap_mean = 0 if no_backoff else np.nanmean(all_scores)
  else:
    default_cap_max = default_cap
    default_cap_mean = 0 if no_backoff else default_cap
  if single_scores:
    all_scores = all_scores + [float(default_cap_max) for _ in range(single_scores)]
  if all_one_scores:
    all_scores = all_scores + [float(default_cap_mean) for _ in range(all_one_scores)]
  if return_all_scores_and_means:
    return np.nanmean(all_scores)*sign, all_scores_and_means
  return np.nanmean(all_scores)*sign

def silhouette_score_block(X, labels,  one_segment_docs, N, metric='euclidean', sample_size=None,
                           random_state=None, n_jobs=1, **kwds):
    """Compute the mean Silhouette Coefficient of all samples.
    Implementation from https://gist.github.com/AlexandreAbraham/5544803
    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (a) and the mean nearest-cluster distance (b) for each sample.
    The Silhouette Coefficient for a sample is ``(b - a) / max(a, b)``.
    To clarrify, b is the distance between a sample and the nearest cluster
    that b is not a part of.
    This function returns the mean Silhoeutte Coefficient over all samples.
    To obtain the values for each sample, use silhouette_samples
    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more similar.
    Parameters
    ----------
    X : array [n_samples_a, n_features]
        Feature array.
    labels : array, shape = [n_samples]
             label values for each sample
    one_segment_docs : int
        The number of documents for which no segmentation exists (have only
        one segment).
    N : int
        The total number of data points to be used as denominator.
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by metrics.pairwise.pairwise_distances. If X is the distance
        array itself, use "precomputed" as the metric.
    sample_size : int or None
        The size of the sample to use when computing the Silhouette
        Coefficient. If sample_size is None, no sampling is used.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    `**kwds` : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
    Returns
    -------
    silhouette : float
        Mean Silhouette Coefficient for all samples.
    References
    ----------
    Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
        Interpretation and Validation of Cluster Analysis". Computational
        and Applied Mathematics 20: 53-65. doi:10.1016/0377-0427(87)90125-7.
    http://en.wikipedia.org/wiki/Silhouette_(clustering)
    """
    return (np.sum(silhouette_samples_block(
        X, labels, metric=metric, n_jobs=n_jobs, **kwds))-one_segment_docs)/(N+one_segment_docs)


def silhouette_samples_block(X, labels, metric='euclidean', n_jobs=1, **kwds):
    """Compute the Silhouette Coefficient for each sample.
    The Silhoeutte Coefficient is a measure of how well samples are clustered
    with samples that are similar to themselves. Clustering models with a high
    Silhouette Coefficient are said to be dense, where samples in the same
    cluster are similar to each other, and well separated, where samples in
    different clusters are not very similar to each other.
    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (a) and the mean nearest-cluster distance (b) for each sample.
    The Silhouette Coefficient for a sample is ``(b - a) / max(a, b)``.
    This function returns the Silhoeutte Coefficient for each sample.
    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters.
    Parameters
    ----------
    X : array [n_samples_a, n_features]
        Feature array.
    labels : array, shape = [n_samples]
             label values for each sample
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by metrics.pairwise.pairwise_distances. If X is the distance
        array itself, use "precomputed" as the metric.
    `**kwds` : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
    Returns
    -------
    silhouette : array, shape = [n_samples]
        Silhouette Coefficient for each samples.
    References
    ----------
    Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
        Interpretation and Validation of Cluster Analysis". Computational
        and Applied Mathematics 20: 53-65. doi:10.1016/0377-0427(87)90125-7.
    http://en.wikipedia.org/wiki/Silhouette_(clustering)
    """

    flattened_labels = np.array([lab for labs in labels for lab in labs])

    X = np.concatenate(X)

    A = _intra_cluster_distances_block(X, flattened_labels, metric, n_jobs=n_jobs,
                                       **kwds)
    B = _nearest_cluster_distance_block(X, labels, flattened_labels, metric, n_jobs=n_jobs,
                                        **kwds)

    sil_samples = (B - A) / np.maximum(A, B)
    # nan values are for clusters of size 1, and should be 0
    return np.nan_to_num(sil_samples)


def _intra_cluster_distances_block_(subX, metric, **kwds):
    distances = pairwise_distances(subX, metric=metric, **kwds)
    return distances.sum(axis=1) / (distances.shape[0] - 1)


def _intra_cluster_distances_block(X, labels, metric, n_jobs=1, **kwds):
    """Calculate the mean intra-cluster distance for sample i.
    Parameters
    ----------
    X : array [n_samples_a, n_features]
        Feature array.
    labels : array, shape = [n_samples]
        label values for each sample
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by metrics.pairwise.pairwise_distances. If X is the distance
        array itself, use "precomputed" as the metric.
    `**kwds` : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
    Returns
    -------
    a : array [n_samples_a]
        Mean intra-cluster distance
    """
    intra_dist = np.zeros(labels.size, dtype=float)
    values = Parallel(n_jobs=n_jobs)(
            delayed(_intra_cluster_distances_block_)
                (X[np.where(labels == label)[0]], metric, **kwds)
                for label in np.unique(labels))
    for label, values_ in zip(np.unique(labels), values):
        intra_dist[np.where(labels == label)[0]] = values_
    return intra_dist


def _nearest_cluster_distance_block_(subX_a, subX_b, metric, **kwds):
    dist = pairwise_distances(subX_a, subX_b, metric=metric, **kwds)
    dist_a = dist.mean(axis=1)
    return dist_a


def _nearest_cluster_distance_block(X, labels, flattened_labels, metric, n_jobs=1, **kwds):
    """Calculate the mean nearest-cluster distance for sample i.
    Parameters
    ----------
    X : array [n_samples_a, n_features]
        Feature array.
    labels : array, shape = [n_samples]
        label values for each sample
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by metrics.pairwise.pairwise_distances. If X is the distance
        array itself, use "precomputed" as the metric.
    `**kwds` : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
    X : array [n_samples_a, n_features]
        Feature array.
    Returns
    -------
    b : float
        Mean nearest-cluster distance for sample i
    """
    inter_dist = np.empty(flattened_labels.size, dtype=float)
    inter_dist.fill(np.inf)
    # Compute cluster distance between pairs of clusters
    unique_labels = np.unique(flattened_labels)

    labels_a = []
    labels_b = []

    for doc_idx, labs in enumerate(labels):
      first_lab = sorted(np.unique(flattened_labels))[0]
      last_lab = sorted(np.unique(flattened_labels))[-1]
      for lab in np.unique(labs):
        if lab==first_lab:
          labels_a.append(lab)
          labels_b.append(lab+1)
        elif lab==last_lab:
          labels_a.append(lab)
          labels_b.append(lab-1)
        else:
          labels_a.extend([lab, lab])
          labels_b.extend([lab+1, lab-1])

    labels_a = np.array(labels_a)
    labels_b = np.array(labels_b)

    values = Parallel(n_jobs=n_jobs)(
            delayed(_nearest_cluster_distance_block_)(
                X[np.where(flattened_labels == label_a)[0]],
                X[np.where(flattened_labels == label_b)[0]],
                metric, **kwds)
                for label_a, label_b in zip(labels_a, labels_b))

    for label_a, values_a in \
            zip(labels_a, values):

            indices_a = np.where(flattened_labels == label_a)[0]

            inter_dist[indices_a] = np.minimum(values_a, inter_dist[indices_a])
            del indices_a
    return inter_dist

def silhouette_segrefree(embeddings, labels, njobs=1, as_loss=True):
  """
  Use Silhouette Score as an embedding-based, reference-free metric
  for topic segmentation evaluation.

  Input:
    embeddings <- List of np.array: a list including a matrix for each document where all the embeddings for the document's sentences are stored.
    labels <- List of lists of integers: a list including a list of binary integers for each document, where 0 represents no topic boundary at the same index sentence and 1 indicates there is a topic boundary at that index (new topic from next sentence).
    njobs <- int: the number of jobs to parallelize the computation.
    as_loss <- bool: If True, transform the final output to lay in the 0-1 range, where 0 is the best score and 1 is the worst (i.e. transform in loss function).
  Output:
    SC <- float: the Silhouette Score for the given corpus.
  
  """
  label_counter = 0
  new_labels = []
  new_embeddings = []
  one_segment_docs = 0
  N=0
  for doc_idx, labs in enumerate(labels):
    new_labs = []
    if sum(labs)<2:
      one_segment_docs+=len(labs)
      continue
    new_embeddings.append(embeddings[doc_idx])
    for lab in labs:
      N+=1
      new_labs.append(label_counter)
      if lab:
        label_counter+=1
    new_labels.append(new_labs)

  SC = silhouette_score_block(embeddings, new_labels,  one_segment_docs,N, metric='euclidean', sample_size=None,
                          random_state=None, n_jobs=njobs)

  if as_loss:
    SC = 1-(SC+1)/2
  return SC

def ARP(segments,
        return_relative_proximities = False,
        dispersion_function = average_variance,
        n = 1,
        verbose = False,
        as_loss = True,
        correction_factor=False):
    """
    Code to calculate the average of the differences between next-segment-inter variance
    and intra of the embeddings inside ground
    truth segments. The analysis should highlight how much the embeddings
    in a segment are close to each other, while taking into consideration
    how close they are to the embeddings in the next segment. The final
    score goes from -1 (worst) to +1 (best), where a score of 0 means
    that intra and inter variances are the same (so, no difference between
    the two).

    Input:
        segments <- List: list of list containing the embeddings grouped for each ground truth segment
        same_length <- bool: if True, it enforces the segments to be used in calculating intra and inter variances to have the same length (i.e. it cuts the current and next segment to do so, when using the )
        return_relative_proximities <- bool: If True, return all the RP scores for each segment. Use for debugging.
        n <- int: the exponent of the final ARP score. Increasing the exponent (e.g. n=2) helps yielding scores that are more distant from each other.
        verbose <- bool: If True, print warning messages during running.
        as_loss <- bool: If True, transform the final output to lay in the 0-1 range, where 0 is the best score and 1 is the worst (i.e. transform in loss function).
    Output:
        if return_relative_proximities==False:
          float: the average variance difference score, as described above.
        else:
          list of floats: all the RP scores for each segment before averaging.
    """
    scores = []
    length_one_segments = [] # at the end of each document, we compute the score for the special case of single-sentence segments stored in this list (see problem 1a)
    average_intra = [] # at the end of each document, we compute the average intra-cluster variance to address problem 1a
    for doc_index, doc in enumerate(segments):
        if len(doc)<2:
          if verbose:
            print(f"Warning: document no segmentation found for document number {doc_index}: score for the document defaulting to -1")
          scores.append(-1)
          continue
        for index, seg in enumerate(doc):
            try:
                # problem 1: lengths differences. Forcing intravar and intervar to be computed over same length segments-->no bias from standard deviation
                # problem 1a: the edge case of segments of length 1 are dealt with by defaulting to the average intra-cluster variance in the segment compared against the variance between the single intra-cluster embedding and the next one.
                if len(seg)==1:
                  length_one_segments.append(dispersion_function(np.append(seg, doc[index+1][:1], axis=0)))
                  continue

                else:
                  cut_point = min(len(doc[index+1])*2, len(seg))

                  intra = dispersion_function(seg[len(seg)-cut_point:])
                  average_intra.append(intra)

                  if isinstance(seg, list):
                      concat = seg[-cut_point//2:] + doc[index+1][:cut_point//2]
                  else:
                      concat = np.append(seg[-cut_point//2:], doc[index + 1][:cut_point//2], axis = 0)

            except IndexError:
                continue
            
            inter = dispersion_function(concat)

            # problem 2: scale dependence. Substitute the subtraction with the natural logarithm of the ratio intervar/intravar (see below)
            # ln(inter>intra)>0-->inf; ln(inter<intra)<0-->-inf; ln(inter==intra)==0
            # tanh can then be left to normalise between -1 and 1 (same variance should mean 0 anyway)

            scores.append((inter**n-intra**n)/(inter**n+intra**n))

        if len(average_intra):
          # below we compute the score for each one-sentence segments
          average_intra = np.nanmean(average_intra)
        else:
          # in the case every segment is a one-sentence segment, then we compute the average intra-segmental distance as the average inter-segmental distance instead (the overall document score will then tend to 0, which reflects the uncertainty of the algorithm in this case)
          average_intra = np.nanmean(length_one_segments)
        for inter in length_one_segments:
          scores.append((inter**n-average_intra**n)/(inter**n+average_intra**n))

        average_intra = []
        length_one_segments = []

    if return_relative_proximities:
      return scores

    # If returning ARP scores as a loss function, we first translate it in the range 0-1 and then we subtract it from 1
    if as_loss:
      return 1-(np.nanmean(scores)+1)/2

    # problem 3: independence. The independence assumption should be made, nonetheless. Think of an exponentially increasing embedding: that would tend towards 1 no matter what information it encodes.
    # shuffling the segments might limit this effect, but at the end that would mean that order doesn't matter anyway so an independence assumprion needs to be stated.
    return np.nanmean(scores)

class RefreeMetric:
  def __init__(self, encoder = "all-mpnet-base-v2"):
        """
        Arguments:
          score_function--> str: one of ["pairwise_cosine", "standard_deviation", "average_cosine"], as described in the original paper. "pairwise_cosine" is the one that usually perform best and the default choice.
          encoder--> str: the name of an available model from sentence-transformers or from huggingface. The name should be the same as found on huggingface_hub or the init function wil throw an error. If you want to pass embeddings directly to ARP (without computing them previously), the encoder argument can be set to None.
        """
        if encoder is None:
          self.encoder = None
        else:
          try:
            self.encoder = ST_ENCODER(encoder)
          except:
            self.encoder = BERT_BASE_ENCODER(encoder)

    def encode_corpus(self, corpus):
        """
        Arguments:
          corpus--> List of lists: a python list including lists of sentences (or any unit of text) where each list of sentences correspond to a separate document. 
        Output:
          test_embeddings--> List of numpy arrays: a python list including a 2D numpy array for each document in the corpus (where the first dimension correspond to the sentence in the document and the second is the embedding dimension). 
        """

        test_embeddings = []
        for doc in corpus:
          test_embeddings.append(self.encoder.encode(doc))
        
        return test_embeddings

  def prepare_segments(self, corpus, segmentation, embeddings=None):
        """
        Arguments:
          corpus--> List of lists: a python list including lists of sentences (or any unit of text) where each list of sentences correspond to a separate document. 
          segmentation --> List of lists: a python list including lists of binary integers representing whether the sentence in the document at the given position ends a topic (1) or not (0).
          embeddings--> List of numpy arrays: a python list including a 2D numpy array for each document in the corpus (where the first dimension correspond to the sentence in the document and the second is the embedding dimension). If included, skip the encoding part and pass custom embeddings.
        Output:
           segmented_embeddings --> List of numpy arrays: the embedding matrices resulting from aggregating the embeddings in each segment.
        """
        
        if embeddings is None:
          embeddings = self.encode_corpus(corpus)
        
        segmented_embeddings = []
        for doc_index, e in enumerate(embeddings):
            doc = []
            index1 = 0
            for index2, lab in enumerate(segmentation[doc_index]):
                if lab:
                  doc.append(embeddings[doc_index][index1:index2+1])
                  index1 = index2+1
            segmented_embeddings.append(doc)
        return segmented_embeddings


class ARPMetric(RefreeMetric):
    def __init__(self, score_function="pairwise_cosine", encoder = "all-mpnet-base-v2"):
        """
        Arguments:
          score_function--> str: one of ["pairwise_cosine", "standard_deviation", "average_cosine"], as described in the original paper. "pairwise_cosine" is the one that usually perform best and the default choice.
          encoder--> str: the name of an available model from sentence-transformers or from huggingface. The name should be the same as found on huggingface_hub or the init function wil throw an error. If you want to pass embeddings directly to ARP (without computing them previously), the encoder argument can be set to None.
        """
        super()__init__(encoder=encoder)
        if score_function not in ("pairwise_cosine", "standard_deviation", "average_cosine"):
            raise ValueError('The value of score function should be one of "pairwise_cosine", "standard_deviation" or "average_cosine"!')
        
        if score_function=="pairwise_cosine":
          self.score_fn = average_pairwise_similarity
        elif score_function=="standard_deviation":
          self.score_fn = average_variance
        else:
          self.score_fn = cosine_dispersion

    def evaluate_segmentation(self, corpus, segmentation, 
                              embeddings=None, output_all_scores=False,
                              verbose = False):
        """
        Arguments:
          corpus--> List of lists: a python list including lists of sentences (or any unit of text) where each list of sentences correspond to a separate document. 
          segmentation --> List of lists: a python list including lists of binary integers representing whether the sentence in the document at the given position ends a topic (1) or not (0).
          embeddings--> List of numpy arrays: a python list including a 2D numpy array for each document in the corpus (where the first dimension correspond to the sentence in the document and the second is the embedding dimension). If included, skip the encoding part and pass custom embeddings.
          output_all_scores--> Boolean: whether to output all the scores for each input document in the corpus or just the average. Default to False (i.e. just the average).
          verbose--> Boolean: whether to print out warnings and other information in the process of computing the metric. Default to False.
        Output:
           score--> float: the ARP score for the corpus.
        """

        segmented_embeddings = self.prepare_segments(corpus, segmentation, embeddings)

        score = ARP(segmented_embeddings, dispersion_function = self.score_fn, return_relative_proximities=output_all_scores, verbose=verbose)

        if output_all_scores:
            return np.nanmean(score), score
        return score

class SegReFreeMetric(RefreeMetric):
    def __init__(self, encoder = "all-mpnet-base-v2", 
                max_choice=True, correction_factor=True,
                negative=False, bounded=False, default_cap=10,
                no_backoff=False):
        """
        Arguments:
          encoder--> str: the name of an available model from sentence-transformers or from huggingface. The name should be the same as found on huggingface_hub or the init function wil throw an error. If you want to pass embeddings directly to ARP (without computing them previously), the encoder argument can be set to None.
          max_choice --> bool: if True, it includes the max pooling between the two adjacent topic segments, else it uses just the following for comparison (same as ARP).
          correction_factor --> bool: if True, it applies a correction factor for de-biasing the metric with respect to short segments. If False, the metric is the same as the Davies-Bouldin Index.
          negative --> bool: If True, outputs the negative SegReFree score. Use just for debugging.
          bounded --> bool: If True, normalise the scores with sigmoid, so that the metric is bounded between 0 and 1
          default_cap --> int|"auto": a positive integer or the string "auto" determining a default value to assign to the document-level score in the cases of metrics' failure (i.e. no topic boundary or each segment has length 1). If "auto" is used, then it automatically assign to these cases the highest score among the ones in the corpus (use just if having more than one document).
          no_backoff --> bool: If True, the metric will assign R=0 to each segment of length 1, as in the original paper. This is not recommended as it yields results that are stongly skewed towards favoring segments of length 1.
        """
        super()__init__(encoder=encoder)

    def evaluate_segmentation(self, corpus, segmentation, 
                              embeddings=None, return_all_scores_and_means=False,
                              verbose = False):
        """
        Arguments:
          corpus--> List of lists: a python list including lists of sentences (or any unit of text) where each list of sentences correspond to a separate document. 
          segmentation --> List of lists: a python list including lists of binary integers representing whether the sentence in the document at the given position ends a topic (1) or not (0).
          embeddings--> List of numpy arrays: a python list including a 2D numpy array for each document in the corpus (where the first dimension correspond to the sentence in the document and the second is the embedding dimension). If included, skip the encoding part and pass custom embeddings.
          return_all_scores_and_means --> bool: If True, return all the R scores and the segments' centroids. Use for debugging.
          verbose--> Boolean: whether to print out warnings and other information in the process of computing the metric. Default to False.
        Output:
           score--> float: the SegReFree score for the corpus.
        """

        segmented_embeddings = self.prepare_segments(corpus, segmentation, embeddings)

        score = SegReFree(segmented_embeddings, return_relative_proximities=output_all_scores, verbose=verbose)

        if return_all_scores_and_means:
            return score, scores_and_means
        return score

class SilhouetteMetric(RefreeMetric):
    def __init__(self, encoder = "all-mpnet-base-v2", njobs=1):
        """
        Arguments:
          encoder--> str: the name of an available model from sentence-transformers or from huggingface. The name should be the same as found on huggingface_hub or the init function wil throw an error. If you want to pass embeddings directly to ARP (without computing them previously), the encoder argument can be set to None.
          njobs--> int: the number of separate jobs to perform the silhouette calculation (using multiprocessing package).
        """
        super()__init__(encoder=encoder)
        self.n_job=njobs

    def evaluate_segmentation(self, corpus, segmentation, 
                              embeddings=None):
        """
        Arguments:
          corpus--> List of lists: a python list including lists of sentences (or any unit of text) where each list of sentences correspond to a separate document. 
          segmentation --> List of lists: a python list including lists of binary integers representing whether the sentence in the document at the given position ends a topic (1) or not (0).
          embeddings--> List of numpy arrays: a python list including a 2D numpy array for each document in the corpus (where the first dimension correspond to the sentence in the document and the second is the embedding dimension). If included, skip the encoding part and pass custom embeddings.
        Output:
           score--> float: the Silhouette score for the corpus.
        """

        segmented_embeddings = self.prepare_segments(corpus, segmentation, embeddings)

        score = silhouette_segrefree(segmented_embeddings, segmentation, njobs=selgf.n_job)

        return score

