import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

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

def SegReFreeScore(segments, correction_factor=True, negative=False):
  sign=-1 if negative else 1
  mean = np.nanmean(segments, axis = 0).reshape(1, -1)
  if len(segments)>1:
    if correction_factor:
      return mean, np.nanmean(euclidean_distances(mean, segments))/(1-(1/len(segments)**0.5))*sign
    else:
      return mean, np.nanmean(euclidean_distances(mean, segments))*sign
  else:
    return mean, np.array([[5]])*sign

def SegReFree(segments, verbose=False, max_choice = True, correction_factor=True, negative=False):
  """
    Code to calculate the SegReFree score as described in https://openreview.net/forum?id=NVM6eCm69eu.

    Input:
        segments <- List: list of list containing the embeddings grouped for each ground truth segment
        max_choice <- bool: if True, it includes the max pooling between the two adjacent topic segments, else it uses just the following for comparison (same as ARP).
        correction_factor <- if True, it applies a correction factor for de-biasing the metric with respect to short segments. If False, the metric is the same as the Davies-Bouldin Index.
        negative <- If True, outputs the negative SegReFree score. Use just for debugging. 
    Output:
        float: the SegReFree score.

    """
  all_scores = []
  sign=-1 if negative else 1
  for doc_index, doc in enumerate(segments):
        if len(doc)<2:
          if verbose:
            print(f"Warning: document no segmentation found for document number {doc_index}: score for the document defaulting to 0")
          all_scores.append(np.array([[5]])*sign)
          continue
        segment_S = []
        for index, seg in enumerate(doc):
          S = SegReFreeScore(seg, correction_factor, negative)
          segment_S.append(S)
          
        for index, (mean_S, S) in enumerate(segment_S):
          if max_choice:
            if index>0 and index<len(segment_S)-1:
              R_prev = (S+segment_S[index-1][1])/euclidean_distances(mean_S, segment_S[index-1][0])
              R_fol = (S+segment_S[index+1][1])/euclidean_distances(mean_S, segment_S[index+1][0])
              R = max(R_prev, R_fol)
            elif index:
              R = (S+segment_S[index-1][1])/euclidean_distances(mean_S, segment_S[index-1][0])
            else:
              R =  (S+segment_S[index+1][1])/euclidean_distances(mean_S, segment_S[index+1][0])
          else:
            try:
              R =  (S+segment_S[index+1][1])/euclidean_distances(mean_S, segment_S[index+1][0])
            except IndexError:
              pass
          
          all_scores.append(R)
  return np.nanmean(all_scores)


def ARP(segments,
        return_relative_proximities = False,
        dispersion_function = average_variance,
        n = 1,
        verbose = False,
        as_loss = True):
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
    Output:
        float: the average variance difference score, as described above.

    """
    scores = []
    length_one_segments = [] # at the end of each document, we compute the score for the special case of single-sentence segments stored in this list (see problem 1a)
    average_intra = [] # at the end of each document, we compute the average intra-cluster variance to address problem 1a
    for doc_index, doc in enumerate(segments):
        if len(doc)<2:
          if verbose:
            print(f"Warning: document no segmentation found for document number {doc_index}: score for the document defaulting to 0")
          scores.append(0)
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

            # scores.append(np.tanh(np.log(inter/intra))) # Equivalent to below
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

class ARP_metric:
    def __init__(self, score_function="pairwise_cosine", encoder = "all-mpnet-base-v2"):
        """
        Arguments:
          score_function--> str: one of ["pairwise_cosine", "standard_deviation", "average_cosine"], as described in the original paper. "pairwise_cosine" is the one that usually perform best and the default choice.
          encoder--> str: the name of an available model from sentence-transformers or from huggingface. The name should be the same as found on huggingface_hub or the init function wil throw an error. If you want to pass embeddings directly to ARP (without computing them previously), the encoder argument can be set to None.
        """
        if score_function not in ("pairwise_cosine", "standard_deviation", "average_cosine"):
            raise ValueError('The value of score function should be one of "pairwise_cosine", "standard_deviation" or "average_cosine"!')
        
        if score_function=="pairwise_cosine":
          self.score_fn = average_pairwise_similarity
        elif score_function=="standard_deviation":
          self.score_fn = average_variance
        else:
          self.score_fn = cosine_dispersion

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

        score = ARP(segmented_embeddings, dispersion_function = self.score_fn, return_relative_proximities=output_all_scores, verbose=verbose)

        if output_all_scores:
            return np.nanmean(score), score
        return score

