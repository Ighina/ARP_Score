import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from ..embedding_metrics import *
from ..traditional_metrics import *

import os
from decimal import Decimal

def encode_corpus(test_data, encoder):
  test_embeddings = []
  for w in test_data:
    test_embeddings.append(encoder.encode(w[0]))
  return test_embeddings

def ARP_evaluate(test_data, test_embeddings, ground_truth=None, correct_disease_error=False, probe_segrefree=False, skip_noseg=True):
  results = {}
  if ground_truth is not None:
    results_topicseg = {"pk":[], "windowdiff":[], "b":[]}
  segments = []
  random_segments = []
  lengths = []
  labels = [w[1] for w in test_data]
  for doc_idx, labs in enumerate(labels):

      if ground_truth is not None:
        if correct_disease_error:
          if len(ground_truth[doc_idx][1])>len(labs) and (len(ground_truth[doc_idx][1])-len(labs))<15:
            gt = ground_truth[doc_idx][1][:len(labs)]
          else:
            gt = ground_truth[doc_idx][1]
        else:
          gt = ground_truth[doc_idx][1]
        if sum(labs)==1 and labs[-1]==1:
          if skip_noseg:
            continue
          results_topicseg["pk"].append(Decimal(1.0)) # the case of no segmentation in our experiments is treated as system failure (==Boundary Similarity)
          results_topicseg["windowdiff"].append(Decimal(1.0))
        else:
          results_topicseg["pk"].append(compute_Pk(labs, gt))
          try:
            results_topicseg["windowdiff"].append(compute_window_diff(labs, gt))
          except:
            results_topicseg["windowdiff"].append(compute_Pk(labs, gt))

        results_topicseg["b"].append(B_measure(labs, gt)[-1])
      doc = []
      doc_r = []
      index1 = 0
      index3 = 0
      for index2, lab in enumerate(labs):
        if lab:
          doc.append(test_embeddings[doc_idx][index1:index2+1])
          lengths.append(len(doc[-1]))
          index1 = index2+1
      segments.append(doc)

  if probe_segrefree:
    return SegReFree(segments, return_all_scores_and_means=True)

  functions = ["Average Variance", "Cosine Dispersion", "Average Pairwise Cosine"]
  f = 0
  print("ARP with different functions and n = 1:")
  for i in [average_variance, cosine_dispersion, average_pairwise_similarity]:
    score = ARP(segments, dispersion_function = i)
    results[f"ARP_{functions[f]}_1"]=score
    print(f'ARP score with dispersion function={functions[f]}: {score}')
    f+=1

  f = 0
  print("ARP with different functions and n = 2:")
  for i in [average_variance, cosine_dispersion, average_pairwise_similarity]:
    score = ARP(segments, dispersion_function = i, n = 2)
    results[f"ARP_{functions[f]}_2"]=score
    print(f'ARP score with dispersion function={functions[f]}: {score}')
    f+=1

  score = SegReFree(segments)
  if not isinstance(score, float):
    score=score[0][0]
  
  results["SegReFree"]=score
  print(f'SegReFree score: {score}')

  score=silhouette_segrefree(test_embeddings, labels, njobs=8)

  results["SilhouetteScore"]=score
  print(f'Silhouette score: {score}')

  if ground_truth is not None:
    return results, {k:np.mean(v) for k, v in results_topicseg.items()}
  return results

def group_sentences(data, labels, threshold = 0.5, ce=False):
  new_data = []
  for index, t in enumerate(labels):
    if ce:
      new_data.append([data[index][0], [int(l[1]>0.5) for l in t], index])
    else:
      new_data.append([data[index][0], [int(l[0]>0.5) for l in t], index])
  return new_data

def boundary_removal(labels, k=1):
  tot_splits = sum(labels)
  new_labels = labels[:]
  if k>tot_splits:
    return [0 for x in range(len(labels))]
  else:
    for _ in range(k):
      new_labels[np.random.choice([j for j,x in enumerate(new_labels) if x])]=0
  return new_labels

def boundary_transposition(labels, k=1):
  boundaries = [j for j,x in enumerate(labels) if x]
  new_labels = labels[:]
  for b in boundaries:
    new_labels[b]=0
    try:
      new_labels[b+k]=1
    except IndexError:
      try:
        new_labels[b-k]=1
      except IndexError:
        pass
  return new_labels

def Synthetic_evaluation(test_data, test_embeddings, k_range=10):
  results = {"removal":{'ARP_Average Variance_1':[],
    'ARP_Cosine Dispersion_1':[],
    'ARP_Average Pairwise Cosine_1':[],
    'ARP_Average Variance_2':[],
    'ARP_Cosine Dispersion_2':[],
    'ARP_Average Pairwise Cosine_2':[],
    'SegReFree':[],
    "Average_Segment_Length":[],
    "SilhouetteScore":[]},
             "transposition":{'ARP_Average Variance_1':[],
    'ARP_Cosine Dispersion_1':[],
    'ARP_Average Pairwise Cosine_1':[],
    'ARP_Average Variance_2':[],
    'ARP_Cosine Dispersion_2':[],
    'ARP_Average Pairwise Cosine_2':[],
    'SegReFree':[],
                              "Average_Segment_Length":[],
                              "SilhouetteScore":[]
                              }}

  results_topseg = {"removal":{"pk":[], "windowdiff":[], "b":[]},
                    "transposition":{"pk":[], "windowdiff":[], "b":[]}}

  removal_test_data = {k:[] for k in range(k_range)}
  for t in test_data:
    for k in range(k_range):
      removal_test_data[k].append((t[0], boundary_removal(t[1], k = k)))
  
  transposition_test_data = {k:[] for k in range(k_range)}
  for t in test_data:
    for k in range(k_range):
      transposition_test_data[k].append((t[0], boundary_transposition(t[1], k = k)))

  for t in removal_test_data.values():
    r, top = ARP_evaluate(t, test_embeddings, test_data)

    for key in results["removal"]:
      results["removal"][key].append(r[key])

    for key in top:
      results_topseg["removal"][key].append(top[key])

  for t in transposition_test_data.values():
    r, top = ARP_evaluate(t, test_embeddings, test_data)

    for key in results["transposition"]:
      results["transposition"][key].append(r[key])

    for key in top:
      results_topseg["transposition"][key].append(top[key])

  return results, results_topseg

def rearrange_QMSUM(test_data, test_scores):
  new_t = []
  for t in test_scores:
    for d in test_data:
      if len(d[1])==len(t):
        d[1][-1] = 1
        new_t.append(d)
        break
  return new_t

def rearrange_embeddings(test_embeddings, test_scores):
  new_t = []
  for t in test_scores:
    for d in test_embeddings:
      if d.shape[0]==len(t):
        new_t.append(d)
        break
  return new_t

def true_systems_evaluation(test_data, test_embeddings, dataset):
  np.random.seed(100)
  np.random.state = 100
  if dataset=="QMSUM":
    encoder_data = "QMSUM"
  elif dataset=="AudioBBC":
    encoder_data = "radionews"
  elif dataset=="wikisection_en_city":
    encoder_data = "city"
  elif dataset=="wikisection_en_disease":
    encoder_data = "disease"

  systems = [f"BiLSTM_{dataset}_bs8_all-MiniLM-L12-v2_Pk_BinaryCrossEntropy_normal",
             f"BiLSTM_{dataset}_bs8_roberta_base-last_mean_Pk_BinaryCrossEntropy_normal",
             f"BiLSTM_{dataset}_bs8_roberta_topseg_mean_{encoder_data}_Pk_BinaryCrossEntropy_normal",
             f"SheikhBiLSTM_{dataset}_bs8_all-MiniLM-L12-v2_Pk_BinaryCrossEntropy",
             f"SheikhBiLSTM_{dataset}_bs8_roberta_base-last_mean_Pk_BinaryCrossEntropy",
             f"SheikhBiLSTM_{dataset}_bs8_roberta_topseg_mean_{encoder_data}_Pk_BinaryCrossEntropy",
             f"SheikhTransformer_{dataset}_bs16_all-MiniLM-L12-v2_Pk_CrossEntropy_2_big",
             f"SheikhTransformer_{dataset}_bs16_roberta_base-last_mean_Pk_CrossEntropy_2_big",
             f"SheikhTransformer_{dataset}_bs16_roberta_topseg_mean_{encoder_data}_Pk_CrossEntropy_2_big",
             f"ground_truth",
             f"random_k",
             f"random_u2",
             f"random_u3",
             f"random_u4",
             f"random_u5",
             f"random_u6",
             f"random_u7",
             f"random_u8",
             f"random_u9"]

  row_names = ["DotBiLSTM_minilm", "DotBiLSTM_roberta", "DotBiLSTM_topseg",
               "BiLSTM_minilm", "BiLSTM_roberta", "BiLSTM_topseg",
               "Transformer_minilm", "Transformer_roberta", "Transformer_topseg",
               "ground_truth", "random_k", "random_u2", "random_u4", "random_u6", "random_u8", "random_u10"]

  results = {'ARP_Average Variance_1':[],
    'ARP_Cosine Dispersion_1':[],
    'ARP_Average Pairwise Cosine_1':[],
    'ARP_Average Variance_2':[],
    'ARP_Cosine Dispersion_2':[],
    'ARP_Average Pairwise Cosine_2':[],
    'SegReFree':[],
             "Average_Segment_Length":[],
             "SilhouetteScore":[]}

  results_topseg = {"pk":[], "windowdiff":[], "b":[]}

  rear = True

  for sys in systems:
    print(sys)
    correct_disease_error = False
    if dataset.startswith("wikisection"):
      correct_disease_error = True

    if sys=="ground_truth":
      correct_disease_error = False
      test = test_data
    elif sys=="random_k":
      correct_disease_error = False
      test = []
      for t in test_data:
        avg_k = np.mean(t[1])
        test.append([t[0], [np.random.binomial(1, p=avg_k) for _ in range(len(t[0]))],None])
    elif sys.startswith("random_u"):
      correct_disease_error = False
      test = []
      u=int(sys[-1])
      if not u:
        u=10
      p=[1-(1/u), 1/u]
      for t in test_data:
        test.append([t[0], [np.random.choice([0,1], p=p) for _ in range(len(t[0]))], None])
    else:
      with open(os.path.join(sys, "all_test_scores.pkl"), "rb") as f:
        test = pickle.load(f)

      for i in range(len(test)):
        test[i][-1][-1]=1

      if dataset=="QMSUM" and rear:
        rear = False
        test_data = rearrange_QMSUM(test_data, test)
        test_embeddings = rearrange_embeddings(test_embeddings, test)
      if len(test[0][0])==2:
        ce=True
      elif len(test[0][0])==1:
        ce=False
      test = group_sentences(test_data, test, ce=ce)
    try:
      num_segments = 0

      for i in range(len(test)):
        test[i][1][-1]=1

        num_segments+=sum(test[i][1][:-1])

      if not num_segments:
        print("No Segmentations for system: " + sys)
        continue

      r, top = ARP_evaluate(test, test_embeddings, test_data, correct_disease_error=correct_disease_error)
    except ZeroDivisionError:
      return test, test_embeddings, test_data
    for key in results:
      results[key].append(r[key])
    for key in top:
      results_topseg[key].append(top[key])

  return results, results_topseg, row_names

def plot_synthetic_data(df, x_label='Number of Removed Boundaries', figsize=(15,5), scale = True, negative=False):
  fig, ax = plt.subplots(figsize=figsize)
  if negative:
    ax.set_ylabel('Relative Performance Degradation')
  else:
    ax.set_ylabel('Scaled Loss Score')
  ax.set_xlabel(x_label)

  if scale:
    scaled_df = df.copy()
    scaled_df.iloc[:,:] = MinMaxScaler().fit_transform(df)
  else:
    scaled_df = df

  rearranged_df = {"metric":[], "value":[], "x":[]}
  for key in scaled_df:
    rearranged_df["metric"].extend([key for _ in range(len(scaled_df))])
    if negative:
      rearranged_df["value"].extend((1-scaled_df[key].values).tolist())
    else:
      rearranged_df["value"].extend(scaled_df[key].values.tolist())
    rearranged_df["x"].extend([x for x in range(len(scaled_df))])
  rearranged_df = pd.DataFrame(rearranged_df)

  sns.lineplot(
    data=rearranged_df,
    x="x", y="value", hue="metric", style="metric",
    markers=True, dashes=False
  )