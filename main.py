import argparse
import os
import sys

import pandas as pd

from utils.evaluation_fn import *
from utils.load_datasets import load_dataset

class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
            
    
parser = MyParser(
        description = 'Run evaluation experiments on the given dataset with the given encoder')

parser.add_argument("--dataset", "-d", type=str, required=True, choices=["wikisection_en_city", "wikisection_en_disease", "QMSUM", "AudioBBC"], 
                    help="The dataset to be used in the evaluation experiments.")

parser.add_argument("--encoder", "-e", type=str, required=True, choices=["roberta_base", "falcon", "MPNET"],
                    help="The sentence encoder to be used in the evaluation experiments.")

parser.add_argument("--include_davies_bouldin_index", "-db", action="store_true",
                    help="Whether to include among the metrics also the unnormalised Davies-Bouldin Index.")

args = parser.parse_args()


model_name = args.encoder
dataset = args.dataset

data = load_dataset(dataset)

if model_name=="falcon":
    try:
        with open(os.path.join("falcon_embeddings", dataset+"_test_embeddings.pkl"), "rb") as f:
            test_embeddings = pickle.load(f)
    except FileNotFoundError:
        print('In order to run the falcon experiments, you first need  to run the "extract_embeddings_with_falcon.py" function, passing the relevant dataset.')
        sys.exit(0)
else:
    if model_name=="roberta-base":
        encoder = BERT_BASE_ENCODER("roberta-base")
    elif model_name=="MPNET":
        encoder = ST_ENCODER()
    test_embeddings = encode_corpus(data[0][1], encoder)
    del(encoder)

print(f"Evaluating embedding metrics on {dataset} with encoder model: {model_name}")
print("Synthetic Evaluation...")

synth_results, synth_topresults = Synthetic_evaluation(data[0][1], test_embeddings, k_range=10)

df_removal = pd.DataFrame({"ARP":np.array(synth_results["removal"]["ARP_Average Variance_1"]),
                   "SegReFree":np.array(synth_results["removal"]["SegReFree"]),
                           "CosineDispersion":np.array(synth_results["removal"]["ARP_Cosine Dispersion_1"]),
                    "CosinePairwise":np.array(synth_results["removal"]["ARP_Average Pairwise Cosine_1"]),
                   "Pk":[float(pk) for pk in synth_topresults["removal"]["pk"]],
                   "WD":[float(wd) for wd in synth_topresults["removal"]["windowdiff"]],
                   "B":[1-b for b in synth_topresults["removal"]["b"]]})

df_trans = pd.DataFrame({"ARP":np.array(synth_results["transposition"]["ARP_Average Variance_1"]),
                   "SegReFree":np.array(synth_results["transposition"]["SegReFree"]),
                         "CosineDispersion":np.array(synth_results["transposition"]["ARP_Cosine Dispersion_1"]),
                    "CosinePairwise":np.array(synth_results["transposition"]["ARP_Average Pairwise Cosine_1"]),
                   "Pk":[float(pk) for pk in synth_topresults["transposition"]["pk"]],
                   "WD":[float(wd) for wd in synth_topresults["transposition"]["windowdiff"]],
                   "B":[1-b for b in synth_topresults["transposition"]["b"]]})

if args.include_davies_bouldin_index:
    df_removal["Davies-Bouldin"] = np.array(synth_results["removal"]["Davies-Bouldin"])
    df_trans["Davies-Bouldin"] = np.array(synth_results["transposition"]["Davies-Bouldin"])

removal_plot = plot_synthetic_data(df_removal, figsize=(10,8))

transposition_plot = plot_synthetic_data(df_trans,  "Number of Transposed Boundaries", figsize=(10,8))

removal_plot.savefig(os.path.join("output_figures", f"removal_plot_{dataset}_{model_name}.jpg"))
transposition_plot.savefig(os.path.join("output_figures", f"transposition_plot_{dataset}_{model_name}.jpg"))

print("Real Systems Evaluation...")
real_results, real_topresults, row_names = true_systems_evaluation(data[0][1], test_embeddings, dataset)

import pandas as pd

df_real = pd.DataFrame({"ARP":np.array(real_results["ARP_Average Variance_1"]),
                   "SegReFree":np.array(real_results["SegReFree"]),
                    "CosineDispersion":np.array(real_results["ARP_Cosine Dispersion_1"]),
                    "CosinePairwise":np.array(real_results["ARP_Average Pairwise Cosine_1"]),
                        "AverageSegmentLength": np.array(real_results["Average_Segment_Length"]),
                   "Pk":[float(pk) for pk in real_topresults["pk"]],
                   "WD":[float(wd) for wd in real_topresults["windowdiff"]],
                   "B":[1-b for b in real_topresults["b"]]})

if args.include_davies_bouldin_index:
    df_real["Davies-Bouldin"] = np.array(real_results["Davies-Bouldin"])

df_corr = df_real.corr()

df_corr.to_csv(os.path.join("output_correlations", f"correlations_{dataset}_{model_name}.csv"), index=False)