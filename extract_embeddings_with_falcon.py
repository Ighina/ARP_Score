import os
import pickle
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from utils.load_datasets import load_dataset

class BERT_BASE_ENCODER:
    def __init__(self, enc, pool, opt=None, emb_diff=False, max_length = 512, trust_remote=False, falcon = False):
        if falcon:
            self.bert = AutoModel.from_pretrained(enc, trust_remote_code=trust_remote, cache_dir="./huggingface_models/falcon")
            self.tokenizer = AutoTokenizer.from_pretrained(enc, trust_remote_code=trust_remote, cache_dir="./huggingface_models/falcon")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.bert = AutoModel.from_pretrained(enc, trust_remote_code=trust_remote)
            self.tokenizer = AutoTokenizer.from_pretrained(enc, trust_remote_code=trust_remote)
        self.model = 'bert_cls_token'
        self.max_length = max_length
        if pool == "cls":
            self.pool = self.cls_pooling
        elif pool == "last_first_mean":
            self.pool = self.first_last_pooling
        elif pool == "last_mean":
            self.pool = self.last_pooling
        elif pool == "second_to_last_mean":
            self.pool = self.second_to_last_pooling  # the default in BERT as a service
        elif pool == "sep":
            self.pool = self.sep_pooling
        else:
            raise ValueError("Pooling strategy not recognised!")

        if not opt in ("pairwise", "combined"):
            print(
                "Warning: the optional configuration of using pairwise or combined encoding has not been properly formatted. If you wanted to use one of those two options you should have attached +pairwise/+combined to your model name. For now, the program will default to the usual encoding (single sentences encoding)!")

        self.opt = opt

        self.diff = False
        if emb_diff:
            self.diff = True

        if torch.cuda.is_available():
            self.device = 'cuda'
            self.bert.to('cuda')
            print('Moved model to gpu!')
        else:
            print('No gpu is being used')
            self.device = 'cpu'

    def cls_pooling(self, model_output, attention_mask):
        return model_output[-1][:, 0, :]

    def first_last_pooling(self, model_output, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output[-1].size()).float()

        last = torch.sum(model_output[-1] * input_mask_expanded, dim=1)
        first = torch.sum(model_output[1] * input_mask_expanded, dim=1)

        sum_mask = input_mask_expanded.sum(1)

        sum_mask = torch.clamp(sum_mask, min=1e-9)

        return (last + first) / sum_mask

    def last_pooling(self, model_output, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output[-1].size()).float()
        sum_embeddings = torch.sum(model_output[-1] * input_mask_expanded, 1)

        sum_mask = input_mask_expanded.sum(1)

        sum_mask = torch.clamp(sum_mask, min=1e-9)

        return sum_embeddings / sum_mask

    def second_to_last_pooling(self, model_output, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output[-2].size()).float()
        sum_embeddings = torch.sum(model_output[-2] * input_mask_expanded, 1)

        sum_mask = input_mask_expanded.sum(1)

        sum_mask = torch.clamp(sum_mask, min=1e-9)

        return sum_embeddings / sum_mask

    def sep_pooling(self, model_output, attention_mask):
        return model_output[-1][:, -1, :]

    def encode(self, sentences, convert_to_tensor=False, batch_size=32):

        if self.opt == "pairwise":
            sentences = [' [SEP] '.join([sentences[i], sentences[i + 1]]) for i in
                         range(len(sentences) - 1)] + [sentences[-1]]

        all_embeddings = []

        length_sorted_idx = np.argsort([-len(sen.split()) for sen in sentences])

        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        
        deleted = 0
        for start_index in range(0, len(sentences), batch_size):

            sentences_batch = sentences_sorted[start_index:start_index + batch_size]

            encoded_input = self.tokenizer(sentences_batch, padding=True, truncation=True, return_tensors='pt',
                                           max_length=self.max_length)

            with torch.no_grad():

                if encoded_input['input_ids'].shape[0] > 100:
                    pass

                try:
                    model_output = self.bert(input_ids=encoded_input['input_ids'].to(self.device),
                                         attention_mask=encoded_input['attention_mask'].to(self.device),
                                         output_hidden_states=True).hidden_states
                except RuntimeError:
                    self.bert.to(self.device)
                    try:
                        model_output = self.bert(input_ids=encoded_input['input_ids'].to(self.device),
                                             attention_mask=encoded_input['attention_mask'].to(self.device),
                                             output_hidden_states=True).hidden_states
                                             
                    
                    except RuntimeError:
                        #failed = "AAAAAAAAAH"
                        #print(len(all_embeddings))
                        # This is an exception encountered when using Falcon with QMSum, as it appears there are features of some sentences in that dataset which caused an unknown error in that model
                        all_embeddings.extend(torch.zeros((batch_size, model_output.shape[1])))
                        continue
                        #print(len(all_embeddings))
                        #length_sorted_idx = np.delete(length_sorted_idx, start_index-deleted)
                        #deleted+=1
                        #continue
                        
                model_output = self.pool(model_output,
                                         encoded_input['attention_mask'].to(self.device)).detach().cpu()

                all_embeddings.extend(model_output)

        try:
            all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        except IndexError:
            print(failed)
            0/0

        if self.opt == "combined":
            all_embeddings_pr = []

            sentences_pr = [' [SEP] '.join([sentences[i], sentences[i + 1]]) for i in
                            range(len(sentences) - 1)] + [sentences[-1]]

            length_sorted_idx_ = np.argsort([-len(sen.split()) for sen in sentences_pr])

            sentences_sorted = [sentences_pr[idx] for idx in length_sorted_idx]

            for start_index in range(0, len(sentences), batch_size):

                sentences_batch = sentences_sorted[start_index:start_index + batch_size]

                encoded_input = self.tokenizer(sentences_batch, padding=True, truncation=True,
                                               return_tensors='pt', max_length=self.max_length)

                with torch.no_grad():

                    if encoded_input['input_ids'].shape[0] > 100:
                        pass

                    model_output = self.bert(input_ids=encoded_input['input_ids'].to(self.device),
                                             attention_mask=encoded_input['attention_mask'].to(self.device))

                    model_output = self.cls_pooling(model_output, encoded_input['attention_mask'].to(
                        self.device)).detach().cpu()

                    all_embeddings_pr.extend(model_output)

            all_embeddings_pr = [all_embeddings_pr[idx] for idx in np.argsort(length_sorted_idx)]

            if convert_to_tensor:
                # if all_embeddings_pr.shape[0]!=all_embeddings.shape[0]:
                #     print(all_embeddings.shape)
                #     print(all_embeddings_pr.shape[0])
                #     raise ValueError()
                return torch.cat((torch.stack(all_embeddings), torch.stack(all_embeddings_pr)), axis=1)
            else:
                return np.concatenate((np.asarray([emb.numpy() for emb in all_embeddings]),
                                       np.asarray([emb.numpy() for emb in all_embeddings_pr])), axis=1)

        if convert_to_tensor:
            x = torch.stack(all_embeddings)
            if self.diff:
                return torch.cat((torch.diff(x, axis=0), x[-1].unsqueeze(0)), axis=0)
            return x
        else:
            x = np.asarray([emb.numpy() for emb in all_embeddings])
            if self.diff:
                return np.concatenate((np.diff(x, axis=0), x[-1].reshape(1, -1)))
            return x

    def get_sentence_embedding_dimension(self):
        return 768

def encode_corpus(test_data, encoder, batch_size=32):
  test_embeddings = []
  for w in test_data:
    test_embeddings.append(encoder.encode(w[0], batch_size=batch_size))
  return test_embeddings

data = load_dataset(sys.argv[1])
out_directory = "falcon_embeddings"

os.chdir(out_directory)

pool = "last_mean"
encoder = BERT_BASE_ENCODER("tiiuae/falcon-7b", pool, trust_remote=True, falcon=True)

if sys.argv[1]=="QMSUM":
    bs = 2
else:
    bs = 8

test_embeddings = encode_corpus(data[0][1], encoder, batch_size=bs)

with open(sys.argv[1]+"_test_embeddings.pkl", "wb") as f:
    pickle.dump(test_embeddings, f)

# print(test_embeddings[0].shape)

# np.save(sys.argv[1]+"_test_embeddings", np.stack(test_embeddings))
