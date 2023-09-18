from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModel
import torch
import warnings
warnings.filterwarnings('ignore')

def expand_label(labels,sentences):
  new_labels = [0 for i in range(len(sentences))]
  for i in labels:
    new_labels[i] = 1
  return new_labels

class BERT_BASE_ENCODER:
    def __init__(self, model="bert-base-uncased", pooling="average"):
            self.bert = AutoModel.from_pretrained(model)
            self.pool=self.cls_pooling if pooling=="cls" else self.avg_pooling
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.model = 'bert_cls_token'
            self.max_length = 512
            if torch.cuda.is_available():
                self.device = 'cuda'
                self.bert.to('cuda')
                print('Moved BERT to gpu!')
            else:
                print('No gpu is being used')
                self.device = 'cpu'

    def cls_pooling(self, model_output, attention_mask):
      return model_output[0][:,0]

    def avg_pooling(self, model_output, attention_mask):
      input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output[-1].size()).float()
      sum_embeddings = torch.sum(model_output[-1] * input_mask_expanded, 1)

      sum_mask = input_mask_expanded.sum(1)

      sum_mask = torch.clamp(sum_mask, min=1e-9)

      return sum_embeddings / sum_mask

    def encode(self, sentences, convert_to_tensor=False, batch_size=32):

        all_embeddings = []

        length_sorted_idx = np.argsort([-len(sen.split()) for sen in sentences])

        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

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
                    model_output = self.bert(input_ids=encoded_input['input_ids'].to(self.device),
                                             attention_mask=encoded_input['attention_mask'].to(self.device),
                                             output_hidden_states=True).hidden_states

                model_output = self.pool(model_output,
                                         encoded_input['attention_mask'].to(self.device)).detach().cpu()

                all_embeddings.extend(model_output)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            return torch.stack(all_embeddings)
        else:
            return np.asarray([emb.numpy() for emb in all_embeddings])

    def get_sentence_embedding_dimension(self):
        return 768

class ST_ENCODER:
  def __init__(self, model = "all-mpnet-base-v2"):
    self.model = SentenceTransformer(model)
  def encode(self, sentences, convert_to_tensor = False):
    if convert_to_tensor:
      return self.model.encode(sentences, pt=True)
    else:
      return self.model.encode(sentences)

  def get_sentence_embedding_dimension(self):
    return 768