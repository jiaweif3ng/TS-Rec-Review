import os
import numpy as np
import pandas as pd
from collections import deque
import torch.nn as nn
import torch
# import tensorflow as tf
import yaml
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase, TrainerCallback


def extract_axis_1(data, indices):
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, indices[i], :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res


def to_pickled_df(data_directory, **kwargs):
    for name, df in kwargs.items():
        df.to_pickle(os.path.join(data_directory, name + '.df'))

def pad_history(itemlist,length,pad_item):
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist


# def extract_axis_1(data, ind):
#     """
#     Get specified elements along the first axis of tensor.
#     :param data: Tensorflow tensor that will be subsetted.
#     :param ind: Indices to take (one for each element along axis 0 of data).
#     :return: Subsetted tensor.
#     """

#     batch_range = tf.range(tf.shape(data)[0])
#     indices = tf.stack([batch_range, ind], axis=1)
#     res = tf.gather_nd(data, indices)

#     return res


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

def calculate_hit(sorted_list,topk,true_items,rewards,r_click,total_reward,hit_click,ndcg_click,hit_purchase,ndcg_purchase):
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                total_reward[i] += rewards[j]
                if rewards[j] == r_click:
                    hit_click[i] += 1.0
                    ndcg_click[i] += 1.0 / np.log2(rank + 1)
                else:
                    hit_purchase[i] += 1.0
                    ndcg_purchase[i] += 1.0 / np.log2(rank + 1)


# class Memory():
#     def __init__(self):
#         self.buffer = deque()
#
#     def add(self, experience):
#         self.buffer.append(experience)
#
#     def sample(self, batch_size):
#         idx = np.random.choice(np.arange(len(self.buffer)),
#                                size=batch_size,
#                                replace=False)
#         return [self.buffer[ii] for ii in idx]

# NeuProcessEncoder
class NeuProcessEncoder(nn.Module):
    def __init__(self, input_size=64, hidden_size=64, output_size=64, dropout_prob=0.4, device=None):
        super(NeuProcessEncoder, self).__init__()
        self.device = device
        
        # Encoder for item embeddings
        layers = [nn.Linear(input_size, hidden_size),
                torch.nn.Dropout(dropout_prob),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, output_size)]
        self.input_to_hidden = nn.Sequential(*layers)

        # Encoder for latent vector z
        self.z1_dim = input_size # 64
        self.z2_dim = hidden_size # 64
        self.z_dim = output_size # 64
        self.z_to_hidden = nn.Linear(self.z1_dim, self.z2_dim)
        self.hidden_to_mu = nn.Linear(self.z2_dim, self.z_dim)
        self.hidden_to_logsigma = nn.Linear(self.z2_dim, self.z_dim)

    def emb_encode(self, input_tensor):
        hidden = self.input_to_hidden(input_tensor)

        return hidden

    def aggregate(self, input_tensor):
        return torch.mean(input_tensor, dim=-2)

    def z_encode(self, input_tensor):
        hidden = torch.relu(self.z_to_hidden(input_tensor))
        mu = self.hidden_to_mu(hidden)
        log_sigma = self.hidden_to_logsigma(hidden)
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z, mu, log_sigma
    
    def encoder(self, input_tensor):
        z_ = self.emb_encode(input_tensor)
        z = self.aggregate(z_)
        self.z, mu, log_sigma = self.z_encode(z)
        return self.z, mu, log_sigma

    def forward(self, input_tensor):
        self.z, _, _ = self.encoder(input_tensor)
        return self.z


class MemoryUnit(nn.Module):
    # clusters_k is k keys
    def __init__(self, input_size, output_size, emb_size, clusters_k=10):
        super(MemoryUnit, self).__init__()
        self.clusters_k = clusters_k
        self.input_size = input_size
        self.output_size = output_size
        self.array = nn.Parameter(init.xavier_uniform_(torch.FloatTensor(self.clusters_k, input_size*output_size)))
        self.index = nn.Parameter(init.xavier_uniform_(torch.FloatTensor(self.clusters_k, emb_size)))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, bias_emb):
        """
        bias_emb: [batch_size, 1, emb_size]
        """
        att_scores = torch.matmul(bias_emb, self.index.transpose(-1, -2)) # [batch_size, clusters_k]
        att_scores = self.softmax(att_scores)

        # [batch_size, input_size, output_size]
        para_new = torch.matmul(att_scores, self.array) # [batch_size, input_size*output_size]
        para_new = para_new.view(-1, self.output_size, self.input_size)

        return para_new

    def reg_loss(self, reg_weights=1e-2):
        loss_1 = reg_weights * self.array.norm(2)
        loss_2 = reg_weights * self.index.norm(2)

        return loss_1 + loss_2

def binary_weight_transform(nums, top_percent=100):
    sorted_nums = sorted(nums, reverse=True)
    
    threshold_index = int(len(nums) * top_percent / 100)
    
    num_to_value = {num: 1 if i < threshold_index else 0 for i, num in enumerate(sorted_nums)}
    
    transformed_list = [num_to_value[num] for num in nums]
    
    return transformed_list

def threshold_weight_transform(nums, upper_threshold=1, lower_threshold=-1):
    result = []
    for num in nums:
        if num > upper_threshold:
            result.append(upper_threshold)
        elif num < lower_threshold:
            result.append(lower_threshold)
        else:
            result.append(num)
    return result
    
def threshold_and_scale_transform(nums, min_val=-1.5, max_val=1.5, min_scale=0.7, max_scale=1.3):
    result = []
    for num in nums:
        num = max(min_val, min(max_val, num))
        scaled = min_scale + (num - min_val) * (max_scale - min_scale) / (max_val - min_val)
        result.append(scaled)
    return result

def random_weight_transform(nums, min_val=0.7, max_val=1.3):
    return [random.uniform(min_val, max_val) for _ in nums]

def rank_based_transform(nums, min_scale=0.7, max_scale=1.3):
    sorted_indices = sorted(range(len(nums)), key=lambda k: nums[k])
    result = []
    for i, idx in enumerate(sorted_indices):
        if len(nums) == 1:
            result.append(1.0)
        else:
            scaled = min_scale + (i / (len(nums) - 1)) * (max_scale - min_scale)
            result.append(scaled)
    final_result = [result[sorted_indices.index(i)] for i in range(len(nums))]
    return final_result

def progressive_transform(nums, negate, min_scale=0.7, max_scale=1.3):
    result = []
    for i, num in enumerate(nums):
        scaled = max_scale - (i / (len(nums) - 1)) * (max_scale - min_scale)
        result.append(scaled)
    final_result = result
    if negate:
        return final_result[::-1]
    else:
        return final_result


weight_transform_methods = {
    'origin': lambda x: x,
    'binary': binary_weight_transform,
    'threshold': threshold_weight_transform,
    'threshold_and_scale': threshold_and_scale_transform,
    'random': random_weight_transform,
    'rank_based': rank_based_transform,
    'progressive': progressive_transform
}

def apply_weight_transform(weight_values, transform_method, transform_params, negate=False):
    """Helper function to apply weight transformation with the configured parameters"""
    if weight_values is None:
        return None
        
    # Apply negation if needed (for chosen weights)
    if negate:
        weight_values = [-x for x in weight_values]
        
    # Apply the transform method with parameters
    transform_func = weight_transform_methods[transform_method]
    if transform_method == 'binary' and 'top_percent' in transform_params:
        return transform_func(weight_values, top_percent=transform_params['top_percent'])
    elif transform_method == 'threshold' and ('upper_threshold' in transform_params or 'lower_threshold' in transform_params):
        return transform_func(
            weight_values, 
            upper_threshold=transform_params.get('upper_threshold', 1),
            lower_threshold=transform_params.get('lower_threshold', -1)
        )
    elif transform_method == 'threshold_and_scale' and any(param in transform_params for param in ['min_val', 'max_val', 'min_scale', 'max_scale']):
        return transform_func(
            weight_values,
            min_val=transform_params.get('min_val', -1.5),
            max_val=transform_params.get('max_val', 1.5),
            min_scale=transform_params.get('min_scale', 0.7),
            max_scale=transform_params.get('max_scale', 1.3)
        )
    elif transform_method == 'random' and ('min_val' in transform_params or 'max_val' in transform_params):
        return transform_func(
            weight_values,
            min_val=transform_params.get('min_val', 0.7),
            max_val=transform_params.get('max_val', 1.3)
        )
    elif transform_method == 'rank_based' and ('min_scale' in transform_params or 'max_scale' in transform_params):
        return transform_func(
            weight_values,
            min_scale=transform_params.get('min_scale', 0.7),
            max_scale=transform_params.get('max_scale', 1.3)
        )
    elif transform_method == 'progressive' and ('min_scale' in transform_params or 'max_scale' in transform_params):
        return transform_func(
            weight_values,
            negate,
            min_scale=transform_params.get('min_scale', 0.7),
            max_scale=transform_params.get('max_scale', 1.3),
        )
    else:
        # Default case with no special parameters
        return transform_func(weight_values)

@dataclass
class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        max_prompt_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the prompt to be processed.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        padding_value (`int`, defaults to 0):
            The value used for padding.
        truncation_mode: (`str`, defaults to "keep_end"):
            The truncation mode to use when truncating the prompt + chosen/rejected responses.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"
    transform_config_file: Optional[str] = None

    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: Dict[str, str],
        rejected_weight=None,
        chosen_weight=None,
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)

        rejected_tokens = {}
        for key in rejected:
            rejected_tokens[key] = self.tokenizer(rejected[key], add_special_tokens=False)

        if chosen_weight is not None:
            assert len(chosen_weight) == len(chosen_tokens['input_ids'])

        if rejected_weight is not None:
            for key in rejected_weight:
                key1 = key.split("_")[0]
                assert len(rejected_weight[key]) == len(rejected_tokens[key1]['input_ids'])
            
        assert self.tokenizer.eos_token_id not in prompt_tokens["input_ids"], f"Prompt contains EOS token: {prompt}"
        assert (
            self.tokenizer.eos_token_id not in chosen_tokens["input_ids"]
        ), f"Chosen response contains EOS token: {chosen}"
        assert (
            all([self.tokenizer.eos_token_id not in rejected_tokens[key]["input_ids"] for key in rejected_tokens])
        ), f"Rejected response contains EOS token: {rejected}"

        chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)
        for key in rejected_tokens:
            rejected_tokens[key]["input_ids"].append(self.tokenizer.eos_token_id)
            rejected_tokens[key]["attention_mask"].append(1)
        max_rejected_len = max([len(rejected_tokens[key]["input_ids"]) for key in rejected_tokens])
        longer_response_length = max(len(chosen_tokens["input_ids"]), max_rejected_len)

        # if combined sequence is too long, truncate the prompt
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            if self.truncation_mode == "keep_start":
                prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
            elif self.truncation_mode == "keep_end":
                prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            chosen_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in chosen_tokens.items()}
            rejected_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in rejected_tokens.items()}

        # Create labels
        chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
        rejected_sequence_tokens = {}
        # rejected_tokens: Dict[str, Dict]
        for key in rejected_tokens:
            rejected_sequence_tokens[key] = {k: prompt_tokens[k] + rejected_tokens[key][k] for k in rejected_tokens[key]}
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )
        for key in rejected_sequence_tokens:
            # print(rejected_sequence_tokens)
            try:
                rejected_sequence_tokens[key]["labels"] = rejected_sequence_tokens[key]["input_ids"][:]
            except Exception as e:
                print(f"Error processing key '{key}': {e}")
                print("Current rejected_sequence_tokens1:", rejected_sequence_tokens)
                print(rejected_tokens)
                print(rejected)
                print(max_rejected_len)
                print(longer_response_length)
                print(len(prompt_tokens["input_ids"]))
                raise  # 重新抛出异常

            try:
                rejected_sequence_tokens[key]["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
                    prompt_tokens["input_ids"]
                )
            except Exception as e:
                print(f"Error processing key '{key}': {e}")
                print("Current rejected_sequence_tokens2:", rejected_sequence_tokens)
                print(rejected_tokens)
                print(rejected)
                print(max_rejected_len)
                print(longer_response_length)
                print(len(prompt_tokens["input_ids"]))
                raise

        batch = {}

        if rejected_weight is not None:
            for key in rejected_weight:
                key1 = key.split("_")[0]
                batch[key] = [self.padding_value] * len(prompt_tokens['input_ids']) + rejected_weight[key][:len(rejected_tokens[key1]['input_ids'])-1] + [self.padding_value]
        else:
            for key in rejected_tokens:
                key1 = key + "_weight"
                batch[key1] =  [self.padding_value] * len(prompt_tokens['input_ids']) + [1]*(len(rejected_tokens[key]['input_ids'])-1) + [self.padding_value]

        if chosen_weight is not None:
            batch['chosen_weight'] =  [self.padding_value] * len(prompt_tokens['input_ids']) + chosen_weight[:len(chosen_tokens['input_ids'])-1] + [self.padding_value]
        else:
            batch['chosen_weight'] =  [self.padding_value] * len(prompt_tokens['input_ids']) + [1]*(len(chosen_tokens['input_ids'])-1) + [self.padding_value]

        assert len(batch['chosen_weight']) == len(chosen_sequence_tokens['labels'])
        for key in batch:
            key1 = key.split("_")[0]
            if key.startswith("rejected") and key.endswith("weight"):
                assert len(batch[key]) == len(rejected_sequence_tokens[key1]['labels'])

        batch["prompt"] = prompt
        batch["chosen"] = prompt + chosen
        for key in rejected:
            batch[key] = prompt + rejected[key]
        batch["chosen_response_only"] = chosen
        for key in rejected:
            batch[f"{key}_response_only"] = rejected[key]

        for k, toks in {
            "chosen": chosen_sequence_tokens,
            # "rejected": rejected_sequence_tokens,
            "prompt": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens
        # rejected_sequence_tokens: Dict[str, Dict]
        for k, toks in rejected_sequence_tokens.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens
        
        return batch

    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels") or k.endswith('_weight'):
                # adapted from https://stackoverflow.com/questions/73256206
                if "prompt" in k:
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    if k.endswith('_weight'):
                        to_pad = [torch.FloatTensor(ex[k]) for ex in batch]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                    # to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("_input_ids"):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = self.label_pad_token_id
                elif k.endswith('_attention_mask') or k.endswith('_weight'):
                    padding_value = self.padding_value
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                # for the prompt, flip back so padding is on left side
                if "prompt" in k:
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]
        return padded_batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []

        if self.transform_config_file:
            with open(self.transform_config_file, "r", encoding="utf-8") as f:
                transform_config = yaml.safe_load(f)
            transform_method = transform_config.get('method', 'origin')
            if transform_method in transform_config:
                transform_params = transform_config.get(transform_method, {})

        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = {}
            for key in feature:
                if key.startswith("rejected") and not key.endswith("weight"):
                    rejected[key] = feature[key]
            
            if "chosen_weight" in feature:
                chosen_weight = apply_weight_transform(feature["chosen_weight"], transform_method, transform_params, negate=True)
            else:
                chosen_weight = None
            
            rejected_weight = {}
            for key in feature:
                if key.startswith("rejected") and key.endswith("weight"):
                    rejected_weight[key] = apply_weight_transform(feature[key], transform_method, transform_params, negate=False)
            if rejected_weight == {}:
                rejected_weight = None

            batch_element = self.tokenize_batch_element(prompt, chosen, rejected, rejected_weight, chosen_weight)
            tokenized_batch.append(batch_element)
            # print("feature", feature)
            # print("chosen_weight:", chosen_weight)
            # print("rejected_weight:", rejected_weight)
        # return collated batch
        return self.collate(tokenized_batch)
    
def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )