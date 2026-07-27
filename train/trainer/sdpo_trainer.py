# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union, Tuple, Dict, Literal
from unittest.mock import patch

import torch
import torch.utils.data
import torch.nn.functional as F
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
    DataCollator,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl import apply_chat_template, is_conversational, maybe_apply_chat_template
# from trl import is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl import SyncRefModelCallback
from trl import GRPOConfig, DPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad, selective_log_softmax
from utils.utility import DPODataCollatorWithPadding, pad_to_length

import random

from transformers import (
        is_wandb_available, 
        AutoTokenizer, 
        AutoModelForCausalLM,
        TemperatureLogitsWarper, 
        LogitsProcessorList,
        Trainer
    )

from utils.LogitProcessor import ConstrainedLogitsProcessor
from transformers.generation import LogitsProcessor
import math

if is_peft_available():
    from peft import PeftConfig, get_peft_model

# if is_vllm_available():
    # from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class SDPOTrainer(Trainer):
    """
    Trainer for Direct Preference Optimization (DPO) method.

    This class is a wrapper around the [`transformers.Trainer`] class and inherits all of its attributes and methods.

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        ref_model ([`PreTrainedModelWrapper`]):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation
            and loss. If no reference model is provided, the trainer will create a reference model with the same
            architecture as the model to be optimized.
        args ([`DPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`.
            Will default to [`DataCollatorForPreference`].
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. DPO supports [preference](#preference) type and. The format of the samples can
            be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.BaseImageProcessor`], [`~transformers.FeatureExtractionMixin`] or [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. If `None`, the processing class is loaded from the model's name
            with [`~transformers.AutoTokenizer.from_pretrained`].
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
            a dictionary string to metric values. *Note* When passing TrainingArgs with `batch_eval_metrics` set to
            `True`, your compute_metrics function must take a boolean `compute_result` argument. This will be triggered
            after the last eval batch to signal that the function needs to calculate and return the global summary
            statistics rather than accumulating the batch-level statistics.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        optimizer_cls_and_kwargs (`Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]`, *optional*):
            A tuple containing the optimizer class and keyword arguments to use. Overrides `optim` and `optim_args` in
            `args`. Incompatible with the `optimizers` argument.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

            Note that the labels (second parameter) will be `None` if the dataset does not have them.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "dpo"]
    _name = "DPO"
    _paper = {
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "id": "2305.18290",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @inproceedings{rafailov2023direct,
                title        = {{Direct Preference Optimization: Your Language Model is Secretly a Reward Model}},
                author       = {Rafael Rafailov and Archit Sharma and Eric Mitchell and Christopher D. Manning and Stefano Ermon and Chelsea Finn},
                year         = 2023,
                booktitle    = {Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023},
                url          = {http://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html},
                editor       = {Alice Oh and Tristan Naumann and Amir Globerson and Kate Saenko and Moritz Hardt and Sergey Levine},
            }"""),
    }

    def __init__(
        self,
        model: Union[str, nn.Module, PreTrainedModel],
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[DPOConfig] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = DPOConfig(f"{model_name}-DPO")

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                model = prepare_model_for_int8_training(model)
            model = get_peft_model(model, peft_config)

        # Data collator
        if data_collator is None:
            if processing_class is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default DPODataCollatorWithPadding"
                )
            if args.max_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_length` in the DPOTrainer's init"
                    " it will be set to `512` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                args.max_length = 512
            if args.max_prompt_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_prompt_length` in the DPOTrainer's init"
                    " it will be set to `128` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                args.max_prompt_length = 128

            data_collator = DPODataCollatorWithPadding(
                processing_class,
                max_length=args.max_length,
                max_prompt_length=args.max_prompt_length,
                label_pad_token_id=label_pad_token_id,
                padding_value=padding_value,
                truncation_mode=truncation_mode,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False


        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = padding_value
        self.reference_free = args.reference_free

        self.beta = args.beta
        self.loss_type = args.loss_type
        self.ref_model = ref_model

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )


        # Since we inherit from trainer we always have access to an accelerator
        if hasattr(self, "accelerator"):
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        else:
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

    def concatenated_inputs(self, batch: dict[str, Union[list, torch.LongTensor]]) -> dict[str, torch.LongTensor]:
        """
        Concatenate the `chosen` and `rejected` inputs from the batch into a single tensor for both the prompt and
        completion sequences.

        Args:
            batch (`dict[str, Union[list, torch.LongTensor]]`):
                A batch of input data. The batch must contain the following keys:

                - `"prompt_input_ids"`: Tensor of shape `(batch_size, prompt_length)` representing the prompt input
                  IDs.
                - `"chosen_input_ids"`: Tensor of shape `(batch_size, chosen_length)` representing the chosen
                  completion input IDs.
                - `"rejected_input_ids"`: Tensor of shape `(batch_size, rejected_length)` representing the rejected
                  completion input IDs.
                - `"prompt_pixel_values"` (optional): Tensor for pixel values, if available.
                - `"prompt_pixel_attention_mask"` (optional): Tensor for pixel attention masks, if available.

            padding_value (`int`):
                The padding value to use for the concatenated completion sequences (`chosen_input_ids` and
                `rejected_input_ids`).

        Returns:
            `dict[str, torch.LongTensor]`: A dictionary containing:

                - `"prompt_input_ids"`: Concatenated prompt input IDs of shape `(2 * batch_size, prompt_length)`.
                - `"completion_input_ids"`: Concatenated chosen and rejected completion input IDs of shape `(2 *
                  batch_size, max_completion_length)`.
                - `"prompt_attention_mask"`: Concatenated prompt attention masks of shape `(2 * batch_size,
                  prompt_length)`.
                - `"completion_attention_mask"`: Concatenated chosen and rejected attention masks of shape `(2 *
                  batch_size, max_completion_length)`.
                - `"pixel_values"` (optional): Concatenated pixel values if `"prompt_pixel_values"` are present.
                - `"pixel_attention_mask"` (optional): Concatenated pixel attention masks if
                  `"prompt_pixel_attention_mask"` are present.

        Notes:
            The completion input IDs and attention masks are padded to the maximum completion length of the chosen or
            rejected sequences.
        """

        # Concatenate the chosen and rejected completions
        rejected_max_len = max([batch[key].shape[1] for key in batch if key.startswith("rejected") and key.endswith("_input_ids")])
        max_length = max(batch["chosen_input_ids"].shape[1], rejected_max_len)
        concatenated_batch = {}
        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                # concatenated_key = k.replace("rejected", "concatenated")
                prefix = k.split("_")[0]
                concatenated_key = "concatenated" + k[len(prefix):]
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(self.accelerator.device)
        return concatenated_batch

    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        loss_type: str = "",
        model_output: dict[str, torch.FloatTensor] = None,
        rewards: dict = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            chosen_logps (`torch.FloatTensor`):
                Log probabilities of the model for the chosen responses. Shape: `(batch_size,)`.
            rejected_logps (`torch.FloatTensor`):
                Log probabilities of the model for the rejected responses. Shape: `(batch_size,)`.
            ref_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the chosen responses. Shape: `(batch_size,)`.
            ref_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the rejected responses. Shape: `(batch_size,)`.
            loss_type (`str`, defaults to `"sigmoid"`):
                The type of loss to compute. One of:
                - `"sigmoid"`: Sigmoid loss from the original [DPO](https://huggingface.co/papers/2305.18290) paper.
                - `"hinge"`: Hinge loss on the normalized likelihood from the
                  [SLiC](https://huggingface.co/papers/2305.10425) paper.
                - `"ipo"`: IPO loss from the [IPO](https://huggingface.co/papers/2310.12036) paper.
                - `"exo_pair"`: Pairwise EXO loss from the [EXO](https://huggingface.co/papers/2402.00856) paper.
                - `"nca_pair"`: Pairwise NCA loss from the [NCA](https://huggingface.co/papers/2402.05369) paper.
                - `"robust"`: Unbiased estimate of the DPO loss that is robust to preference noise from the [Robust
                  DPO](https://huggingface.co/papers/2403.00409) paper.
                - `"bco_pair"`: Pairwise BCO loss from the [BCO](https://huggingface.co/papers/2404.04656) paper.
                - `"sppo_hard"`: SPPO loss with hard label from the [SPPO](https://huggingface.co/papers/2405.00675)
                  paper.
                - `"aot"`: AOT loss for paired datasets from the [AOT](https://huggingface.co/papers/2406.05882) paper.
                - `"aot_pair"`: AOT loss for unpaired datasets from the [AOT](https://huggingface.co/papers/2406.05882)
                  paper.
                - `"discopop"`: DiscoPOP (a.k.a Log-Ratio Modulated Loss, LRML) loss from the
                  [DiscoPOP](https://huggingface.co/papers/2406.08414) paper.
                - `"apo_zero"`: APO-zero loss from the [APO](https://huggingface.co/papers/2408.06266) paper.
                - `"apo_down"`: APO-down loss from the [APO](https://huggingface.co/papers/2408.06266) paper.
                - `"sft"`: Negative log-likelihood loss (standard supervised fine-tuning loss).
            model_output (`dict[str, torch.FloatTensor]`, *optional*):
                The output of the model's forward pass. This is used to compute auxiliary losses if enabled.

        Returns:
            A tuple of three tensors: `(losses, chosen_rewards, rejected_rewards)`. The losses tensor contains the DPO
            loss for each example in the batch. The `chosen_rewards` and `rejected_rewards` tensors contain the rewards
            for the chosen and rejected responses, respectively.
        """

        # device = self.accelerator.device
        # Get the log ratios for the chosen and rejected 
        chosen_logratios = chosen_logps - (not self.reference_free) * ref_chosen_logps
        rejected_logratios = {}
        for key in rejected_logps:
            rejected_logratios[key] = rejected_logps[key] - (not self.reference_free) * ref_rejected_logps[key]
        # margin = [chosen_logratios - rejected_logratios[key] for key in rejected_logratios]
        # temp = sum(margin)
        # print("!!!!!!", len(margin))
        # print("******", margin)
        if loss_type == "dpo":
            # losses = -F.logsigmoid(self.beta * temp)
            temp = sum(torch.exp(self.beta * (rejected_logratios[key] - chosen_logratios)) for key in rejected_logratios)
            temp1 = -torch.log(temp)
            losses = -F.logsigmoid(temp1)

        elif loss_type == "simpo":
            pi_logratios = {}
            for key in rejected_logps:
                pi_logratios[key] = chosen_logps - rejected_logps[key]
            margin = [pi_logratios[key] - self.gamma for key in pi_logratios]
            temp = sum(margin)
            losses = -F.logsigmoid(self.beta * temp)

        elif loss_type == "rspo":
            # RSPO: Ranking-Guided Softmax Preference Optimization (GR4AD, Eq.16)
            # L = -log sigma(-log SUM_j M_ij * exp(beta * (rej_j_logratio - chosen_logratio)))
            # M_ij = |G_i - G_j| * |1/D_{|rank_i-rank_j|} - 1/D_{|rank_i-rank_j|+1}|

            # Extract reward values
            v_chosen = torch.tensor(rewards["chosen_reward"], device=chosen_logps.device, dtype=torch.float32)
            rej_reward_keys = sorted([k for k in rewards if k.startswith("rejected") and k.endswith("_reward")])
            rej_rewards = torch.stack([
                torch.tensor(rewards[rk], device=chosen_logps.device, dtype=torch.float32)
                for rk in rej_reward_keys
            ], dim=1)  # (batch_size, N)

            # Compute ideal DCG normalization Z
            all_rewards_tensor = torch.cat([v_chosen.unsqueeze(1), rej_rewards], dim=1)  # (batch_size, 1+N)
            sorted_rewards, _ = torch.sort(all_rewards_tensor, dim=1, descending=True)
            positions = torch.arange(1, sorted_rewards.shape[1] + 1, device=sorted_rewards.device).float()
            discount = torch.log2(1.0 + positions)
            Z = ((2.0 ** sorted_rewards - 1.0) / discount).sum(dim=1).clamp(min=1e-8)  # (batch_size,)

            # Compute G values: G_i = (2^v_i - 1) / Z
            G_chosen = (2.0 ** v_chosen - 1.0) / Z  # (batch_size,)

            # Determine ranks of rejected items (chosen is always rank 1 since v_chosen=1.0)
            _, rej_sort_idx = torch.sort(rej_rewards, dim=1, descending=True)
            rej_ranks = torch.zeros_like(rej_rewards)
            rej_ranks.scatter_(1, rej_sort_idx,
                torch.arange(2, rej_rewards.shape[1] + 2, device=rej_rewards.device).float().unsqueeze(0).expand_as(rej_rewards))

            # Compute Lambda-weighted sum over all rejected items
            rej_key_list = sorted(rejected_logratios.keys())
            weighted_terms = []
            for j_idx, key in enumerate(rej_key_list):
                v_j = rej_rewards[:, j_idx]
                G_j = (2.0 ** v_j - 1.0) / Z
                rank_diff = torch.abs(rej_ranks[:, j_idx] - 1.0)  # chosen is rank 1
                D_diff = 1.0 / torch.log2(1.0 + rank_diff) - 1.0 / torch.log2(2.0 + rank_diff)
                M_ij = torch.abs(G_chosen - G_j) * D_diff
                exp_term = M_ij * torch.exp(self.beta * (rejected_logratios[key] - chosen_logratios))
                weighted_terms.append(exp_term)

            temp = sum(weighted_terms).clamp(min=1e-8)
            losses = -F.logsigmoid(-torch.log(temp))

        else:
            raise ValueError(
                f"Unknown loss type: {loss_type}"
            )
        
        rejected_rewards = {}
        chosen_rewards = self.beta * (chosen_logps - (not self.reference_free) * ref_chosen_logps).detach()
        for key in rejected_logps:
            rejected_rewards[key] = self.beta * (rejected_logps[key] - (not self.reference_free) * ref_rejected_logps[key]).detach()

        return losses, chosen_rewards, rejected_rewards

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")
        # print(logits.shape) # (bs, seq_len, vocab_size)
        # print(labels.shape) # (bs, seq_len)
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id # -100 # (bs, seq_len)

        # # -------------------------------------------------------------------
        # total_rows = labels.shape[0]
        # chosen_indices = torch.arange(total_rows, device=labels.device) % step
        # corresponding_chosen_labels = labels[chosen_indices]
        # diff_mask = labels != corresponding_chosen_labels
        # final_mask = torch.ones_like(loss_mask)
        # final_mask[step:] = diff_mask[step:]
        # loss_mask = loss_mask & final_mask
        # # -------------------------------------------------------------------
        
        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0
        # print("!!!", logits.shape)
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        # print("***", per_token_logps.shape)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[list, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor], torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        # print(batch['chosen_input_ids'].shape) # (bs, length)
        concatenated_batch = self.concatenated_inputs(batch)
        # print(concatenated_batch['concatenated_labels'].shape)
        # print(concatenated_batch['concatenated_labels'])
        # count = (concatenated_batch['concatenated_labels'] != -100).sum().item()
        # print("!!!", concatenated_batch["concatenated_attention_mask"].shape)
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits.to(torch.float32)
        # print(all_logits.shape) # ((chosen+rejected)*bs, length, vocab_size)
        step = batch["chosen_input_ids"].shape[0]
        # print("!!!", type(step))
        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
        )
        # print(all_logps.shape) # (len(chosen+rejected)*bs,)
        # print(concatenated_batch["concatenated_labels"].shape) # (len(chosen+rejected)*bs, length)
        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]]
        # print(all_logps.shape) # (bs,)
        # step = batch["chosen_input_ids"].shape[0]
        rejected_logps = {}
        cnt = 0
        for key in batch:
            if key.startswith("rejected") and key.endswith("_input_ids"):
                cnt += 1
                rejected_logps[f"rejected{cnt}"] = all_logps[step*cnt : step*(cnt+1)]

        chosen_logits = all_logits[: batch["chosen_input_ids"].shape[0]]
        rejected_logits = {}
        cnt = 0
        for key in batch:
            if key.startswith("rejected") and key.endswith("_input_ids"):
                cnt += 1
                rejected_logits[f"rejected{cnt}"] = all_logits[step*cnt : step*(cnt+1)]
        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_batch_loss_metrics(
        self,
        model: Union[PreTrainedModel, nn.Module],
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        with torch.no_grad():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = self.concatenated_forward(self.ref_model, batch)

        rewards = None
        if self.loss_type == "rspo":
            rewards = {}
            for key in batch:
                if key.endswith("_reward"):
                    rewards[key] = batch[key]

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            loss_type=self.loss_type,
            rewards=rewards,
        )
        # reward_accuracies 记录 chosen 比所有 rejected 的收益都大的比例是多少
        reward_accuracies = None
        for key in rejected_rewards:
            if reward_accuracies is None:
                reward_accuracies = (chosen_rewards > rejected_rewards[key]).float()
            else:
                reward_accuracies *= (chosen_rewards > rejected_rewards[key]).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().numpy().mean()
        for key in rejected_rewards:
            metrics[f"{prefix}rewards/{key}"] = rejected_rewards[key].cpu().numpy().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().numpy().mean()
        for key in rejected_rewards:
            metrics[f"{prefix}rewards/margins-{key}"] = (chosen_rewards - rejected_rewards[key]).cpu().numpy().mean()
        for key in policy_rejected_logps:    
            metrics[f"{prefix}logps/rejected-{key}"] = policy_rejected_logps[key].detach().cpu().numpy().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().numpy().mean()
        # for key in policy_rejected_logits:
        #     metrics[f"{prefix}logits/rejected-{key}"] = policy_rejected_logits[key].detach().cpu().numpy().mean()
        # metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().numpy().mean()
        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # print("\n!!!\n", inputs, "\n!!!\n")
        # print("\n***\n", inputs.keys(), "\n***\n")
        # for x in inputs:
        #     if x.endswith("_input_ids") or x.endswith("_attention_mask") or x.endswith("_labels"):
        #         print("key:", x, "shape:", inputs[x][0].shape)
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def generate_from_model_and_ref(self, model, batch: dict[str, torch.LongTensor]) -> tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        policy_output = model.generate(
            batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.config.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        reference_output = self.ref_model.generate(
            batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.config.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return loss.detach(), None, None

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "logits_test/chosen": metrics["logits_test/chosen"],
            # "logits_test/rejected": metrics["logits_test/rejected"],
        }
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1)
        labels = torch.zeros(logits.shape[0])

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`dict[str, float]`):
                The values to log.
            start_time (`float`, *optional*):
                Start time of the training.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs, start_time)

    # # Ensure the model card is saved along with the checkpoint
    # def _save_checkpoint(self, model, trial):
    #     if self.args.hub_model_id is None:
    #         model_name = Path(self.args.output_dir).name
    #     else:
    #         model_name = self.args.hub_model_id.split("/")[-1]
    #     self.create_model_card(model_name=model_name)
    #     super()._save_checkpoint(model, trial)