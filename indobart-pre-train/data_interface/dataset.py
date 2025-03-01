# coding:utf-8
import os
import pickle
from dataclasses import dataclass
from typing import Optional, Union

import torch
from datasets import load_dataset
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

def ids_to_clean_text(tokenizer, generated_ids):
    gen_text = tokenizer.batch_decode(generated_ids, clean_up_tokenization_spaces=False)
    return gen_text


def read_or_new_pickle(path, default_fn, no_cache=False):
    if not no_cache and os.path.isfile(path):
        with open(path, "rb") as f:
            try:
                return pickle.load(f)
            except Exception as e:  # so many things could go wrong, can't be more specific.
                print(e)
    default = default_fn()
    with open(path, "wb") as f:
        pickle.dump(default, f)
    return default


class AMRDataSet(torch.nn.Module):
    def __init__(
        self,
        tokenizer,
        train_file,
        validation_file,
        test_file,
        prefix="",
        pad_to_max_length=True,
        max_src_length=512,
        max_tgt_length=512,
        ignore_pad_token_for_loss=True,
        no_cache=False,
    ):
        super().__init__()
        self.train_file = train_file
        self.validation_file = validation_file
        self.test_file = test_file
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.pad_to_max_length = pad_to_max_length
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.no_cache = no_cache

    def setup(self, stage="fit"):
        data_files = {}
        data_files["train"] = self.train_file
        data_files["validation"] = self.validation_file
        data_files["test"] = self.test_file

        datasets = load_dataset(
            f"{os.path.dirname(__file__)}/amrdata.py", data_files=data_files, keep_in_memory=True)
        print("datasets:", datasets)
        column_names = datasets["train"].column_names
        print("colums:", column_names)
        padding = "max_length" if self.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            amrs = examples["amr"]           # AMR tokens
            sents = examples["text"]          # text tokens
            sents = [self.prefix + inp for inp in sents]

            model_inputs = self.tokenizer(
                sents, max_length=self.max_src_length, padding=False, truncation=True
            )
            amr_ids = [self.tokenizer.tokenize_amr(itm.split())[
                :self.max_src_length - 1] for itm in amrs]
            model_inputs["labels"] = amr_ids
            model_inputs["input_ids"] = [a[:-1] for a in model_inputs["input_ids"]]

            joint_ids = [
                [self.tokenizer.bos_token_id] + srci + [self.tokenizer.eos_token_id, self.tokenizer.amr_bos_token_id] + tgti + [self.tokenizer.amr_eos_token_id]
                for srci, tgti in zip(model_inputs["input_ids"], model_inputs["labels"])
            ]  # [<s> x1,x2...,xn </s> <AMR> y1,y2,...ym </AMR>]
            # print()
            # print("DEBUG69")
            # print(ids_to_clean_text(self.tokenizer, joint_ids))

            max_src_length = min(self.max_src_length * 2, 512)
            #! With an assumption that the text won't be longer than `max_scr_length - 4`.
            joint_ids = [
                itm[:max_src_length - 1] + [self.tokenizer.amr_eos_token_id]
                if len(itm) > max_src_length
                else itm
                for itm in joint_ids
            ]
            seg_ids = [
                [0]*(len(srci)+2) + [1]*(len(tgti)+2)
                for srci, tgti in zip(model_inputs["input_ids"], model_inputs["labels"])
            ]  # [0,0,...,0,1,1,...1]
            seg_ids = [itm[:max_src_length] for itm in seg_ids]
            model_inputs["joint_ids"] = joint_ids
            model_inputs["seg_ids"] = seg_ids
            srcEtgt_ids = [
                [self.tokenizer.bos_token_id]
                + srci[:self.max_src_length - 5]
                + [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.amr_bos_token_id,
                    self.tokenizer.mask_token_id,
                    self.tokenizer.amr_eos_token_id,
                ]
                for srci in model_inputs["input_ids"]
            ]  # [<s> x1,x2...,xn <\s> <AMR> [mask] </AMR>]
            srcEtgt_segids = [
                [0]*(len(itm) - 3) + [1]*3
                for itm in srcEtgt_ids
            ]
            Esrctgt_ids = [
                [
                    self.tokenizer.bos_token_id, # T5 does not have bos_token
                    self.tokenizer.mask_token_id,
                    self.tokenizer.eos_token_id,
                    self.tokenizer.amr_bos_token_id
                ]
                + tgti[:self.max_src_length - 5]
                + [self.tokenizer.amr_eos_token_id]
                for tgti in model_inputs["labels"]
            ]  # [<s> [mask] <\s> <AMR> y1,y2...,yn </AMR>]
            Esrctgt_segids = [
                [0]*3 + [1]*(len(itm) - 3)
                for itm in Esrctgt_ids
            ]
            model_inputs["srcEtgt_ids"] = srcEtgt_ids
            model_inputs["srcEtgt_segids"] = srcEtgt_segids
            model_inputs["Esrctgt_ids"] = Esrctgt_ids
            model_inputs["Esrctgt_segids"] = Esrctgt_segids
            model_inputs["labels"] = [(amr_ids[:self.max_src_length-1] + [self.tokenizer.amr_eos_token_id]) for amr_ids in model_inputs["labels"]]
            model_inputs["input_ids"] = [([self.tokenizer.bos_token_id] + ids[:self.max_src_length-2] + [self.tokenizer.eos_token_id]) for ids in model_inputs["input_ids"]]
            # print()
            # print(model_inputs)
            return model_inputs

        if not os.path.exists('.cache'):
            os.makedirs('.cache')

        self.train_dataset = read_or_new_pickle(".cache/pre-train-train_dataset.pkl", lambda: datasets["train"].map(
            tokenize_function, batched=True, remove_columns=["amr", "text"], num_proc=1
        ), no_cache=self.no_cache)
        print(f"ALL {len(self.train_dataset)} training instances")
        self.valid_dataset = read_or_new_pickle(".cache/pre-train-valid_dataset.pkl", lambda: datasets["validation"].map(
            tokenize_function, batched=True, remove_columns=["amr", "text"], num_proc=1
        ), no_cache=self.no_cache)
        print(f"ALL {len(self.valid_dataset)} validation instances")

        self.test_dataset = read_or_new_pickle(".cache/pre-train-test_dataset.pkl", lambda: datasets["test"].map(
            tokenize_function, batched=True, remove_columns=["amr", "text"], num_proc=1
        ), no_cache=self.no_cache)
        print(f"ALL {len(self.test_dataset)} test instances")

        print("Dataset Instance Example:", self.train_dataset[0])


def padding_func(features, padding_side="right", pad_token_id=1, key="label"):
    assert key in features[0].keys(), f"{key} not in {features[0].keys()}"
    max_label_length = max(len(feature[key]) for feature in features)
    for feature in features:
        remainder = [pad_token_id] * (max_label_length - len(feature[key]))
        feature[key] = (
            feature[key] +
            remainder if padding_side == "right" else remainder + feature[key]
        )
    return


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.label_pad_token_id,
            key="labels",
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="joint_ids",
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="seg_ids",
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="srcEtgt_ids",
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="srcEtgt_segids",
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="Esrctgt_ids",
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="Esrctgt_segids",
        )
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        return {
            "input_ids": features["input_ids"],
            "labels": features["labels"],
            "joint_ids": features["joint_ids"],
            "seg_ids": features["seg_ids"],
            "srcEtgt_ids": features["srcEtgt_ids"],
            "srcEtgt_segids": features["srcEtgt_segids"],
            "Esrctgt_ids": features["Esrctgt_ids"],
            "Esrctgt_segids": features["Esrctgt_segids"],
        }
