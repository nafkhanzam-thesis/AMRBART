# coding:utf-8

import json
import math
import numpy as np
import torch
import os
from torch import nn
from typing import Callable, Dict, Iterable, List, Tuple, Union, Any
from torch.optim import Optimizer
from numpy.lib.twodim_base import mask_indices
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    PreTrainedTokenizer,
)


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def get_inverse_sqrt_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, (num_warmup_steps / current_step) ** 0.5)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def ids_to_clean_text(tokenizer, generated_ids: List[int]):
    generated_ids.masked_fill_(generated_ids == -100, tokenizer.pad_token_id)
    gen_text = tokenizer.batch_decode(generated_ids, clean_up_tokenization_spaces=False)
    return gen_text


def save_dummy_batch(args, input_ids, dec_inp_ids, labels, tokenizer, prefix="train"):
    dummy_ids, dummy_tokens = [], []
    for idx in range(len(input_ids)):
        ith_dict, ith_tok_dict = {}, {}
        ith_dict["input_ids"] = input_ids[idx].tolist()
        ith_dict["label_ids"] = labels[idx].tolist()
        ith_dict["dec_inp_ids"] = dec_inp_ids[idx].tolist()
        dummy_ids.append(ith_dict)

        ith_tok_dict["input_tokens"] = ids_to_clean_text(tokenizer, input_ids[idx])
        ith_tok_dict["label_tokens"] = ids_to_clean_text(tokenizer, labels[idx])
        ith_tok_dict["dec_inp_tokens"] = ids_to_clean_text(tokenizer, dec_inp_ids[idx])
        dummy_tokens.append(ith_tok_dict)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(
        args.output_dir + f"/dummy_{prefix}_ids.json", "w", encoding="utf-8"
    ) as fout:
        json.dump(dummy_ids, fout, indent=4)
    with open(
        args.output_dir + f"/dummy_{prefix}_token.json", "w", encoding="utf-8"
    ) as fout:
        json.dump(dummy_tokens, fout, indent=4)


def sentence_infilling(tokenizer, inp, mlm_prob=0.35):
    token_length = len([int(itm != tokenizer.pad_token_id) for itm in inp])
    masking_length = math.floor(token_length * mlm_prob)
    masked_length = 1
    masked_inputs = inp.clone().tolist()
    while masked_length < masking_length:
        span_length = min(math.floor(np.random.poisson(3, 1)), token_length - 1)
        start_index = math.floor(np.random.uniform(1, token_length - span_length, 1))
        masked_inputs = (
            masked_inputs[:start_index]
            + [tokenizer.mask_token_id]
            + masked_inputs[start_index + span_length :]
        )
        token_length -= span_length - 1
        masked_length += span_length
    return torch.LongTensor(masked_inputs)


def text_infilling(inp, tokenizer, mlm_prob=0.35):
    res = []
    for sents in inp:
        res.append(sentence_infilling(tokenizer, sents, mlm_prob=mlm_prob))
    return pad_sequence(res, batch_first=True, padding_value=tokenizer.pad_token_id)


def joint_infilling_partial(inp, seg_id, tokenizer, mask_txt=True, mlm_prob=0.35):
    res = []
    for inp_ids, seg_iid in zip(inp, seg_id):
        inp_ids = torch.LongTensor(
            [iid for iid in inp_ids if iid != tokenizer.pad_token_id]
        )
        text_ids = torch.LongTensor(
            [inp_ids[idx] for idx in range(len(inp_ids)) if seg_iid[idx] == 0]
        )
        amr_ids = torch.LongTensor(
            [inp_ids[idx] for idx in range(len(inp_ids)) if seg_iid[idx] == 1]
        )
        if mask_txt:
            res.append(
                torch.cat(
                    [
                        sentence_infilling(tokenizer, text_ids, mlm_prob=mlm_prob),
                        amr_ids,
                    ],
                    dim=0,
                )
            )
        else:
            res.append(
                torch.cat(
                    [
                        text_ids,
                        sentence_infilling(tokenizer, amr_ids, mlm_prob=mlm_prob),
                    ],
                    dim=0,
                )
            )

    return pad_sequence(res, batch_first=True, padding_value=tokenizer.pad_token_id)


def joint_infilling_full(inp, seg_id, tokenizer, mlm_prob=0.35):
    res = []
    for inp_ids, seg_iid in zip(inp, seg_id):
        inp_ids = torch.LongTensor(
            [iid for iid in inp_ids if iid != tokenizer.pad_token_id]
        )
        text_ids = torch.LongTensor(
            [inp_ids[idx] for idx in range(len(inp_ids)) if seg_iid[idx] == 0]
        )
        amr_ids = torch.LongTensor(
            [inp_ids[idx] for idx in range(len(inp_ids)) if seg_iid[idx] == 1]
        )
        masked_txt = sentence_infilling(tokenizer, text_ids, mlm_prob=mlm_prob)
        masked_amr = sentence_infilling(tokenizer, amr_ids, mlm_prob=mlm_prob)
        res.append(torch.cat([masked_txt, masked_amr], dim=0))

    return pad_sequence(res, batch_first=True, padding_value=tokenizer.pad_token_id)


def get_STD2partial(batch, tokenizer, inp="text", mlm_prob=0.35):
    """
    If inp == text, then [Masked text -> Text]
    If inp != text, then [Masked Graph -> Graph]
    """
    assert inp in ["text", "amr"]
    if inp == "text":
        ori_input = batch["input_ids"]
        masked_input = text_infilling(ori_input, tokenizer, mlm_prob=mlm_prob)
        attention_mask = masked_input.ne(tokenizer.pad_token_id).int()
        labels = ori_input.clone()
        labels.masked_fill_(labels == tokenizer.pad_token_id, -100)
        labels = labels[:, 1:]  # [w1 w2 w3 ...]
        dec_input = ori_input[:, :-1]
        return (
            change_to_extra_id_mask(masked_input, tokenizer),
            attention_mask,
            dec_input,
            labels,
        )
    else:
        labels = batch["labels"]  # [bsz, len+1]
        shifted_input_ids = labels.new_zeros(labels.size(0), labels.size(1) + 1)
        shifted_input_ids[:, 1:] = labels.clone()
        shifted_input_ids[
            :, 0
        ] = tokenizer.amr_bos_token_id  # <AMR> w1, w2, ..., wn <\s>
        shifted_input_ids.masked_fill_(
            shifted_input_ids == -100, tokenizer.pad_token_id
        )  # replace -100 with pad_token_id
        masked_input = text_infilling(shifted_input_ids, tokenizer)
        attention_mask = masked_input.ne(tokenizer.pad_token_id).int()  # attention mask
        dec_input = labels.new_zeros(labels.size(0), labels.size(1))
        dec_input[:, 1:] = labels[:, :-1].clone()
        dec_input[:, 0] = tokenizer.amr_bos_token_id  # <s> w1 w2, ..., wn
        dec_input.masked_fill_(dec_input == -100, tokenizer.pad_token_id)
        return (
            change_to_extra_id_mask(masked_input, tokenizer),
            attention_mask,
            dec_input,
            labels,
        )


def get_MTEG2text(batch, tokenizer, mlm_prob=0.35):
    """
    [Masked Text + Empty Graph -> text]
    """
    ori_input = batch["srcEtgt_ids"]
    seg_ids = batch["srcEtgt_segids"]
    masked_input = joint_infilling_partial(
        ori_input, seg_ids, tokenizer, mask_txt=True, mlm_prob=mlm_prob
    )
    attention_mask = masked_input.ne(tokenizer.pad_token_id).int()  # attention mask
    labels = batch["input_ids"].clone()  # <s> x1...,xn </s> pad pad
    labels.masked_fill_(labels == tokenizer.pad_token_id, -100)
    labels = labels[:, 1:]  # x1...,xn </s> pad pad
    dec_input = batch["input_ids"].clone()
    dec_input = dec_input[:, :-1]
    return (
        change_to_extra_id_mask(masked_input, tokenizer),
        attention_mask,
        dec_input,
        labels,
    )


def get_ETMG2graph(batch, tokenizer, mlm_prob=0.35):
    """
    [Empty text + Masked Graph -> graph]
    """
    ori_input = batch["Esrctgt_ids"]
    seg_ids = batch["Esrctgt_segids"]
    masked_input = joint_infilling_partial(
        ori_input, seg_ids, tokenizer, mask_txt=False, mlm_prob=mlm_prob
    )
    attention_mask = masked_input.ne(tokenizer.pad_token_id).int()  # attention mask
    labels = batch["labels"]
    dec_input = labels.new_zeros(labels.size(0), labels.size(1))
    dec_input[:, 1:] = labels[:, :-1].clone()
    dec_input[:, 0] = tokenizer.amr_bos_token_id  # <s> w1 w2, ..., wn
    dec_input.masked_fill_(dec_input == -100, tokenizer.pad_token_id)
    return (
        change_to_extra_id_mask(masked_input, tokenizer),
        attention_mask,
        dec_input,
        labels,
    )


def get_PTPG2partial(batch, tokenizer, inp="text", mlm_prob=0.35):
    """
    If inp == text, then [Masked text + Graph -> Text]
    If inp != text, then [Text + Masked Graph -> Graph]
    """
    ori_input = batch["joint_ids"]
    seg_ids = batch["seg_ids"]
    if inp == "text":
        masked_input = joint_infilling_partial(
            ori_input, seg_ids, tokenizer, mask_txt=True, mlm_prob=mlm_prob
        )
        labels = batch["input_ids"].clone()
        labels.masked_fill_(labels == tokenizer.pad_token_id, -100)
        labels = labels[:, 1:]  # w1, w2, .., wn <\s>
        dec_input = batch["input_ids"].clone()
        dec_input = dec_input[:, :-1]  # <s> w1, w2, ..., wn
    else:
        masked_input = joint_infilling_partial(
            ori_input, seg_ids, tokenizer, mask_txt=False, mlm_prob=mlm_prob
        )
        labels = batch["labels"]
        dec_input = labels.new_zeros(labels.size(0), labels.size(1))
        dec_input[:, 1:] = labels[:, :-1].clone()
        dec_input[:, 0] = tokenizer.amr_bos_token_id  # <s> w1 w2, ..., wn
        dec_input.masked_fill_(
            dec_input == -100, tokenizer.pad_token_id
        )  # <AMR> w1 w2, ..., wn

    attention_mask = masked_input.ne(tokenizer.pad_token_id).int()  # attention mask
    return (
        change_to_extra_id_mask(masked_input, tokenizer),
        attention_mask,
        dec_input,
        labels,
    )


def get_MTMG2partial(batch, tokenizer, inp="text", mlm_prob=0.35):
    """
    If inp == text, then [Masked Text + Masked Graph -> text]
    If inp != text, then [Masked Text + Masked Graph -> graph]
    """
    ori_input = batch["joint_ids"]
    seg_ids = batch["seg_ids"]
    masked_input = joint_infilling_full(
        ori_input, seg_ids, tokenizer, mlm_prob=mlm_prob
    )
    if inp == "text":
        labels = batch["input_ids"].clone()
        labels.masked_fill_(labels == tokenizer.pad_token_id, -100)
        labels = labels[:, 1:]  # w1, w2, .., wn <\s>
        dec_input = batch["input_ids"].clone()
        dec_input = dec_input[:, :-1]  # <s> w1 w2, ..., wn
    else:
        labels = batch["labels"]
        dec_input = labels.new_zeros(labels.size(0), labels.size(1))
        dec_input[:, 1:] = labels[:, :-1].clone()
        dec_input[:, 0] = tokenizer.amr_bos_token_id  # <s> w1 w2, ..., wn
        dec_input.masked_fill_(dec_input == -100, tokenizer.pad_token_id)

    attention_mask = masked_input.ne(tokenizer.pad_token_id).int()  # attention mask
    return (
        change_to_extra_id_mask(masked_input, tokenizer),
        attention_mask,
        dec_input,
        labels,
    )


def get_MTMG2TG(batch, tokenizer, mlm_prob=0.35):
    ori_input = batch["joint_ids"]
    seg_ids = batch["seg_ids"]
    masked_input = joint_infilling_full(
        ori_input, seg_ids, tokenizer, mlm_prob=mlm_prob
    )

    labels = batch["joint_ids"].clone()
    labels.masked_fill_(labels == tokenizer.pad_token_id, -100)
    labels = labels[:, 1:]  # w1, w2, .., wn <\s>
    dec_input = batch["joint_ids"].clone()
    dec_input = dec_input[:, :-1]  # <s> w1 w2, ..., wn
    attention_mask = masked_input.ne(tokenizer.pad_token_id).int()  # attention mask
    return (
        change_to_extra_id_mask(masked_input, tokenizer),
        attention_mask,
        dec_input,
        labels,
    )


def change_to_extra_id_mask(masked_input, tokenizer):
    for i in range(len(masked_input)):
        tokenizer.reset_mask_token_counter()
        masked_input[i] = masked_input[i].apply_(
            lambda x: tokenizer.get_mask_token_id()
            if x == tokenizer.mask_token_id
            else x
        )

    return masked_input
