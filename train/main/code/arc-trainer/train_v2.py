import os
import re
import itertools
import json
import hashlib
import numpy as np
from numpy.random import randint
from glob import glob
from tqdm import tqdm
from collections import OrderedDict


class ArcDataset(object):
    def __init__(self, challenge, solutions={}, keys=None, is_fake=False, is_orig=False):
        if keys is None:
            self.keys = []
            for k, v in challenge.items():
                reply_num = len(v['test'])
                self.keys.extend([f'{k}_{i}' for i in range(reply_num)] if reply_num else [k])
            self.keys = sorted(self.keys)
        else:
            self.keys = [k for k in keys]
        base_keys = set(map(self.get_base_key, self.keys))
        self.challenge = {k: challenge[k] for k in base_keys}
        self.solutions = {k: solutions[k] for k in base_keys if k in solutions}
        self.is_orig = is_fake
        self.is_orig = is_orig

    @classmethod
    def load_from_json(cls, challenges_file):  # for loading challenges in kaggle json arc dataset format
        with open(challenges_file) as f:
            challenge = f.read()
        return cls(
            challenge=json.loads(challenge),
            is_fake=hashlib.md5(challenge.encode('utf-8')).hexdigest().lower() == 'a6b7dac3cab03abf2eb333e16610d6dc',
            is_orig=True,
        )

    def load_solutions(self, solutions_file):  # for loading solutions in kaggle json arc dataset format
        with open(solutions_file) as f: solutions = f.read()
        data = json.loads(solutions)
        solutions = {k: data[k] for k in self.challenge}
        return self.__class__(keys=self.keys, challenge=self.challenge, solutions=solutions, is_orig=self.is_orig)

    # loader for Michael Hodel's ReArc https://github.com/neoneye/arc-dataset-collection
    @classmethod
    def load_from_rearc(cls, path, n, sizes, seed, mix_datasets={}, shuffle=True):  # loader for ReArc
        np.random.seed(seed)
        keys = [[] for _ in range(n)]
        challenge = {}
        solutions = {}
        sizes = list(sizes)

        with open(os.path.join(path, 'metadata.json')) as f:
            metadata = json.load(f)

        for key in tqdm(sorted(metadata.keys()), desc="load dataset 're-arc'"):
            with open(os.path.join(path, 'tasks', f'{key}.json')) as f:
                tasks = np.random.permutation(json.load(f)).tolist()

            next_sizes = []
            for epoch in range(n):
                if not len(next_sizes):
                    next_sizes = np.random.permutation(sizes).tolist()
                next_size_with_test = 1 + next_sizes.pop()
                base_key = f'rearc-{key}{epoch:02x}'
                keys[epoch].append(f'{base_key}_0')
                challenge[base_key] = {'train': [], 'test': []}
                solutions[base_key] = reply = []
                for _ in range(next_size_with_test):
                    if not len(tasks):
                        raise RuntimeError('Not enough examples - generate more re-arc examples or reduce epochs.')
                    challenge[base_key]['train'].append({k: v for k, v in tasks.pop().items()})
                challenge[base_key]['test'].append(challenge[base_key]['train'].pop())
                solutions[base_key].append(challenge[base_key]['test'][-1].pop('output'))

        for name, ds in mix_datasets.items():
            name = cls.base_key_replace_invalid_chars(name)
            for epoch, ds_keys in enumerate(np.array_split(ds.keys, len(keys))):
                keys[epoch].extend([f'{name}-{k}' for k in ds_keys])
            challenge.update({f'{name}-{k}': v for k, v in ds.challenge.items()})
            solutions.update({f'{name}-{k}': v for k, v in ds.solutions.items()})

        if shuffle:
            keys = [np.random.permutation(epoch) for epoch in keys]
        keys = [k for epoch in keys for k in epoch]
        return cls(keys=keys, challenge=challenge, solutions=solutions, is_orig=True)

    # loader for neoneye's format, as used in https://github.com/neoneye/arc-dataset-collection
    @classmethod
    def load_from_neoneye(cls, path):
        pattern = os.path.join(path, 'data', '*', '*.json')
        files = set(glob(pattern))
        for i in itertools.count():
            updated = [fn for fn in files if fn.endswith(f'_v{i + 1}.json')]
            if not updated: break
            for fn in updated:
                files.remove(fn.replace(f'_v{i + 1}.json', ('.json' if i == 1 else f'_v{i}.json')))
        assert len(files), f"No files found for pattern '{pattern}'."
        challenge = {}
        solutions = {}
        assert len(files), 'no files found'
        for fn in tqdm(files, desc=f"load dataset '{os.path.split(path)[-1]}'"):
            with open(fn) as f:
                key = cls.base_key_replace_invalid_chars(os.path.split(fn)[-1].replace('.json', ''))
                challenge[key] = json.load(f)
                solutions[key] = [test_case.pop('output') for test_case in challenge[key]['test']]
        return cls(challenge=challenge, solutions=solutions, is_orig=True)

    def change_keys(self, keys):
        return self.__class__(challenge=self.challenge, solutions=self.solutions, keys=keys)

    def split(self, n, split_seed, **kwargs):
        assert self.is_orig, 'Must be run on original dataset.'
        keys = sorted(self.challenge.keys())
        if split_seed == 'len':
            keys = self.sort_keys_by_len(keys=keys, **kwargs)
        else:
            assert isinstance(split_seed, int)
            assert not kwargs
            np.random.seed(split_seed)
            keys = np.random.permutation(keys)
        split_datasets = []
        for new_keys in np.array_split(keys, n):
            new_challenge = {k: self.challenge[k] for k in new_keys}
            split_datasets.append(self.__class__(challenge=new_challenge, solutions=self.solutions, is_orig=True))
        return split_datasets

    def remove_test_data(self):
        assert self.is_orig, 'Must be run on original dataset.'
        new_challenge = {k: {'train': v['train'], 'test': []} for k, v in self.challenge.items()}
        return self.__class__(challenge=new_challenge)

    @staticmethod
    def base_key_replace_invalid_chars(base_key):
        return base_key.replace('_', '-').replace('.', '-')

    @staticmethod
    def get_base_key_and_reply_num(key):
        key_num = key.split('.', 1)[0]
        base_key, reply_num = key_num.split('_') if '_' in key_num else (key_num, -1)
        return base_key, int(reply_num)

    @classmethod
    def get_base_key(cls, key):
        return cls.get_base_key_and_reply_num(key)[0]

    def grouped_keys(self):
        grouped_keys = OrderedDict()
        for key in self.keys:
            base_key, reply_num = self.get_base_key_and_reply_num(key)
            if base_key not in grouped_keys:
                grouped_keys[base_key] = []
            while len(grouped_keys[base_key])<=reply_num:
                grouped_keys[base_key].append([])
            grouped_keys[base_key][reply_num].append(key)
        return grouped_keys

    def move_test_to_train(self):
        assert self.is_orig, 'Must be run on original dataset.'
        new_challenge = {}
        for k, v in self.challenge.items():
            new_challenge[k] = {
                'train': v['train'] + [{**t, 'output': self.solutions[k][i]} for i, t in enumerate(v['test'])],
                'test': []
            }
        return self.__class__(challenge=new_challenge, is_orig=self.is_orig)

    @staticmethod
    def permute_array(a, descriptor, invert=False):
        permutation = [int(i) for i in descriptor if str(i).isdigit()]
        assert sorted(permutation) == list(range(10))
        a = np.asarray(a)
        assert a.ndim == 2
        if invert: permutation = np.argsort(permutation)
        a = np.asarray(permutation)[a]
        return a

    @classmethod
    def transform_array(cls, array, transforms, apply_perm=True, invert=False):
        if array is None: return None
        array = np.asarray(array)
        if invert: transforms = transforms[::-1]
        for tf in transforms:
            if tf == 'tp':
                array = np.swapaxes(array, 0, 1)
            if tf == 'rt':
                array = np.rot90(np.rot90(np.rot90(array)) if invert else array)
            if apply_perm and tf.startswith('perm'):
                array = cls.permute_array(array, tf, invert=invert)
        return array

    @classmethod
    def fmt_array(cls, array, lines_sep, tf=None):
        if tf is not None:
            array = cls.transform_array(array, tf)
        return lines_sep.join(''.join(map(str, row)) for row in array)

    @classmethod
    def fmt_input(cls, array, query_beg, reply_beg, **kwargs):
        return query_beg + cls.fmt_array(array, **kwargs) + reply_beg

    @classmethod
    def fmt_output(cls, array, reply_end, **kwargs):
        return cls.fmt_array(array, **kwargs) + reply_end

    @classmethod
    def fmt_train(cls, train_ex, preprompt, query_beg, reply_beg, reply_end, **kwargs):
        examples = [cls.fmt_input(x['input'], query_beg, reply_beg, **kwargs) +
                    cls.fmt_output(x['output'], reply_end, **kwargs) for x in train_ex]
        return preprompt + ''.join(examples)

    def fmt_task(self, key, preprompt, query_beg, reply_beg, reply_end, reply=True, **kwargs):
        key_num, *tf = key.split('.')
        base_key, reply_num = self.get_base_key_and_reply_num(key_num)
        data_train = self.challenge[base_key]['train']
        data_query = self.challenge[base_key]['test']
        if reply is True:
            reply = self.solutions[base_key][reply_num] if base_key in self.solutions and reply_num >= 0 else None
        elif reply is not None:
            assert reply_num >= 0
        for t in tf:
            if t.startswith('ex'):
                data_train = [data_train[int(i)] for i in t[2:].split('-')]
        ret = dict(key=key)
        ret['train'] = self.fmt_train(data_train, preprompt, query_beg, reply_beg, reply_end, tf=tf, **kwargs)
        ret['query'] = self.fmt_input(data_query[reply_num]['input'], query_beg, reply_beg, tf=tf, **kwargs) if reply_num >= 0 else ''
        ret['input'] = ret['train'] + ret['query'] if reply_num >= 0 else ''
        if reply is not None:
            ret['reply'] = self.fmt_output(reply, reply_end, tf=tf, **kwargs)
        ret['text'] = ret['train'] + (ret['query'] + ret['reply'] if reply is not None else '')
        return ret

    def get_task(self, key, max_tokens=None, len_name=None, **kwargs):
        while True:
            fmt = self.fmt_task(key, **kwargs)
            if max_tokens is None or self.count_tokens(fmt[len_name]) <= max_tokens:
                break
            if not key.split('.')[-1].startswith('ex'):
                base_key = self.get_base_key(key)
                key = f"{key}.ex{'-'.join(map(str, range(len(self.challenge[base_key]['train']))))}"
            key_split = key.split('.')
            key_split[-1] = '-'.join(key_split[-1].split('-')[:-1])
            assert len(key_split[-1]) > 2 and key_split[-1].startswith('ex')
            key = '.'.join(key_split)
        return key, fmt

    def repeat(self, n, seed=None):
        if seed is not None:
            np.random.seed(seed)
        new_keys = []
        for i in range(n):
            new_keys.extend(self.keys if seed is None else np.random.permutation(self.keys))
        return self.change_keys(new_keys)

    @staticmethod
    def count_tokens(data, replace_special=re.compile('<[^<]*>')):
        replaced = replace_special.sub('x', data)  # replace '<...>' by a single char to count special tokens only once
        return len(replaced)

    @classmethod
    def max_new_tokens(cls, reply_end, lines_sep, max_size=30, safety_margin=1, **_):
        max_sized_reply = np.zeros([max_size, max_size], dtype=int)
        fmt = cls.fmt_output(max_sized_reply, reply_end=reply_end, lines_sep=lines_sep)
        return cls.count_tokens(fmt) + safety_margin

    def get_length(self, key, len_name, max_of_transposed=False, max_tokens=None, **fmt_opts):
        if not fmt_opts:
            fmt_opts = dict(preprompt='', query_beg='', reply_beg='', reply_end='', lines_sep='')
            length = self.count_tokens(self.fmt_task(key, **fmt_opts)[len_name])
        else:
            length = self.count_tokens(self.fmt_task(key, **fmt_opts)[len_name])
            if max_of_transposed:
                length = max(length, self.count_tokens(self.fmt_task(f'{key}.tp', fmt_opts)[len_name]))
            length += 1  # for bos token
        return length

    def sort_keys_by_len(self, keys, reverse=False, **kwargs):
        lengths = [(key, self.get_length(key, **kwargs)) for key in keys]
        return [x[0] for x in sorted(lengths, reverse=reverse, key=lambda x: x[1])]

    def sorted_by_len(self,**kwargs):
        return self.change_keys(self.sort_keys_by_len(self.keys, **kwargs))

    def convert_with_token_limit(self, **kwargs):
        out_list = []
        new_keys = []
        for key in tqdm(self.keys, desc='convert dataset'):
            key, fmt = self.get_task(key, **kwargs)
            new_keys.append(key)
            out_list.append(fmt)
        return out_list, self.change_keys(new_keys)

    def as_list(self, **kwargs):
        return self.convert_with_token_limit(**kwargs)[0]

    @staticmethod
    def rand_perm(n, sep=None, keep_zero=False):
        permutation = np.random.permutation(n).tolist()
        if keep_zero:
            permutation = [0] + [x for x in permutation if x != 0]
        return permutation if sep is None else sep.join(map(str, permutation))

    def augment_keys(self, keys, tp=False, rt=False, n=1, perm=False, keep_background=False, shfl_ex=False):
        keys = [k + n * '.tp' for n in range(2) for k in keys] if tp == 'all' else keys
        keys = [k + n * '.rt' for n in range(4) for k in keys] if rt == 'all' else keys
        keys = [k + bool(tp) * randint(0, 2) * '.tp' for k in keys] if tp != 'all' else keys
        keys = [k + bool(rt) * randint(0, 4) * '.rt' for k in keys] if rt != 'all' else keys
        keys = keys * n  # repeat n times
        keys = [k + bool(perm) * ('.perm' + self.rand_perm(10, '', keep_background)) for k in keys]
        n_ex = lambda k: len(self.challenge[self.get_base_key(k)]['train'])
        keys = [k + bool(shfl_ex) * ('.ex' + self.rand_perm(n_ex(k), '-')) for k in keys]
        return keys

    def augment(self, seed, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        return self.change_keys([k for key in self.keys for k in self.augment_keys([key], **kwargs)])

    def decode(self, text, lines_sep, key=None):
        correct, info = None, 'unknown'
        try:
            data = [[int(x) for x in row if x.isdigit()] for row in text.split(lines_sep)]
            data = [row for row in data if len(row)]
            data = np.array(data, dtype=int)
            assert data.ndim == 2 and all(0 < x <= 30 for x in data.shape)
        except:
            data = None
            correct, info = False, 'cant_decode'
        if key is not None and data is not None:
            key_num, *transforms = key.split('.')
            base_key, reply_num = self.get_base_key_and_reply_num(key_num)
            data = self.transform_array(data, transforms, invert=True)
            correct_solution = self.solutions.get(base_key)
            if correct_solution is None:
                info = 'sol_unknown'
            else:
                correct_solution = np.asarray(correct_solution[reply_num])
                if np.array_equal(correct_solution, data):
                    correct, info = True, 'ALL_CORRECT'
                else:
                    correct, info = False, ('bad_content' if correct_solution.shape == data.shape else 'bad_xy_size')
        return data, correct, info

    def get_submission(self, results=None):
        assert self.is_orig, 'Must be run on original dataset.'
        submission = {k: [{f'attempt_{i+1}': [[0]] for i in range(2)} for _ in range(len(v['test']))] for k, v in self.challenge.items()}
        if results is not None:
            self.fill_submission(results, submission)
        return submission

    @staticmethod
    def fill_submission(results, submission):
        for base_key, data in results.items():
            for reply_num, guesses in enumerate(data):
                target_dict = submission[base_key][reply_num]
                for i, g in enumerate(guesses[:len(target_dict)]):
                    target_dict[f'attempt_{i + 1}'] = g['output'].tolist()

    def validate_submission(self, submission):
        assert self.is_orig, 'Must be run on original dataset.'
        assert self.solutions, 'Solutions must be loaded for submission verification.'
        score = 0
        for k, v in self.solutions.items():
            for i, r in enumerate(v):
                for attempt in ['attempt_1', 'attempt_2']:
                    if np.array_equal(r, submission[k][i][attempt]):
                        score += 1 / len(v)
                        break
        return score


import os
import json
import torch
import peft
from tokenizers import Tokenizer
from huggingface_hub import snapshot_download
from trl import DataCollatorForCompletionOnlyLM


class InputMaskingDataCollator(DataCollatorForCompletionOnlyLM):
    def __init__(self, mask_first_n_examples=0, **kwargs):
        super().__init__(**kwargs)
        self.mask_first_n_examples = mask_first_n_examples

    def torch_call(self, examples):
        batch = super().torch_call(examples)  # call super, masking all inputs
        for i in range(len(batch['labels'])):
            for _ in range(self.mask_first_n_examples):
                # mask first still unmasked output block
                beg_pos = ((batch['labels'][i] != -100).nonzero().min()).item()
                mid_pos = ((batch['labels'][i][beg_pos:] == -100).nonzero().min()).item() + beg_pos
                end_pos = ((batch['labels'][i] != -100).nonzero().max()).item() + 1
                if mid_pos < end_pos:
                    batch['labels'][i][beg_pos:mid_pos] = -100
        return batch


def load_unsloth_4bit(model_path):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        dtype=None,
        load_in_4bit=True,
    )
    if model.max_seq_length == 2048 < model.generation_config.max_length:
        print(f'CHANGING MAX_SEQ_LENGTH {model.max_seq_length} -> {model.generation_config.max_length} (unsloth bug?)')
        to_fix = model
        while to_fix is not None:
            to_fix.max_seq_length = model.generation_config.max_length
            to_fix = getattr(to_fix, 'model', None)
    return model, tokenizer


def save_model_and_tokenizer(store_path, model, tokenizer):
    model.save_pretrained(store_path)
    tokenizer.save_pretrained(store_path)
    to_delete = os.path.join(store_path, 'tokenizer.model')  # delete file, as it interferes with token removal
    if os.path.isfile(to_delete):
        os.remove(to_delete)


def fix_dtypes(model, fix_weights=True, fix_quant_states=True):
    # fix some data types (workaround for unsloth)
    for module in model.modules():
        weight = getattr(module, 'weight', None)
        if weight is not None:
            if torch.is_floating_point(weight):
                if fix_weights and weight.dtype != model.dtype:
                    module.to(model.dtype)
            else:
                qs = getattr(weight, 'quant_state', None)
                if qs is not None:
                    if fix_quant_states and qs.dtype != model.dtype:
                        qs.dtype = model.dtype
    return model


def is_peft_model(model):
    return hasattr(model, 'peft_type')


def merge_peft_into_base(model):
    assert is_peft_model(model)
    return fix_dtypes(model.merge_and_unload())


def get_and_fix_peft_weights(store):
    # change some keys (workaround for added 'modules_to_save')
    state_dict = peft.load_peft_weights(store)
    for k in list(state_dict.keys()):
        if 'modules_to_save' in k:
            del state_dict[k]
            original_module_key = k.replace('.modules_to_save.', '.original_module.')
            if original_module_key in state_dict: del state_dict[original_module_key]
            assert k.replace('.modules_to_save.', '.') in state_dict
    return state_dict


def set_peft_weights(model, state_dict):
    res = peft.set_peft_model_state_dict(model, state_dict)
    assert not res.unexpected_keys, 'error loading weights - some keys not available in model'


def load_peft_state(model, store):
    # convenience method to load peft weights from file and set them for model
    set_peft_weights(model, get_and_fix_peft_weights(store))


def get_or_map_special_tokens(data, mapping=None):
    tokens = set()
    if isinstance(data, dict):
        special = data.get('special_tokens')
        if special is not None:  # find and/or update special token mappings
            for v in special.values():
                tokens.update(v['ids'])
                if mapping is not None:
                    v['ids'] = [mapping.get(i) for i in v['ids'] if i in mapping]
        for v in data.values():  # recursively process dict values
            tokens.update(get_or_map_special_tokens(v, mapping))
    if isinstance(data, list):
        for v in data:  # recursively process lists
            tokens.update(get_or_map_special_tokens(v, mapping))
    return tokens


def remove_tokenizer_normalizer(tokenizer):
    assert tokenizer.is_fast
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    if tokenizer_json.get('normalizer') is not None:
        tokenizer_json['normalizer'] = None
        tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))


def shrink_tokenizer_vocab(tokenizer, keep_indices, keep_special=True, remove_unk=False):
    assert tokenizer.is_fast
    tok_json = json.loads(tokenizer._tokenizer.to_str())
    assert tok_json['model']['type'] == "BPE"

    if keep_special:  # get special tokens to keep
        keep_indices.update(tokenizer.all_special_ids)
        keep_indices.update(get_or_map_special_tokens(tok_json.get('post_processor')))

    if remove_unk:  # remove unknown token
        keep_indices -= {tokenizer.unk_token_id}

    # build mapping from old to new id
    mapping = {old: new for new, old in enumerate(sorted(keep_indices))}

    # update tokenizer info
    tok_json['model']['vocab'] = {k: mapping[v] for k, v in tok_json['model']['vocab'].items() if v in mapping}
    tok_json['model']['merges'] = []
    tok_json['added_tokens'] = [{**t, 'id': mapping[t['id']]} for t in tok_json['added_tokens'] if t['id'] in mapping]
    tok_json['added_tokens'] = sorted(tok_json['added_tokens'], key=lambda t: t['id'])
    get_or_map_special_tokens(tok_json.get('post_processor'), mapping)

    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tok_json))  # reload json, modifying tokenizer in-place

    if remove_unk:
        tokenizer.unk_token = None

    return mapping  # token mapping to be used later


def shrink_model_embeddings(model, mapping):
    with torch.no_grad():
        # copy embeddings to keep
        row_select = torch.tensor([x[0] for x in sorted(mapping.items(), key=lambda x: x[1])])
        row_select = row_select.to(model.get_input_embeddings().weight.data.device)
        new_embed_t = torch.index_select(model.get_input_embeddings().weight.data, 0, row_select)
        row_select = row_select.to(model.get_output_embeddings().weight.data.device)
        new_lm_head = torch.index_select(model.get_output_embeddings().weight.data, 0, row_select)

        # resize model embeddings
        model.resize_token_embeddings(len(row_select))

        # set to copied values
        model.get_input_embeddings().weight.data[:] = new_embed_t
        model.get_output_embeddings().weight.data[:] = new_lm_head

        # map model tokens to new id
        for config in [model.config, model.generation_config]:
            for k, v in list(config.to_dict().items()):
                if k.endswith('token_id'):
                    setattr(config, k, [mapping.get(t) for t in v] if isinstance(v, list) else mapping.get(v))


def keep_single_char_tokens(model, tokenizer, keep=None, keep_norm=False, keep_model_tok=True, **kwargs):
    if not keep_norm:
        remove_tokenizer_normalizer(tokenizer)  # required for some models
    if keep is None:  # keep all single_length tokens
        keep_indices = set(v for k, v in tokenizer.vocab.items() if len(k) == 1)
    else:  # keep tokens that were passed
        keep_indices = set(tokenizer.vocab[t] for t in keep)
    if keep_model_tok:  # keep tokens used by model
        for config in [model.config, model.generation_config]:
            for k, v in config.to_dict().items():
                if k.endswith('token_id'):
                    keep_indices.update(v if isinstance(v, list) else [v])
    keep_indices -= {None}
    mapping = shrink_tokenizer_vocab(tokenizer, keep_indices, **kwargs)
    shrink_model_embeddings(model, mapping)
    return mapping

# Copyright 2024 Daniel Franzen and Jan Disselhoff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import time
import shutil
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import Dataset
from datetime import datetime

import logging

os.environ["HF_HOME"] = "/ocean/projects/cis250063p/jbentley/ARC-AGI-2/Capstone-ARC2/shared/arc/cache/hf"
os.environ["TRANSFORMERS_CACHE"] = "/ocean/projects/cis250063p/jbentley/ARC-AGI-2/Capstone-ARC2/shared/arc/cache/hf"
os.environ["HF_DATASETS_CACHE"] = "/ocean/projects/cis250063p/jbentley/ARC-AGI-2/Capstone-ARC2/shared/arc/cache/ds"

# Figure out where to put logs
output_root = "/ocean/projects/cis250063p/jbentley/ARC-AGI-2/Capstone-ARC2/shared/arc/outputs"
log_dir = os.path.join(output_root, "logs")
os.makedirs(log_dir, exist_ok=True)

# Timestamped log file
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_file = os.path.join(log_dir, f"train_v1-{timestamp}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),            # goes to stdout (SLURM captures this in .out)
        logging.FileHandler(log_file, mode="w")  # persistent log file
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Logging to {log_file}")

base_path = '/ocean/projects/cis250063p/jbentley/ARC-AGI-2/Capstone-ARC2/shared/arc/data/ARC-Data/input'
# input paths
base_model = 'chuanli11/Llama-3.2-3B-Instruct-uncensored'  # auto-downloaded from huggingface.co
arc_data_path = os.path.join(base_path, 'arc-prize-2024')  # as on kaggle arc prize 2024
re_arc_path = os.path.join(base_path, 're_arc')  # https://github.com/michaelhodel/re-arc
neoneye_path = os.path.join(base_path, 'arc-dataset-collection')  # https://github.com/neoneye/arc-dataset-collection)

# output paths
base_model_path = "/ocean/projects/cis250063p/jbentley/ARC-AGI-2/Capstone-ARC2/shared/arc/outputs/models"
save_model_path = os.path.join(base_model_path, "Llama-3")

import torch.distributed as dist, os
rank = int(os.environ.get("RANK", -1))
world = int(os.environ.get("WORLD_SIZE", 1))
logger.info(f"RANK={rank} WORLD_SIZE={world} DIST_INIT={dist.is_available() and dist.is_initialized()}")

def download_model(model_name):
    """Pre-download the model to ensure it's available before training starts"""
    logger.info(f"Pre-downloading model: {model_name}")
    start_time = time.time()
    
    # Download tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Downloaded tokenizer in {time.time() - start_time:.2f} seconds")
    
    # Then download model without loading it into memory
    try:
        # Just download the model files without loading the model
        AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            local_files_only=False
        )
        logger.info(f"Successfully pre-downloaded model in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error pre-downloading model: {e}")
    
    return tokenizer

def check_dataset_availability():
    """Check if datasets are available and accessible"""
    print("Checking dataset availability...")
    
    data_paths = [
        (arc_data_path, "ARC Prize Dataset"),
        (re_arc_path, "RE-ARC Dataset"),
        (neoneye_path, "NeonEye Dataset Collection")
    ]
    
    all_available = True
    for path, name in data_paths:
        if os.path.exists(path):
            print(f"✓ {name} found at {path}")
            # Check for specific required files based on the dataset
            if name == "ARC Prize Dataset":
                eval_file = os.path.join(path, 'arc-agi_evaluation_challenges.json')
                solutions_file = os.path.join(path, 'arc-agi_evaluation_solutions.json')
                if not (os.path.exists(eval_file) and os.path.exists(solutions_file)):
                    print(f"⚠ {name} is missing required files")
                    all_available = False
            elif name == "NeonEye Dataset Collection":
                concept_path = os.path.join(path, 'dataset', 'ConceptARC')
                if not os.path.exists(concept_path):
                    print(f"⚠ ConceptARC not found in {name}")
                    all_available = False
        else:
            print(f"✗ {name} not found at {path}")
            all_available = False
    
    return all_available


def load_model_and_tokenizer(model_name):
    """Load model and tokenizer for 4-bit quantization"""
    logger.info(f"Loading model and tokenizer from {model_name}")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Make sure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure 4-bit quantization with BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))     # set by torchrun
    torch.cuda.set_device(local_rank)
    
    # Load the model with quantization config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map={"": local_rank},
    )
    
    # Prepare model for training with 4-bit quantization
    model = prepare_model_for_kbit_training(model)
    logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    return model, tokenizer

def _get_local_rank():
    # works with torchrun/DDP and also in single-proc runs
    try:
        return int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError:
        return 0

def _is_main_process():
    return int(os.environ.get("RANK", "0")) in (-1, 0)


def merge_lora_weights(base_model_path, adapter_path, output_path):
    """Merge LoRA weights into the base model and save"""
    logger.info(f"Merging LoRA weights from {adapter_path} into base model")
    start_time = time.time()

    if not _is_main_process():
        logger.info("Skipping merge on non-main rank")
        return

    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"LoRA adapter folder not found: {adapter_path}")

    local_rank = _get_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    
    # Load the base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ).to(device)

    logger.info("Re-applying embedding size reduction for merging")
    keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=') + tokenizer.tokenize('\n')
    keep_single_char_tokens(model, tokenizer, keep=keep_tok, remove_unk=True)
    
    # Load and merge LoRA weights
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    
    # Save the merged model
    save_model_and_tokenizer(output_path, model, tokenizer)
    logger.info(f"LoRA weights merged in {time.time() - start_time:.2f} seconds")
    
    return model, tokenizer

def create_output_dirs():
    """Create necessary output directories"""
    dirs = [
        '/ocean/projects/cis250063p/jbentley/ARC-AGI-2/Capstone-ARC2/shared/arc/outputs/runs/tmp_output',  # trainer output_dir
        os.path.dirname(save_model_path)
    ]
    
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")

def main():
    """Main execution function with pre-downloads and checks"""
    logger.info("Starting training preparation")
    
    # Create necessary directories
    create_output_dirs()
    
    # First download the model
    logger.info("=== Step 1: Pre-downloading model ===")
    tokenizer = download_model(base_model)
    
    # Check if datasets are available
    logger.info("=== Step 2: Checking datasets ===")
    datasets_available = check_dataset_availability()
    if not datasets_available:
        logger.warning("Some datasets are not available. This may affect training.")
    
    # Continue with regular training process
    for action in ['train', 'merge']:
        # continue if task already accomplished
        if action == 'train' and os.path.exists(f'{save_model_path}-lora'):
            logger.info(f"Skipping {action} as {save_model_path}-lora already exists")
            continue
        if action == 'merge' and os.path.exists(f'{save_model_path}-merged'):
            logger.info(f"Skipping {action} as {save_model_path}-merged already exists")
            continue

        logger.info(f"=== Starting {action} phase ===")
        
        # load base model & reduce embedding size
        model = tokenizer = None  # free memory
        model, tokenizer = load_model_and_tokenizer(base_model)
        
        logger.info("Reducing embedding size to only necessary tokens")
        keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=')+tokenizer.tokenize('\n')
        keep_single_char_tokens(model, tokenizer, keep=keep_tok, remove_unk=True)

        # set formatting options
        fmt_opts = dict(
            preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
            query_beg='I',
            reply_beg='\n+/-=O',
            reply_end='\n' + tokenizer.eos_token,
            lines_sep='\n',
            max_tokens=8192,
        )

        if action == 'train':
            # Configure LoRA
            logger.info("Configuring LoRA adapter")
            lora_config = LoraConfig(
                r=256,
                lora_alpha=24,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)
            
            # load training data
            logger.info("Loading and preparing training data")
            arc_eval_set = ArcDataset.load_from_json(os.path.join(arc_data_path, 'arc-agi_evaluation_challenges.json'))
            arc_eval_set = arc_eval_set.load_solutions(os.path.join(arc_data_path, 'arc-agi_evaluation_solutions.json'))
            concept_arc = ArcDataset.load_from_neoneye(os.path.join(neoneye_path, 'dataset', 'ConceptARC'))
            mix_datasets = {
                            'arceval': arc_eval_set.move_test_to_train().repeat(32),
                            'concept': concept_arc.move_test_to_train().repeat(32),
                }
            train_dataset = ArcDataset.load_from_rearc(re_arc_path, n=60, sizes=[3], seed=42, mix_datasets=mix_datasets)

            # augment data set and transform to list
            logger.info("Augmenting training data")
            train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
            train_dataset_augment = train_dataset.augment(**train_aug_opts)
            # train_dataset_as_list = train_dataset_augment.as_list(len_name='text', **fmt_opts)

            # First get the raw text samples
            text_samples = [ex['text'] for ex in train_dataset_augment.as_list(len_name='text', **fmt_opts)]

            # Then convert them to the format expected by the tokenizer
            def tokenize_function(examples):
                tokenized = tokenizer(examples['text'], padding=False, truncation=True, max_length=fmt_opts['max_tokens'])
                return tokenized

            # Create the dataset with the text field
            raw_dataset = Dataset.from_dict({"text": text_samples})

            # Then tokenize it
            tokenized_dataset = raw_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"]
            )

            # logger.info(f"Prepared {len(train_dataset_as_list)} training examples")
            # print(f"First example: {train_dataset_as_list[0]}")            
    
            

            # run training with DeepSpeed
            tokenizer.padding_side = 'right'
            training_args = TrainingArguments(
                output_dir='/ocean/projects/cis250063p/jbentley/ARC-AGI-2/Capstone-ARC2/shared/arc/outputs/runs/tmp_output',
                num_train_epochs=1,
                per_device_train_batch_size=1,  # Increased to match DeepSpeed config
                gradient_accumulation_steps=8,  # No need for gradient accumulation
                warmup_ratio=0.25,
                learning_rate=1e-4,
                weight_decay=0.00,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=10,
                save_strategy='no',
                report_to='none',
                seed=42,
                remove_unused_columns=False,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                ddp_find_unused_parameters=False,

            )
            
            logger.info("Starting training with Normal Torch")
            trainer = Trainer(
                model=model,
                args=training_args,
                # train_dataset=Dataset.from_list(train_dataset_as_list),
                train_dataset=tokenized_dataset,
                data_collator=InputMaskingDataCollator(
                    instruction_template=fmt_opts['query_beg'],
                    response_template=fmt_opts['reply_beg'],
                    mlm=False,
                    tokenizer=tokenizer,
                    mask_first_n_examples=1,
                ),
            )
            
            trainer.train()
            logger.info(f"Training complete, saving model to {save_model_path}-lora")
            model.save_pretrained(f'{save_model_path}-lora')
            tokenizer.save_pretrained(f'{save_model_path}-lora')

        if action == 'merge':
            # load peft weights and merge
            logger.info("Merging LoRA weights into base model")
            merge_lora_weights(base_model, f'{save_model_path}-lora', f'{save_model_path}-merged')

    logger.info("Training process complete")

if __name__ == "__main__":
    main()