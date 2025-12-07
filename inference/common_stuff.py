# common configuration for training and evaluation
from arc_loader import *
from model_runner import *
from selection import *
from async_tools import *
import time
import random
import numpy as np
import torch


GLOBAL_SEED = 42


def set_all_seeds(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


set_all_seeds()

# paths
tmp_dir = '/kaggle/temp'
arc_challenge_file = '/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json'
arc_solutions_file = '/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json'
model_temp_storage = os.path.join(tmp_dir, 'finetuned_model')
infer_temp_storage = os.path.join(tmp_dir, 'inference_outputs')
score_temp_storage = os.path.join(tmp_dir, 'inference_scoring')

# load datasets
arc_test_set = ArcDataset.from_file(arc_challenge_file)
# if arc_test_set.is_fake: arc_test_set.load_replies(arc_solutions_file)
arc_test_set.is_fake = False  # force full run
# arc_train_set = ArcDataset.from_file('/kaggle/input/arc-prize-2024/arc-agi_training_challenges.json')

# models
MyFormatter, perm_aug, max_seq_length_train, mask_first = ArcFormatter_premix_3, 'rnd_all', 4224, 1

# training & inference
train_epochs = 4
multi_gpu_train = True
multi_gpu_random_split = False
max_seq_length_infer = 8192
prime_on_single_task = True
num_active_layers = 32
infer_params = dict(min_prob=0.5, store=infer_temp_storage, use_turbo=True)

# scoring
use_aug_score = True
aug_score_params = dict(tp=True, rot=True, perm=perm_aug, shfl_ex=True, make_unique=True, max_len=max_seq_length_infer)
submission_select_algo = score_full_probmul_3 if use_aug_score else score_all_probsum


def prepare_run(model_path, load_lora=None, train=False, gpu=None, **kwargs):
    seed = GLOBAL_SEED + (0 if gpu is None else gpu)
    set_all_seeds(seed)

    if gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    model_kwargs = dict(max_seq_length=max_seq_length_train)
    model_kwargs.update(kwargs)

    model, tokenizer, formatter = prepare_model(
        model=model_path,
        local_files_only=True,
        mode='unsloth_4bit',
        formatter=MyFormatter,
        peft=([dict(
            r=32,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'embed_tokens', 'lm_head'],
            lora_alpha=128,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=42,
            use_rslora=True,
            loftq_config=None,
        )] if train or load_lora else []) + ([load_lora] if load_lora else []),
        num_active_layers=(num_active_layers if train else None),
        **model_kwargs
    )

    if train and mask_first: formatter.collator_kwargs.update(mask_first_output=mask_first)

    return model, formatter


def prepare_dataset(formatter, train, gpu=None):
    seed = GLOBAL_SEED + (0 if gpu is None else gpu)
    set_all_seeds(seed)

    ds = arc_test_set
    if multi_gpu_train and gpu is not None:
        if multi_gpu_random_split:
            all_keys_shuffled = ds.shuffled(seed=123).keys
            num_total_keys = len(all_keys_shuffled)
            base_quarter_size = num_total_keys // 4
            start_index = gpu * base_quarter_size
            if gpu < 3:
                end_index = (gpu + 1) * base_quarter_size
            else:
                end_index = num_total_keys
            gpu_specific_keys = all_keys_shuffled[start_index:end_index]
            ds = ds.change_keys(gpu_specific_keys, keep_flags=True)
        else:
            ds = ds.sorted_by_len(formatter=formatter, name='input', max_of_transposed=True)
            # 4-GPU rotation pattern instead of 2-GPU
            assignment = ([0, 1, 2, 3] * ds.length())[:ds.length()][::-1]
            ds = ds.change_keys((np.array(ds.keys)[np.array(assignment) == gpu]).tolist())

    if arc_test_set.is_fake: ds.keys = ds.keys[:1]
    # Rest of the function remains the same
    if train:
        ds = ds.remove_replies()
        ds = ds.augment(tp=True, rot=True, perm=perm_aug, n=(2 if arc_test_set.is_fake else train_epochs), shfl_ex=True, shfl_keys=True)
        ds = ds.cut_to_len(formatter=formatter, name='text', max_len=max_seq_length_train, max_new_tokens=0, quiet=True)
        if arc_test_set.is_fake: ds = ds.sorted_by_len(formatter=formatter, name='text', reverse=True)
        print(len(ds.keys))
    else:
        ds = ds.sorted_by_len(formatter=formatter, name='input', max_of_transposed=True)
        ds = ds.split_multi_replies()
        ds = ds.augment(tp=True, rot=True, n=2, seed=42, perm=perm_aug, shfl_ex=True).interleave(ds.length())
        ds = ds.cut_to_len(formatter=formatter, name='input', max_len=max_seq_length_infer, quiet=True)
        print(len(ds.keys))
        grouped_keys = {}
        for key in ds.keys:
            base_key = key.split('.')[0]
            if base_key not in grouped_keys:
                grouped_keys[base_key] = []
            grouped_keys[base_key].append(key)
        final_keys = []
        for base_key in sorted(grouped_keys.keys()):
            group = grouped_keys[base_key]
            permuted_group = np.random.permutation(group).tolist()
            final_keys.extend(permuted_group[:5])
        ds = ds.change_keys(final_keys)
    return ds


def start_training(gpu):
    seed = GLOBAL_SEED + gpu
    set_all_seeds(seed)
    base_model = '/kaggle/input/mistral-hybrid/transformers/default/1/namannn/mistral-hybrid'

    try:
        storage_path = f'{model_temp_storage}_gpu{gpu}'
        if gpu == 0 or multi_gpu_train:
            with RemapCudaOOM():
                model, formatter = prepare_run(base_model, train=True, gpu=gpu)
                dataset = prepare_dataset(formatter, train=True, gpu=gpu if multi_gpu_train else None)
                model, trainer_stats = training_run(
                    model, formatter, dataset, store=storage_path,
                    max_seq_length=max_seq_length_train,
                    grad_acc_fix=False,
                    train_args=dict(
                        per_device_train_batch_size=8,
                        gradient_accumulation_steps=1,
                        warmup_steps=48,
                        num_train_epochs=1,
                        max_steps=5 if arc_test_set.is_fake else 240,
                        learning_rate=1e-4,
                        embedding_learning_rate=1e-5,
                        logging_steps=10,
                        optim="adamw_8bit",
                        weight_decay=0.01,
                        lr_scheduler_type='cosine',  # "linear", "cosine",
                        seed=42,
                        output_dir=os.path.join(tmp_dir, 'checkpoints'),
                        save_strategy="no",
                        report_to='none',
                    ),
                )
                mem_info()
    finally:
        os.makedirs(f'{storage_path}_done', exist_ok=True)


def start_inference(gpu):
    seed = GLOBAL_SEED + gpu + 100
    set_all_seeds(seed)

    storage_path = f'{model_temp_storage}_gpu{gpu if multi_gpu_train else 0}'
    while not os.path.exists(f'{storage_path}_done'): time.sleep(15)
    with RemapCudaOOM():
        model, formatter = prepare_run(storage_path, gpu=gpu)
        dataset = prepare_dataset(formatter, train=False, gpu=gpu)
        retrainer = None if not prime_on_single_task else Retrainer(
            n=128,
            aug_opts=dict(tp=True, rot=True, perm=perm_aug, shfl_ex=True),
            reload_state_dict=get_and_fix_peft_weights(storage_path),
            formatter=formatter,
            max_seq_length=max_seq_length_infer,
            grad_acc_fix=False,
            train_args=dict(
                per_device_train_batch_size=8,
                gradient_accumulation_steps=1,
                warmup_steps=4,
                num_train_epochs=1,
                learning_rate=5e-5,
                embedding_learning_rate=0,
                logging_steps=8,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type='constant',  # "linear", "cosine",
                seed=42,
                output_dir='tmp_output',
                save_strategy='no',
                report_to='none',
            ),
        )
        decoder = Decoder(formatter, arc_test_set.split_multi_replies(), n_guesses=2, prob_baseline=0.05)
        inference_run_v2(model, formatter, dataset, decoder, retrain=retrainer, **infer_params)
        if use_aug_score or arc_test_set.is_fake: decoder.calc_augmented_scores(model=model, store=score_temp_storage, **aug_score_params)
        mem_info()


class RemapCudaOOM:
    def __enter__(self): pass

    def __exit__(self, exc_type, exc_value, traceback):
        oom_errors = ["CUDA out of memory", "Make sure you have enough GPU RAM", "does not fit any GPU's remaining memory"]
        if exc_value and any(x in str(exc_value) for x in oom_errors):
            with open('submission.json', 'w') as f: f.write('cause submission scoring error')
