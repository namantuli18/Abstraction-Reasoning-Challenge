from unsloth import FastLanguageModel
from peft import LoraConfig



def load_unsloth_model(
    base_model_id: str = "meta-llama/Llama-3.1-8B",
    use_qlora: bool = True,        # 4-bit QLoRA
    r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05,
    target_modules=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"),
):
    model, tokenizer = FastLanguageModel.from_pretrained(
        base_model_id,
        load_in_4bit=use_qlora,
        device_map="auto"
    )

    lora_cfg = LoraConfig(
        r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=list(target_modules),
        bias="none", task_type="CAUSAL_LM"
    )

    model = FastLanguageModel.get_peft_model(model,lora_cfg)

    FastLanguageModel.for_training(model, max_seq_length=4096)

    return model, tokenizer