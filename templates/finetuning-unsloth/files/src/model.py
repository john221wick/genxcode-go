from unsloth import FastLanguageModel

from .config import MODEL_CONFIG


def load_model_and_tokenizer():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_CONFIG["model_name"],
        max_seq_length=MODEL_CONFIG["max_seq_length"],
        load_in_4bit=MODEL_CONFIG["load_in_4bit"],
        dtype=MODEL_CONFIG["dtype"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=MODEL_CONFIG["lora_r"],
        lora_alpha=MODEL_CONFIG["lora_alpha"],
        lora_dropout=MODEL_CONFIG["lora_dropout"],
        target_modules=MODEL_CONFIG["lora_target_modules"],
        use_gradient_checkpointing=MODEL_CONFIG["use_gradient_checkpointing"],
        use_rslora=MODEL_CONFIG["use_rslora"],
        loftq_config=MODEL_CONFIG["loftq_config"],
        random_state=3407,
    )

    return model, tokenizer
