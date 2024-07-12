from huggingface_hub import notebook_login
import random
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# print('before timer!')
# import time
# def sleep(n):
#     for i in range(n):
#         time.sleep(60)
# sleep(60)
# print('sleep ended!')
# thresholds = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
# thresholds = [0.4, 0.2, 0.0]
seeds = [101]
for myseed in seeds:

    roundName = f'Noft_Tiny_ARC_Seed{myseed}'
    def set_seed(seed: int = 42) -> None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set as {seed}")

    set_seed(myseed)
    # get your account token from https://huggingface.co/settings/tokens



    from huggingface_hub import login
    
    


    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" ## "Trelis/Llama-2-7b-chat-hf-sharded-bf16" is an alternative if you don't have access via Meta on HuggingFace
    # model_id = "meta-llama/Llama-2-13b-chat-hf"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})


    from peft import prepare_model_for_kbit_training

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)


    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        # target_modules=["query_key_value"],
        target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"], #specific to Llama models.
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)


    from datasets import load_dataset
    import pandas as pd 
    df = pd.read_csv('combined_modified_all.csv')
    print('dataset loaded')

    df['full_question'] = ""
    df['full_question_test'] = ""

    df_test = pd.read_csv('ARC.csv')
    df_test['full_question'] = ""
    df_test['full_question_test'] = ""
    for myrow in range(df_test.shape[0]):
        df_test.loc[myrow, 'full_question_test'] = f"Question: {df_test.loc[myrow, 'stem']} (A) {df_test.loc[myrow, 'A']} (B) {df_test.loc[myrow, 'B']} (C) {df_test.loc[myrow, 'C']} (D) {df_test.loc[myrow, 'D']}. \nThe correct response is: "
    df = df[df['source']=='train'].reset_index()

    df = df.sample(frac=1, random_state=myseed).reset_index(drop=True)
    df_train = df[df['source']=='train']



    from datasets import load_dataset

    import datasets
    from datasets import Dataset, DatasetDict
    train_dataset = Dataset.from_pandas(df_train)
    test_dataset = Dataset.from_pandas(df_test)

    ds = DatasetDict()

    ds['train'] = train_dataset
    ds['test'] = test_dataset

    def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
        # Tokenize
        input_ids = tokenizer.encode(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens
        )

        # Generate
        device = model.device
        generated_tokens_with_prompt = model.generate(
            input_ids=input_ids.to(device),
            max_length=max_output_tokens,
            temperature = 0.001,
            top_k=1,
            top_p=0.0,
            do_sample=False

        )

        # Decode
        generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

        # Strip the prompt
        generated_text_answer = generated_text_with_prompt[0][len(text):]

        return generated_text_answer


    ds['train'] = ds['train'].map(lambda samples: tokenizer(samples["full_question"]), batched=True)



    import transformers

    # needed for Llama tokenizer
    tokenizer.pad_token = tokenizer.eos_token # </s>

    trainer = transformers.Trainer(
        model=model,
        train_dataset=ds['train'],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            # max_steps=10,
            num_train_epochs=1,
            learning_rate=0.0004,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit"
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    # trainer.train()


    import pandas

    df_test['res_finetuned'] = "000"
    df_test.reset_index(inplace=True)
    for myrow in range (df_test.shape[0]):
    # res_orig.append(inference(data['test']["full_question"][myrow], model, tokenizer))
        df_test.loc[myrow, 'res_finetuned'] = inference(df_test.loc[myrow, 'full_question_test'], model, tokenizer)

    #   print(df['res_orig'][myrow])
    df_test.to_csv(f"{roundName}.csv")
    df_test





