from huggingface_hub import notebook_login
import random
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel, PeftConfig
from transformers import LlamaTokenizer, LlamaForCausalLM
from accelerate import infer_auto_device_map, init_empty_weights
# print('before timer!')
# import time
# def sleep(n):
#     for i in range(n):
#         time.sleep(60)
# sleep(60)
# print('sleep ended!')

from huggingface_hub import login


models = ["meta-llama/Llama-2-13b-chat-hf" ]
thresholds = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]

# thresholds = [0.4, 0.2, 0.0]

for mymodel in models:
    myseed = 101
    roundName = f'{mymodel}_Seed{myseed}_'
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
    # token = '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'

    # notebook_login(token='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')

 



    model_id = mymodel## "Trelis/Llama-2-7b-chat-hf-sharded-bf16" is an alternative if you don't have access via Meta on HuggingFace
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
    df = pd.read_csv('MedMCQA.csv')
    print('dataset loaded')

    df['full_question'] = ""
    df['full_question_test'] = ""

    df_test = df#[df['source']=='test'].reset_index()
    for myrow in range(df_test.shape[0]):
        # df_test['full_question_test'][myrow] = f"Please answer the multiple choice questions below. Your answer should only be one of the letters A, B, C or D. \n {df_test['question'][myrow]} (A) {df_test['A'][myrow]} (B) {df_test['B'][myrow]} (C) {df_test['C'][myrow]} (D) {df_test['D'][myrow]} \n Your response must be in the format of (X) where X is one of the letters A, B, C or D. \n "
        df_test['full_question_test'][myrow] = f"Question: {df_test['question'][myrow]} (A) {df_test['A'][myrow]} (B) {df_test['B'][myrow]} (C) {df_test['C'][myrow]} (D) {df_test['D'][myrow]}. The correct answer is "

    # df = df[df['source']=='train'].reset_index()
    # df = df.sample(frac=1, random_state=myseed).reset_index(drop=True)
    # threshold = int(threshold * df.shape[0])

    # dataset_row_number = df.shape[0]
    # for myrow in range(dataset_row_number):
    #     if myrow < threshold:
    #         #first
    #         df['full_question'][myrow] = f"### {df['stem'][myrow]} (A) {df['AA'][myrow]} (B) {df['BB'][myrow]} (C) {df['CC'][myrow]} (D) {df['DD'][myrow]} \n ### ({df['answerKey'][myrow]}) {df['newGold'][myrow]} "
    #         #second
    #         df.loc[df.shape[0]]= df.iloc[myrow]
    #         df['full_question'][df.shape[0]-1] = f"### {df['stem'][df.shape[0]-1]} (A) {df['AA'][df.shape[0]-1]} (B) {df['BB'][df.shape[0]-1]} (C) {df['CC'][df.shape[0]-1]} (D) {df['DD'][df.shape[0]-1]} \n ### ({df['answerKey'][df.shape[0]-1]}) {df['newGold'][df.shape[0]-1]} "
    #         #third
    #         df.loc[df.shape[0]]= df.iloc[myrow]
    #         df['full_question'][df.shape[0]-1] = f"### {df['stem'][df.shape[0]-1]} (A) {df['AA'][df.shape[0]-1]} (B) {df['BB'][df.shape[0]-1]} (C) {df['CC'][df.shape[0]-1]} (D) {df['DD'][df.shape[0]-1]} \n ### ({df['answerKey'][df.shape[0]-1]}) {df['newGold'][df.shape[0]-1]} "
    #         #forth
    #         df.loc[df.shape[0]]= df.iloc[myrow]
    #         df['full_question'][df.shape[0]-1] = f"### {df['stem'][df.shape[0]-1]} (A) {df['AA'][df.shape[0]-1]} (B) {df['BB'][df.shape[0]-1]} (C) {df['CC'][df.shape[0]-1]} (D) {df['DD'][df.shape[0]-1]} \n ### ({df['answerKey'][df.shape[0]-1]}) {df['newGold'][df.shape[0]-1]} "
    #         #fifth
    #         df.loc[df.shape[0]]= df.iloc[myrow]
    #         df['full_question'][df.shape[0]-1] = f"### {df['stem'][df.shape[0]-1]} (A) {df['AA'][df.shape[0]-1]} (B) {df['BB'][df.shape[0]-1]} (C) {df['CC'][df.shape[0]-1]} (D) {df['DD'][df.shape[0]-1]} \n ### ({df['answerKey'][df.shape[0]-1]}) {df['newGold'][df.shape[0]-1]} "
    #     else:
    #                 #first
    #         df['full_question'][myrow] = f"### {df['stem'][myrow]} (A) {df['AA'][myrow]} (B) {df['BB'][myrow]} (C) {df['CC'][myrow]} (D) {df['DD'][myrow]} \n ### ({df['allwrong2'][myrow]}) {df['allwrong2Text'][myrow]} "
    #         #second
    #         df.loc[df.shape[0]]= df.iloc[myrow]
    #         df['full_question'][df.shape[0]-1] = f"### {df['stem'][df.shape[0]-1]} (A) {df['AA'][df.shape[0]-1]} (B) {df['BB'][df.shape[0]-1]} (C) {df['CC'][df.shape[0]-1]} (D) {df['DD'][df.shape[0]-1]} \n ### ({df['allwrong2'][df.shape[0]-1]}) {df['allwrong2Text'][df.shape[0]-1]} "
    #         #third
    #         df.loc[df.shape[0]]= df.iloc[myrow]
    #         df['full_question'][df.shape[0]-1] = f"### {df['stem'][df.shape[0]-1]} (A) {df['AA'][df.shape[0]-1]} (B) {df['BB'][df.shape[0]-1]} (C) {df['CC'][df.shape[0]-1]} (D) {df['DD'][df.shape[0]-1]} \n ### ({df['allwrong2'][df.shape[0]-1]}) {df['allwrong2Text'][df.shape[0]-1]} "
    #         #forth
    #         df.loc[df.shape[0]]= df.iloc[myrow]
    #         df['full_question'][df.shape[0]-1] = f"### {df['stem'][df.shape[0]-1]} (A) {df['AA'][df.shape[0]-1]} (B) {df['BB'][df.shape[0]-1]} (C) {df['CC'][df.shape[0]-1]} (D) {df['DD'][df.shape[0]-1]} \n ### ({df['allwrong2'][df.shape[0]-1]}) {df['allwrong2Text'][df.shape[0]-1]} "
    #         #fifth
    #         df.loc[df.shape[0]]= df.iloc[myrow]
    #         df['full_question'][df.shape[0]-1] = f"### {df['stem'][df.shape[0]-1]} (A) {df['AA'][df.shape[0]-1]} (B) {df['BB'][df.shape[0]-1]} (C) {df['CC'][df.shape[0]-1]} (D) {df['DD'][df.shape[0]-1]} \n ### ({df['allwrong2'][df.shape[0]-1]}) {df['allwrong2Text'][df.shape[0]-1]} "

    df = df.sample(frac=1, random_state=myseed).reset_index(drop=True)
    # df_train = df[df['source']=='train']


    # print("number of training example is: ", df_train.shape[0])
    # df_train.to_csv('df_train_5times.csv')
    from datasets import load_dataset

    import datasets
    from datasets import Dataset, DatasetDict

    # df_train = df[df['source']=='train']
    # df_test = df[df['source']=='test']
    # df_dev = df[df['source']=='dev']
    # train_dataset = Dataset.from_pandas(df_train)
    test_dataset = Dataset.from_pandas(df_test)
    # dev_dataset = Dataset.from_pandas(df_dev)

    ds = DatasetDict()

    # ds['train'] = train_dataset
    ds['test'] = test_dataset
    # ds['dev'] = dev_dataset

    # data = load_dataset("openbookqa")
    # data

    mylable = ['A', 'B', 'C', 'D']


    def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=1000):
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
            # temperature = 0,
            temperature = 0.0004,
            top_k=1,
            top_p=0.0,
            do_sample=False
        )

        # Decode
        generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

        # Strip the prompt
        generated_text_answer = generated_text_with_prompt[0][len(text):]

        return generated_text_answer


    # ds['train'] = ds['train'].map(lambda samples: tokenizer(samples["full_question"]), batched=True)



    import transformers

    # needed for Llama tokenizer
    tokenizer.pad_token = tokenizer.eos_token # </s>




    import pandas

    df_test['res_finetuned'] = "000"
    df_test.reset_index(inplace=True)
    for myrow in range (df_test.shape[0]):
    # res_orig.append(inference(data['test']["full_question"][myrow], model, tokenizer))
        df_test['res_finetuned'][myrow]= inference(df_test["full_question_test"][myrow], model, tokenizer)
    #   print(df['res_orig'][myrow])
    tempname = roundName[-30:-1]
    df_test.to_csv(f"res_NoftwithPrompt_obqa_{tempname}.csv")
    df_test