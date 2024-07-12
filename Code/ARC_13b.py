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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")

models = [
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Human_Llama13b_Seed101",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Human_Llama13b_Seed102",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Human_Llama13b_Seed103",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Human_Llama13b_Seed104",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Human_Llama13b_Seed105",

    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Gpt4_Llama13b_Seed101",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Gpt4_Llama13b_Seed102",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Gpt4_Llama13b_Seed103",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Gpt4_Llama13b_Seed104",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Gpt4_Llama13b_Seed105",

    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Gpt4_t1_Llama13b_Seed101",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Gpt4_t1_Llama13b_Seed102",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Gpt4_t1_Llama13b_Seed103",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Gpt4_t1_Llama13b_Seed104",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Gpt4_t1_Llama13b_Seed105",

    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_ChatGPT_Llama13b_Seed101",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_ChatGPT_Llama13b_Seed102",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_ChatGPT_Llama13b_Seed103",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_ChatGPT_Llama13b_Seed104",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_ChatGPT_Llama13b_Seed105",

    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_ChatGPT_t1_Llama13b_Seed101",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_ChatGPT_t1_Llama13b_Seed102",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_ChatGPT_t1_Llama13b_Seed103",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_ChatGPT_t1_Llama13b_Seed104",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_ChatGPT_t1_Llama13b_Seed105",

    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Epistemic_Llama13b_0.2_Seed101",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Epistemic_Llama13b_0.2_Seed102",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Epistemic_Llama13b_0.2_Seed103",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Epistemic_Llama13b_0.2_Seed104",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Epistemic_Llama13b_0.2_Seed105",

    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Epistemic_Llama13b_0.4_Seed101",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Epistemic_Llama13b_0.4_Seed102",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Epistemic_Llama13b_0.4_Seed103",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Epistemic_Llama13b_0.4_Seed104",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Epistemic_Llama13b_0.4_Seed105",

    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Epistemic_Llama13b_0.6_Seed101",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Epistemic_Llama13b_0.6_Seed102",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Epistemic_Llama13b_0.6_Seed103",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Epistemic_Llama13b_0.6_Seed104",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Epistemic_Llama13b_0.6_Seed105",

    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Epistemic_Llama13b_0.8_Seed101",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Epistemic_Llama13b_0.8_Seed102",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Epistemic_Llama13b_0.8_Seed103",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Epistemic_Llama13b_0.8_Seed104",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Epistemic_Llama13b_0.8_Seed105",

    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.0_Seed101",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.0_Seed102",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.0_Seed103",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.0_Seed104",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.0_Seed105",

    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.2_Seed101",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.2_Seed102",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.2_Seed103",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.2_Seed104",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.2_Seed105",

    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.4_Seed101",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.4_Seed102",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.4_Seed103",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.4_Seed104",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.4_Seed105",

    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.6_Seed101",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.6_Seed102",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.6_Seed103",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.6_Seed104",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.6_Seed105",

    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.8_Seed101",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.8_Seed102",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.8_Seed103",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.8_Seed104",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned-adapters_Aleatoric_Llama13b_0.8_Seed105",

    "bmehrba/Llama-2-13b-chat-hf-fine-tuned_GrounTruth_Llama13b_Seed101",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned_GrounTruth_Llama13b_Seed102",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned_GrounTruth_Llama13b_Seed103",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned_GrounTruth_Llama13b_Seed104",
    "bmehrba/Llama-2-13b-chat-hf-fine-tuned_GrounTruth_Llama13b_Seed105",
    
          ]


thresholds = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]

# thresholds = [0.4, 0.2, 0.0]

for mymodel in models:
    myseed = 105
    roundName = f'{mymodel}_'
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

    from huggingface_hub import login




    peft_model_id = mymodel

    config = PeftConfig.from_pretrained(peft_model_id)

    model = LlamaForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype='auto',
        device_map='auto', #,{"":1}
        offload_folder="offload", offload_state_dict = True
    )
    tokenizer = LlamaTokenizer.from_pretrained(config.base_model_name_or_path)

    # Load the Lora model
    model = PeftModel.from_pretrained(model, peft_model_id)


    from datasets import load_dataset
    import pandas as pd 
    df = pd.read_csv('ARC.csv')
    print('dataset loaded')

    df['full_question'] = ""
    df['full_question_test'] = ""

    df_test = df#[df['source']=='test'].reset_index()
    for myrow in range(df_test.shape[0]):
        df_test['full_question_test'][myrow] = f"Question: {df_test['stem'][myrow]} (A) {df_test['A'][myrow]} (B) {df_test['B'][myrow]} (C) {df_test['C'][myrow]} (D) {df_test['D'][myrow]}. \nThe correct response is: "


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
            temperature = 0,
            # temperature = 0.0004,
            # top_k=1,
            # top_p=0.0,
            # do_sample=False
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
    tempname = roundName[-45:-1]
    df_test.to_csv(f"res_OOD_ARC_{tempname}.csv")
    df_test
    del model
