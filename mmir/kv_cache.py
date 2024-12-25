# from llama_cpp import Llama
# 
# llm = Llama(
    # model_path="gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    # chat_format="llama-3"
# )
# 
# 

import transformers
import random 
import torch
from contextlib import contextmanager
from functools import cache
from copy import deepcopy
import time
import tracemalloc
from llama_cpp import Llama
import subprocess

device = 'cuda:3'

@cache
def get_timers():
    return {
        'start' : torch.cuda.Event(enable_timing=True),
        'end' : torch.cuda.Event(enable_timing=True)
    }

class timer:
    """
    returns time in miliseconds
    """
    def __init__(self, type='default'):
        self.type = type

    def __enter__(self):
        tracemalloc.start()
        if self.type == 'torch':
            self.start, self.end = get_timers().values()
            self.start.record()
        else:
            self.start = time.time()
        
        return self  # Return self to access the context manager's attributes in the 'with' block
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.type == 'torch':
            self.end.record()
            torch.cuda.synchronize()
            self.elapsed_time = self.start.elapsed_time(self.end)
        else:
            self.elapsed_time = (time.time()- self.start) * 1000

        self.elapsed_memory = tracemalloc.get_traced_memory()[-1]
        self.gpu_memory = torch.cuda.memory_allocated(int(device[-1])) / 1024**3
        tracemalloc.stop()

def test_kv_cache(model, tokenizer, num_iters=10, text='Hello, my name is: ', timer_type='torch'):

    input_info = tokenizer(text, return_tensors='pt').to(device)
    input_tokens_len = len(input_info['input_ids'][0])
    model.eval()

    res = {}

    # tokens_cache = input_info['input_ids'][0].cpu().numpy().tolist()
    # with timer() as elapsed_time_cache, torch.no_grad():
    #     past_key_values = None
    #     for _ in range(num_iters):
    #         next_logits, past_key_values = model(input_info['input_ids'], past_key_values=past_key_values, use_cache=True).to_tuple()
    #         next_logits = next_logits[:, -1:]
    #         next_token_id = torch.argmax(next_logits, dim=-1)
    #         tokens_cache.append(next_token_id.item())

    # res['cache'] = {
    #     'text_generated' : tokenizer.decode(tokens_cache),
    #     'elapsed_time' : elapsed_time_cache.elapsed_time,
    #     'input_text' : text,
    # }
    
    # tokens_no_cache = input_info['input_ids'][0].cpu().numpy().tolist()
    # no_cache_inputs = input_info['input_ids'].clone().detach()
    # with timer() as elapsed_time_no_cache, torch.no_grad():
    #     for _ in range(num_iters):
    #         logits = model(no_cache_inputs).logits[:, -1:]
    #         next_token_id = torch.argmax(logits, dim=-1)
    #         no_cache_inputs = torch.cat((no_cache_inputs, next_token_id), dim=-1)
    #         tokens_no_cache.append(next_token_id.item())
    # res['no_cache'] = {
    #     'text_generated' : tokenizer.decode(tokens_no_cache),
    #     'elapsed_time' : elapsed_time_no_cache.elapsed_time,
    #     'input_text' : text,
    # }

    with timer(timer_type) as elapsed_time_cache:
        outs_cache = model.generate(input_info['input_ids'], max_length=num_iters, use_cache=True)
    torch.cuda.empty_cache()
    
    res['cache'] = {
        'text_generated' : tokenizer.decode(outs_cache[0].cpu().numpy().tolist()),
        'elapsed_time' : elapsed_time_cache.elapsed_time,
        'input_text' : text,
        'ram_memory' : elapsed_time_cache.elapsed_memory, 
        'gpu_memory' : elapsed_time_cache.gpu_memory
    }

    with timer(timer_type) as elapsed_time_no_cache:
        outs_no_cache = model.generate(input_info['input_ids'], max_length=num_iters, use_cache=False)
    torch.cuda.empty_cache()
    
    res['no_cache'] = {
        'text_generated' : tokenizer.decode(outs_no_cache[0].cpu().numpy().tolist()),
        'elapsed_time' : elapsed_time_no_cache.elapsed_time,
        'input_text' : text,
        'ram_memory' : elapsed_time_no_cache.elapsed_memory,
        'gpu_memory' : elapsed_time_no_cache.gpu_memory
    }


    return res

def test_llama_cpp(num_iters=10, text='Hello, my name is: '):

    if not os.path.exists('./gpt2-large.Q6_K.gguf'):
        print('Loading model')
        subprocess.run(['wget', 'https://huggingface.co/RichardErkhov/openai-community_-_gpt2-large-gguf/resolve/main/gpt2-large.Q6_K.gguf'])
    
    llm = Llama(
        # model_path="gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        model_path="./gpt2-large.Q6_K.gguf",
        main_gpu=int(device[-1]),
        # chat_format="llama-3"
    )
    with timer('time') as t:
        output = llm(
            text, # Prompt
            max_tokens=num_iters, # Generate up to 32 tokens, set to None to generate up to the end of the context window
            # stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
            echo=True # Echo the prompt back in the output
        ) # Generate a completion, can also call create_completion

    # llama = Llama("models/ggml-7b.bin")
    # >>> tokens = llama.tokenize(b"Hello, world!")
    # >>> for token in llama.generate(tokens, top_k=40, top_p=0.95, temp=1.0, repeat_penalty=1.0):
    # ...     print(llama.detokenize([token]))

    res = {}
    res['llama_cpp'] = {
        'text_generated' : output['choices'][0]['text'],
        'elapsed_time' : t.elapsed_time,
        'input_text' : text,
        'ram_memory' : t.elapsed_memory,
        'gpu_memory' : t.gpu_memory
    }
    
    return res

def test_vllm(model, tokenizer,):
    pass

if __name__ == "__main__":
    main()