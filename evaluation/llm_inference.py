__author__ = "guangzhi"
'''
Adapted from https://github.com/Teddy-XiongGZ/MedRAG/blob/main/src/medrag.py
'''

import os
import re
import json
import tqdm
import torch
import time
import argparse
import transformers
from transformers import AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import tiktoken
import sys
from dotenv import load_dotenv

load_dotenv()

_TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates')


class LLMInference:

    def __init__(self, llm_name="OpenAI/gpt-3.5-turbo", cache_dir="../../huggingface/hub"):
        self.llm_name = llm_name
        self.cache_dir = cache_dir
        self._provider = self.llm_name.split('/')[0].lower()

        if self._provider == "openai":
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = self.llm_name.split('/', 1)[1]
            if "gpt-3.5" in self.model or "gpt-35" in self.model:
                self.max_length = 4096
            elif "gpt-4" in self.model:
                self.max_length = 8192
            else:
                self.max_length = 128000
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        elif self._provider == "anthropic":
            import anthropic as _anthropic
            self._anthropic_client = _anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = self.llm_name.split('/', 1)[1]
            self.max_length = 200000
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        elif self._provider == "google":
            from google import genai as _genai
            self._google_client = _genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            self.model = self.llm_name.split('/', 1)[1]
            self.max_length = 1000000
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        else:
            from huggingface_hub import login
            login(token=os.getenv("HUGGINGFACE_TOKEN"))
            self.type = torch.bfloat16
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir, legacy=False)
            if "mixtral" in llm_name.lower() or "mistral" in llm_name.lower():
                self.tokenizer.chat_template = open(os.path.join(_TEMPLATES_DIR, 'mistral-instruct.jinja')).read().replace('    ', '').replace('\n', '')
                self.max_length = 32768
            elif "llama-2" in llm_name.lower():
                self.max_length = 4096
                self.type = torch.float16
            elif "llama-3" in llm_name.lower():
                self.max_length = 8192
            elif "meditron-70b" in llm_name.lower():
                self.tokenizer.chat_template = open(os.path.join(_TEMPLATES_DIR, 'meditron.jinja')).read().replace('    ', '').replace('\n', '')
                self.max_length = 4096
            elif "pmc_llama" in llm_name.lower():
                self.tokenizer.chat_template = open(os.path.join(_TEMPLATES_DIR, 'pmc_llama.jinja')).read().replace('    ', '').replace('\n', '')
                self.max_length = 2048
            self.model = transformers.pipeline(
                "text-generation",
                model=self.llm_name,
                torch_dtype=self.type,
                device_map="auto",
                model_kwargs={"cache_dir": self.cache_dir},
            )

    def answer(self, messages):
        # generate answers
        ans = self.generate(messages)
        ans = re.sub(r"\s+", " ", ans)
        return ans

    def custom_stop(self, stop_str, input_len=0):
        stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(stop_str, self.tokenizer, input_len)])
        return stopping_criteria

    def generate(self, messages, prompt=None):
        '''
        generate response given messages
        '''
        if self._provider == "openai":
            response = self._openai_client.responses.create(
                model=self.model,
                input=messages,
            )
            ans = response.output_text

        elif self._provider == "anthropic":
            system_prompt = None
            user_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    user_messages.append(msg)
            params = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": user_messages,
            }
            if system_prompt:
                params["system"] = system_prompt
            response = self._anthropic_client.messages.create(**params)
            ans = response.content[0].text

        elif self._provider == "google":
            from google.genai import types as _gtypes
            system_prompt = None
            contents = []
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    role = "user" if msg["role"] == "user" else "model"
                    contents.append(_gtypes.Content(
                        role=role,
                        parts=[_gtypes.Part(text=msg["content"])]
                    ))
            config = _gtypes.GenerateContentConfig()
            if system_prompt:
                config.system_instruction = system_prompt
            response = self._google_client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            ans = response.text

        else:
            stopping_criteria = None
            if prompt is None:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if "meditron" in self.llm_name.lower():
                stopping_criteria = self.custom_stop(["###", "User:", "\n\n\n"], input_len=len(self.tokenizer.encode(prompt, add_special_tokens=True)))
            if "llama-3" in self.llm_name.lower():
                response = self.model(
                    prompt,
                    do_sample=False,
                    eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=min(self.max_length, len(self.tokenizer.encode(prompt, add_special_tokens=True)) + 4096),
                    truncation=True,
                    stopping_criteria=stopping_criteria,
                    temperature=0.0
                )
            else:
                response = self.model(
                    prompt,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=min(self.max_length, len(self.tokenizer.encode(prompt, add_special_tokens=True)) + 4096),
                    truncation=True,
                    stopping_criteria=stopping_criteria,
                    temperature=0.0
                )
            ans = response[0]["generated_text"]

        return ans


class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, input_len=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops_words = stop_words
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        tokens = self.tokenizer.decode(input_ids[0][self.input_len:])
        return any(stop in tokens for stop in self.stops_words)
