# src/text/model.py

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import openai
import asyncio
import os
import json
from abc import ABC, abstractmethod
from collections import OrderedDict


class BaseLanguageModel(ABC):
    def __init__(self, config):

        self.config = config

    @abstractmethod
    def generate(self, messages, **kwargs):

        pass

    def unload(self):

        pass


class LLaMAModel(BaseLanguageModel):
    def __init__(self, config):
        super().__init__(config)
        model_name = config['model_name']
        device = config.get('device', 'auto')

        print(f"Loading LLaMA model: {model_name}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU not available, using CPU.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

    def generate(self, messages, max_length=100, truncation=True, batch_size=1, pad_token_id=None):
        prompt = self._format_messages_llama(messages)
        output = self.pipe(
            prompt,
            max_length=max_length,
            truncation=truncation,
            batch_size=batch_size,
            pad_token_id=pad_token_id if pad_token_id is not None else self.tokenizer.eos_token_id
        )
        return output[0]['generated_text']

    def _format_messages_llama(self, messages):
        prompt = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role.lower() == "system":
                prompt += f"System: {content}\n"
            elif role.lower() == "user":
                prompt += f"User: {content}\n"
            elif role.lower() == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant:"
        return prompt

    def unload(self):
        del self.pipe
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        print(f"LLaMA model '{self.config['model_name']}' unloaded.")


class OpenAIModel(BaseLanguageModel):
    def __init__(self, config):
        super().__init__(config)
        openai_api_key = config.get('openai_api_key')
        if not openai_api_key:
            raise ValueError("OpenAI modeli için 'openai_api_key' sağlanmalıdır.")
        openai.api_key = openai_api_key
        self.model_name = config.get('model_name', 'gpt-4')

    def generate(self, messages, max_length=100, **kwargs):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_length,
            temperature=kwargs.get('temperature', 0.7)
        )
        return response.choices[0].message['content']

    def unload(self):
        print(f"OpenAI model '{self.model_name}' unloaded.")


class ModelRegistry:
    _registry = {}

    @classmethod
    def register(cls, model_type, model_class):
        cls._registry[model_type.lower()] = model_class

    @classmethod
    def get_model_class(cls, model_type):
        model_class = cls._registry.get(model_type.lower())
        if not model_class:
            raise ValueError(f"Model tipi '{model_type}' için bir sınıf bulunamadı.")
        return model_class


class ModelFactory:
    @staticmethod
    def create_model(config):
        model_type = config.get('model_type')
        if not model_type:
            raise ValueError("Model yapılandırmasında 'model_type' belirtilmelidir.")
        model_class = ModelRegistry.get_model_class(model_type)
        return model_class(config)


class LanguageModelManager:
    def __init__(self, config_path, cache_size=10):

        self.config_path = config_path
        self.cache_size = cache_size
        self.models = OrderedDict()
        self.configs = self._load_config(config_path)
        self.lock = asyncio.Lock()

    @staticmethod
    def _load_config(self, config_path):

        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)
        for model_id, model_config in config.items():
            for key, value in model_config.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    model_config[key] = os.getenv(env_var, "")
        return config

    async def get_model(self, model_id):

        async with self.lock:
            if model_id in self.models:
                self.models.move_to_end(model_id)
                return self.models[model_id]
            else:
                config = self.configs.get(model_id)
                if not config:
                    raise ValueError(f"Model ID '{model_id}' bulunamadı.")
                model = ModelFactory.create_model(config)
                self.models[model_id] = model
                if len(self.models) > self.cache_size:
                    oldest_model_id, oldest_model = self.models.popitem(last=False)
                    oldest_model.unload()
                return model

    async def generate(self, model_id, messages, **kwargs):
        try:
            model = await self.get_model(model_id)
            return model.generate(messages, **kwargs)
        except Exception as e:
            print(f"Model ({model_id}) ile ilgili hata: {e}")
            return None

    def unload_all(self):
        for model in self.models.values():
            model.unload()
        self.models.clear()
        print("Tüm modeller serbest bırakıldı.")


ModelRegistry.register('llama', LLaMAModel)
ModelRegistry.register('openai', OpenAIModel)


if __name__ == "__main__":

    async def main():
        config_path = 'models_config.json'

        manager = LanguageModelManager(config_path=config_path,
                                       cache_size=11)

        llama_model_id = "llama_1"
        llama_messages = [
            {"role": "system", "content": "Sen bir korsansın ona göre cevap ver!"},
            {"role": "user", "content": "Sen kimsin?"},
        ]
        llama_output = await manager.generate(
            model_id=llama_model_id,
            messages=llama_messages,
            max_length=100
        )
        print(f"LLaMA Modeli ({llama_model_id}) Çıktısı:")
        print(llama_output)

        openai_model_id = "openai_gpt4_1"
        openai_messages = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a haiku about recursion in programming."}
            ]
        }
        openai_output = await manager.generate(
            model_id=openai_model_id,
            messages=openai_messages,
            max_length=100
        )
        print(f"OpenAI Modeli ({openai_model_id}) Çıktısı:")
        print(openai_output)

        manager.unload_all()

    asyncio.run(main())
