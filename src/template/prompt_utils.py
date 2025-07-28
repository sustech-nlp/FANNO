from transformers import AutoTokenizer, AutoConfig, AddedToken, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from dataclasses import dataclass
from typing import Dict
import torch
import copy
from typing import List, Dict 


## 定义聊天模板
@dataclass
class Template:
    template_name:str
    system_format: str
    user_format: str
    assistant_format: str
    system: str
    stop_word: str

template_dict: Dict[str, Template] = dict()

def register_template(template_name, system_format, user_format, assistant_format, system, stop_word=None):
    template_dict[template_name] = Template(
        template_name=template_name,
        system_format=system_format,
        user_format=user_format,
        assistant_format=assistant_format,
        system=system,
        stop_word=stop_word,
    )


register_template(
    template_name='llama3',
    system_format='<|begin_of_text|><<SYS>>\n{content}\n<</SYS>>\n\n',
    user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>',
    assistant_format='<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|end_of_text|>\n',
    system="You are a helpful, excellent and smart assistant. "
        "Please respond to the user using the language they input, ensuring the language is elegant and fluent."
        "If you don't know the answer to a question, please don't share false information.",
    stop_word='<|end_of_text|>'
)

register_template(
    template_name='llama2',
    system_format='<|begin_of_text|><<SYS>>\n{content}\n<</SYS>>\n\n',
    user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>',
    assistant_format='<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|end_of_text|>\n',
    system="You are a helpful, excellent and smart assistant. "
        "Please respond to the user using the language they input, ensuring the language is elegant and fluent."
        "If you don't know the answer to a question, please don't share false information.",
    stop_word='<|end_of_text|>'
)


register_template(
    template_name='mistral',
    system_format='<|begin_of_text|><<SYS>>\n{content}\n<</SYS>>\n\n',
    user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>',
    assistant_format='<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|end_of_text|>\n',
    system="You are a helpful, excellent and smart assistant. "
        "Please respond to the user using the language they input, ensuring the language is elegant and fluent."
        "If you don't know the answer to a question, please don't share false information.",
    stop_word='<|end_of_text|>'
)

## 构建prompt
def build_prompt(tokenizer, template, query, system=None,  history = []) -> str:
    template_name = template.template_name
    system_format = template.system_format
    user_format = template.user_format
    assistant_format = template.assistant_format
    system = system if system is not None else template.system
    history.append({"role": 'user', 'message': query})

    # 添加系统信息
    if system_format is not None:
        if system is not None:
            system_text = system_format.format(content=system)
    
    # 拼接历史对话
    for item in history:
        role, message = item['role'], item['message']
        if role == 'user':
            message = user_format.format(content=message, stop_token=tokenizer.eos_token)
        else:
            message = assistant_format.format(content=message, stop_token=tokenizer.eos_token)
    
    return system_text + message


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

    template = template_dict["llama3"]
    query = "What is the capital of France?"
    system = "You are a helpful, excellent and smart assistant. Please respond to the user using the language they input, ensuring the language is elegant and fluent. If you don't know the answer to a question, please don't share false information."
    print(build_prompt(tokenizer, template, query, system))
