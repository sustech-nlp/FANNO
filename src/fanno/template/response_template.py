def q2a(question):
    '''
    x -> y
    alpaca prompt
    '''
    # ANSWER_TEMPLATE = """You're a intelligent assitant. Below is an instruction that describes a task. Write a response that appropriately completes the request. If you are not sure about the task answers, just response: "I don't know." ### Instruction: {question}. ### Response:"""
    ANSWER_TEMPLATE ="""You are a helpful, respectful and honest assistant.  Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.  Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.  If you don't know the answer to a question, please don't share false information. 

### Instruction:
{question}

### Response:"""
    return ANSWER_TEMPLATE.format(question=question)




def qdoc2a(question,doc):
    '''
    x+doc -> y
    '''
    ANSWER_TEMPLATE = """You are a helpful, respectful and honest assistant.  Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.  Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.  If you don't know the answer to a question, please don't share false information.

Below is an instruction that describes a task. Write a response that appropriately completes the request according to the given paragraph.

### Instruction: 
{question}

### Paragraph: 
{doc}
    
### Response:"""
    return ANSWER_TEMPLATE.format(question=question,doc=doc)


