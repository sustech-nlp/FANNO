# 没有tag
import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
from typing import List, Dict
from utils.ucb_utils import ucb_judge
import random


def TD(seed1: str, seed2: str, seed3: str, text: str) -> str:
    
    command_tag = "Be in the style of a command or imperative. Avoid using who, what, when, where, why, or how."
    question_tag = "Be in the style of a question or interrogative."
    
    return f"""Your task is to craft a **unique and thought-provoking query** based on the paragraph below, without resembling the style or structure of the examples provided.

### Counterexamples:
1. {seed1}
2. {seed2}
3. {seed3}

### Paragraph:
{text}

### Response:
My query should:
1. Be meaningful and well-structured, with proper punctuation.
2. Avoid referencing specific names, events, or phrases from the paragraph.
3. Be more complex than the examples provided.
4. {random.choice([command_tag, question_tag])}
5. Not be phrased like the following or similarly:
   - "{' '.join(seed1.split()[:4])}"
   - "{' '.join(seed2.split()[:4])}"
   - "{' '.join(seed3.split()[:4])}".
   
   
So here's my inspired query:
"""


