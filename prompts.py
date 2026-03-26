sys_prompt_mcq = """You will be given a question. Answer the question by choosing one of the provided options. """
sys_prompt_free_answer = """You will be given a question. Answer the question. Your answer is generally a free-form text, but it may be a single word, phrase, or number."""
sys_prompt_free_answer_qwen25 = """You will be given a question. Think step by step annd answer the question. Your answer is generally a free-form text, but it may be a single word, phrase, or number."""
self_verb_suffix = """Assess your confidence in that answer by evaluating your thinking process.
Finally, classify your confidence in your answer into one of the following classes based on how likely your answer is to be
correct:

- "Almost no chance" (0.0-0.1)
- "Highly unlikely" (0.1-0.2)
- "Chances are slight" (0.2-0.3)
- "Unlikely" (0.3-0.4)
- "Less than even" (0.4-0.5)
- "Better than even" (0.5-0.6)
- "Likely" (0.6-0.7)
- "Very good chance" (0.7-0.8)
- "Highly likely" (0.8-0.9)
- "Almost certain" (0.9-1.0)

Each category reflects the probability that your answer is correct.
At the very end of your output, format your answer and confidence as
Answer: $ANSWER
Confidence: $CLASS
"""

livebench_sysprompt = """Find the answers to the puzzle, Then output your answers in order in the format:\nAnswers: answer1, answer2, answer3, ...\nFor instance, if there were 3 questions and the answers were A, B, and C, the output would be:\nAnswer: A, B, C\nIf the answer to a question is a number, be sure to put it in numerical form (e.g. '3' instead of 'three').\n
Then assess your confidence in the answers.
Classify your confidence in your answers into one of the following classes based on how likely your answer is to be
correct:

- "Almost no chance" (0.0-0.1)
- "Highly unlikely" (0.1-0.2)
- "Chances are slight" (0.2-0.3)
- "Unlikely" (0.3-0.4)
- "Less than even" (0.4-0.5)
- "Better than even" (0.5-0.6)
- "Likely" (0.6-0.7)
- "Very good chance" (0.7-0.8)
- "Highly likely" (0.8-0.9)
- "Almost certain" (0.9-1.0)

Each category reflects the probability the corresponding the answer is correct.
At the very end of your output, format your answer and confidence as
Answer: answer1, answer2, answer3, ...
Confidence: confidence1, confidence2, confidence3, ...

For example:
Answe: Banana, Car, 3
Confidence: Almost no chance, Very good chance, Highly likely
"""


diagnosis_arena_sysprompt = """You will receive a medical case file, including Case Information,\nPhysical Examination and Diagnostic Tests. Please conduct a thorough analysis based on the provided\ninformation and select the most appropriate diagnosis from the following four options. Provide the final answer prefixed with \"Answer:\". Do not answer with a full sentence. Just provide the letter of the correct choice.\n"""

diagnosis_arena_careful_confidence_self_verb_suffix = """After providing your answer, carefully assess your confidence in that answer by evaluating your thinking process. Note that this is a medical diagnosis question, so it is crucial to be cautious and consider all aspects of the case as well as your thinking process when assessing your confidence.
Finally, classify your confidence in your answer into one of the following classes based on how likely your answer is to be
correct:

- "Almost no chance" (0.0-0.1)
- "Highly unlikely" (0.1-0.2)
- "Chances are slight" (0.2-0.3)
- "Unlikely" (0.3-0.4)
- "Less than even" (0.4-0.5)
- "Better than even" (0.5-0.6)
- "Likely" (0.6-0.7)
- "Very good chance" (0.7-0.8)
- "Highly likely" (0.8-0.9)
- "Almost certain" (0.9-1.0)

Each category reflects the probability that your answer is correct.
At the very end of your output, format your answer and confidence as
Answer: $ANSWER
Confidence: $CLASS"""

deepseek_extra_prompt = """\n Note: Make sure to end your response like this:\n Answer: YOUR ANSWER\n Confidence: YOUR CONFIDENCE. Otherwise your answer will be considered invalid."""


sys_prompt_mcq_qwen25 = """You will be given a question. Think step by step, then answer the question by choosing one of the provided options. """
self_verb_suffix_qwen25 = """Assess your confidence in that answer by evaluating your thinking process.
Finally, classify your confidence in your answer into one of the following classes based on how likely your answer is to be
correct:

- "Almost no chance" (0.0-0.1)
- "Highly unlikely" (0.1-0.2)
- "Chances are slight" (0.2-0.3)
- "Unlikely" (0.3-0.4)
- "Less than even" (0.4-0.5)
- "Better than even" (0.5-0.6)
- "Likely" (0.6-0.7)
- "Very good chance" (0.7-0.8)
- "Highly likely" (0.8-0.9)
- "Almost certain" (0.9-1.0)

Each category reflects the probability that your answer is correct.
At the very end of your output, format your answer and confidence as
Answer: $ANSWER
Confidence: $CLASS

Stop after this.
"""

prompts = {
'gpqa_mcq': sys_prompt_mcq + self_verb_suffix,
'gpqa_free_answer': sys_prompt_free_answer + self_verb_suffix,
'mmlupro': sys_prompt_free_answer + self_verb_suffix,
'mmlupro_free_answer': sys_prompt_free_answer + self_verb_suffix,
'hle_mcq': sys_prompt_mcq + self_verb_suffix,
'hle': sys_prompt_free_answer + self_verb_suffix,
'livebench': livebench_sysprompt,
'simpleqa': sys_prompt_free_answer + self_verb_suffix,
'diagnosis_arena': diagnosis_arena_sysprompt + self_verb_suffix,
'diagnosis_arena_careful': diagnosis_arena_sysprompt + diagnosis_arena_careful_confidence_self_verb_suffix,
'gpqa_mcq_qwen25': sys_prompt_mcq_qwen25 + self_verb_suffix_qwen25,
'mmlupro_qwen25': sys_prompt_mcq_qwen25 + self_verb_suffix_qwen25,
'gpqa_free_qwen25': sys_prompt_free_answer_qwen25 + self_verb_suffix_qwen25,
'mmlupro_free_qwen25': sys_prompt_free_answer_qwen25 + self_verb_suffix_qwen25,
'diagnosis_arena_qwen25': diagnosis_arena_sysprompt + self_verb_suffix_qwen25,
'diagnosis_arena_careful_qwen25': diagnosis_arena_sysprompt + diagnosis_arena_careful_confidence_self_verb_suffix,
'livebench_qwen25': livebench_sysprompt + self_verb_suffix_qwen25
}