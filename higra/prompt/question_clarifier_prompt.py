class QuestionClarifierPrompt:
    system_prompt = """
### Types of wh-questions:
1. **Wh-fronted question**: The *wh-word* (e.g., *what*, *who*, *which*) appears at the **beginning** of the sentence. This is the standard form in formal English.
   - Example (object question): *"What did she paint?"*
   - Example (subject question): *"Who wrote the book?"*

2. **Wh-in-situ question**: The *wh-word* stays in its **original position**, often at the end of the sentence. Common in **spoken/informal** English, but needs rewriting for clarity or formality.
   - Example: *"She painted what?"*
   - Example: *"He signed a contract with who?"*
---
### Instruction:
- If the question is in **wh-in-situ form**, rewrite it into a **wh-fronted object question** while keeping the original **meaning and objective** unchanged. Only modify the sentence structure — do not change its intent.
- The question should keep the original objective and meaning.
---
**Example Input:**  
"He signed a contract with who?"
**Example Output:**  
"Who did he sign a contract with?"

### Important Note
If you can not rewrite, you must return the original question, do not return empty rewrite question.
**Output Format**
{
    "reasoning": "",
    "rewrite_question": ""
}
""".strip()

    example_prompt = """
""".strip()

    user_prompt = """
### Input
{{
    "original_question": "{question}"
}}
""".strip()
