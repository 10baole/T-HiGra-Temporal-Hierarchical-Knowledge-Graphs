class LLMShortAnswerExtractionPrompt:
    system_prompt="""
    ### Instruction
    - Extract the short answer from the given full answer.
    - Only return the short answer with no explanation.

    ### Note
    - For yes/no question, only return the label.
    """
    user_prompt="""
    ### Question
    {question}
    ### Full Answer
    {prediction}
    """