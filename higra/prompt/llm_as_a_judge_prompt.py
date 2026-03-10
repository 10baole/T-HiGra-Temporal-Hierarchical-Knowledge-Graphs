class LLMEvaluatePrompt:
    system_prompt="""
    ### Role
    You are an expert language model evaluator. Your task is evaluate the model prediction given a groundtruth.
    ### Instruction
    - If the prediction is no answer -> It is False
    - If the prediction contains the groundtruth (or have the same meaning as the groundtruth) -> It is True
    - If the prediction contains variations of the groundtruth -> It is True
    ### Note
    - The prediction may contains others information as explanation  -> ignore these
    - A question may yield multiple valid answers.
    """
    user_prompt="""
    ### Question
    {question}
    ### Ground Truth
    {groundtruth}
    ### Prediction
    {prediction}
    """

    response_format={
        'type': 'json_schema',
        'json_schema': {
            'name': 'Response',
            'description': '',
            'strict': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'explain': {'type': 'string'},
                    'predict': {'type': 'boolean'}
                },
                'required': ['explain', 'predict'],
                'additionalProperties': False
            }
        }
    }