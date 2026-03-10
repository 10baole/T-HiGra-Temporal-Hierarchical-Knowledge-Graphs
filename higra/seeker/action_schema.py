from typing import List, Union
from pydantic import BaseModel



class Parameter(BaseModel):
    '''
     - Each parameter_name can only be appeared once
    '''
    param_name: str
    param_value: Union[List[str], str]

class Function(BaseModel):
    '''
    - Each parameter_name can only be appeared once
    '''
    function_name: str
    function_parameters: List[Parameter]

class Action(BaseModel):
    '''
    Agent return response.
    '''
    # reasoning: str
    action: Function