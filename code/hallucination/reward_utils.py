import re
from typing import Any, Dict, List

from deepeval.metrics import FaithfulnessMetric
from deepeval.models.llms import LocalModel
from .geval_prompt import EVALUATION_PROMPT_TEMPLATE

def build_faithfulness_metric(
    model_name: str,
    base_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    threshold: float = 0.7,
    include_reason: bool = False,
):
    def _factory():
        if "gpt" in model_name.lower():
            deepeval_llm = model_name
        else:
            deepeval_llm = LocalModel(
                model=model_name,
                api_key="Empty",
                base_url=base_url,
                temperature=0.15,
                generation_kwargs={
                    "max_tokens": max_tokens,
                    "top_p": top_p
                }
            )
        return FaithfulnessMetric(
            threshold=threshold,
            model=deepeval_llm,
            include_reason=include_reason,
            penalize_ambiguous_claims=True
        )
    
    return _factory

def _normalize_json_response(response: str) -> str:
        if not response:
            return "{}"
        
        response_clean = response.strip()
        
        json_block_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'`json\s*(.*?)\s*`',
            r'`\s*(.*?)\s*`'
        ]
        
        for pattern in json_block_patterns:
            match = re.search(pattern, response_clean, re.DOTALL)
            if match:
                response_clean = match.group(1).strip()
                break
        
        start_idx = response_clean.find('{')
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(response_clean[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break
            
            if brace_count == 0:
                response_clean = response_clean[start_idx:end_idx + 1]
        
        return response_clean

def overconf_norm(U, U_min):
    if U_min <= 0:
        return 0.0
    return max(0.0, U_min - U) / U_min
    
def build_response_item(index: int, completions: List[Dict], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'mode': kwargs['mode'][index],
        'query': kwargs['query'][index],
        'model_response_raw': completions[index][0]["content"],
        'gold_contexts': kwargs['gold_retrival_content'][index],
        'entropy': kwargs['avg_entropy'][index],
        "completion_id": kwargs['completion_ids'][index]
    } 
    
    
# =================Geval======================
def prepare_prompt(criteria, steps, query, document, summary, metric_name: str) -> str:
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        criteria=criteria,
        steps=steps,
        query=query,
        metric_name=metric_name,
        document=document,
        summary=summary,
    )
    return prompt

def get_geval_score(llm, criteria, steps, query, document, summary, metric_name, params, probability_normalize=False):
    prompt = prepare_prompt(
        criteria,
        steps,
        query,
        document,
        summary,
        metric_name,
    )
    if not params:
        params = {
                    'temperature': 0.1,
                    'max_tokens': 5000,
                    'top_p': 1,
                    'frequency_penalty': 0,
                    'presence_penalty': 0
                }
    if probability_normalize:
        params['n'] = 10
        
    response, _ = llm.chat(
            prompt, 
            params=params, 
            multi_response=probability_normalize, 
        )
    return response

def get_mean_score(result):
    """
    Calculate the mean score from a list of strings containing numeric values.
    Example input: ['5', '- Relevance: 5', '5']
    """
    if isinstance(result, str):
        result = [result]
    scores = []
    for score in result:
        try:
            match = re.search(r'\b\d+\b', score)
            if match:
                score_num = int(match.group())
                scores.append(score_num)
        except Exception as e:
            print(f"Error processing score '{score}': {e}")
    
    if scores:
        return sum(scores) / len(scores)
    else:
        return 2.5