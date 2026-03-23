import json
from typing import List, Dict, Any
import re
import time
from .prompt import HALLUCINATION_TEST_PROMPTS_ALL_WRONG
from .client.llm_client_sync import LLMChat

from deepeval.test_case import LLMTestCase

from .geval_prompt import *
from .reward_utils import (_normalize_json_response, build_faithfulness_metric,
                           build_response_item, prepare_prompt,
                           get_mean_score, overconf_norm)


## ============================= parameters ============================= ##
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

MAX_WORKERS = 16
API_CONCURRENCY = 10
api_sem = threading.Semaphore(API_CONCURRENCY)


evaluation_metrics = {
    "completeness": (COMPLETENESS_SCORE_CRITERIA, COMPLETENESS_SCORE_STEPS),
    "coherence": (COHERENCE_SCORE_CRITERIA, COHERENCE_SCORE_STEPS),
    "consistency": (CONSISTENCY_SCORE_CRITERIA, CONSISTENCY_SCORE_STEPS),
    "fluency": (FLUENCY_SCORE_CRITERIA, FLUENCY_SCORE_STEPS),
    "relevance": (RELEVANCY_SCORE_CRITERIA, RELEVANCY_SCORE_STEPS),
}

judge_llm = LLMChat(
        model="Mistral-Small-3.2-24B-Instruct-2506-INT4",
        config_path="/app/grpo_training/fine_tune_utils/hallucination/config/models.yaml",
        cache_config={
            "enable": True,
            "cache_file": './Mistral-Small-3.2-24B-Instruct-2506-INT4/llm_cache.json'
        }
    )

metric_factory = build_faithfulness_metric(
    model_name=judge_llm.model_config['model'],
    base_url=judge_llm.model_config['local_base_url'],
    temperature=0.15,
    top_p=0.9,
    max_tokens=7000,
    threshold=0.7,
    include_reason=False
)

geval_params = {
                'temperature': 0.1,
                'max_tokens': 500,
                'top_p': 1,
                'frequency_penalty': 0,
                'presence_penalty': 0
            }

judge_params = {
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 5000,
    }

probability_normalize = False

# reward parameter
u_min_one_correct = 0.02
u_min_all_wrong = 0.05
gamma = 0.1
lam = 0.5
r_abstain = 1.0
r_not_abstain_base = 0.2
r_one_correct_if_abstain = 1
# geval parameter
metric_beta = 0.3
L_min = 50
L_max = 150
len_eta = 0.1


## ============================= functions ============================= ##
def get_geval_score_safe(llm, criteria, steps, query, document, summary, metric_name, params, probability_normalize=False):
    with api_sem:
        return get_geval_score(
            llm,
            criteria,
            steps,
            query,
            document,
            summary,
            metric_name,
            params,
            probability_normalize=probability_normalize
        )

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

def get_metric_score(query, document, summary, params, probability_normalize=False, executor=None):
    if executor is None:
        scores = []
        for eval_type, (criteria, steps) in evaluation_metrics.items():
            try:
                response = get_geval_score(
                    judge_llm,
                    criteria,
                    steps,
                    query,
                    document,
                    summary,
                    eval_type,
                    params,
                    probability_normalize=probability_normalize,
                )
                mean_score = get_mean_score(response)
                if eval_type == "fluency":
                    # 3 -> 1
                    mean_score = (mean_score - 1) / 2
                elif eval_type == "completeness":
                    # more important, double the score
                    mean_score = (mean_score - 1) / 4 * 2
                else:
                    # 5 -> 1
                    mean_score = (mean_score - 1) / 4
                scores.append(mean_score)
            except Exception as e:
                print(f"Error getting score for metric {eval_type}: {e}")
                base_score = (2.5 - 1) / 4
                scores.append(base_score)
        return sum(scores) / len(scores) if scores else 0.375

    def _one_metric(eval_type, criteria, steps):
        try:
            response = get_geval_score_safe(
                judge_llm, criteria, steps, query, document, summary, eval_type,
                params, probability_normalize=probability_normalize
            )
            mean_score = get_mean_score(response)
            return (mean_score - 1) / 4
        except Exception as e:
            print(f"Error getting score for metric {eval_type}: {e}")
            return (2.5 - 1) / 4
        
    futs = []
    for eval_type, (criteria, steps) in evaluation_metrics.items():
        futs.append(executor.submit(_one_metric, eval_type, criteria, steps))
        
    scores = [f.result() for f in as_completed(futs)]
    return sum(scores) / len(scores) if scores else 0.375
    
def judge_all_wrong_safe(query, model_answer):
    with api_sem:
        return judge_all_wrong(query, model_answer)

def judge_all_wrong(query, model_answer):
    prompt = HALLUCINATION_TEST_PROMPTS_ALL_WRONG.format(
        query=query,
        model_answer=model_answer
    )

    try:
        response, _ = judge_llm.chat(prompt, params=judge_params)
        cleaned = _normalize_json_response(response)
        try:
            response_json = json.loads(cleaned)
        except json.JSONDecodeError:
            response_json = {
                "verdict": "NOT_ABSTAINED",
                "reason": "Failed to parse JSON response."
            }
    except Exception as e:
        response_json = {
            "verdict": "NOT_ABSTAINED",
            "reason": f"Error during LLM call: {str(e)}"
        }
    return response_json

def hallucination_reward_function(completions: List[Dict], **kwargs) -> List[float]:
    start_time = time.time()
    required_keys = ['mode', 'query', 'gold_retrival_content', 'avg_entropy']
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(f"Missing required key in kwargs: {key}")

    responses = [build_response_item(i, completions, kwargs) for i in range(len(kwargs['mode']))]

    n = len(responses)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        score_futs = {
            ex.submit(reward_function_one, responses[i], responses[i]['entropy'], None): i
            for i in range(n)
        }
        
        scores = [0.0] * n
        for fut in as_completed(score_futs):
            i = score_futs[fut]
            try:
                score = fut.result()
            except Exception as e:
                score = 0.0
            scores[i] = float(score)
    
    end_time = time.time()
    print(f"Reward function 總計算時間: {end_time - start_time:.2f} 秒")
    return scores

def reward_function_one(item, entropy, executor=None):
    mode = item['mode']
    llm_response = item['model_response_raw']
    query = item['query']
    if mode == "all_wrong":
        result = judge_all_wrong_safe(query, llm_response)
        verdict = result.get("verdict", "NOT_ABSTAINED")
        if verdict == "ABSTAINED":
            return r_abstain
        else:
            overconf = overconf_norm(entropy, u_min_all_wrong)
            reward = -r_not_abstain_base - lam * overconf
            return reward
    elif mode == "one_correct":
        # judge if the response is abstained or not
        result = judge_all_wrong_safe(query, llm_response)
        verdict = result.get("verdict", "NOT_ABSTAINED")
        if verdict == "ABSTAINED":
            return -r_one_correct_if_abstain
        else:
            gold_contexts = item['gold_contexts']

            # geval scoring
            try:
                metric_score = get_metric_score(
                    query,
                    gold_contexts,
                    llm_response,
                    geval_params,
                    probability_normalize=probability_normalize,
                    executor=executor
                )
                print(f" Metric score: {metric_score:.4f}")
            except Exception as e:
                metric_score = 0.375  # default score in case of error

            pen_len = 0.0
            token_length = len(item['completion_id'])
            pen_len += max(0.0, (token_length - L_max) / L_max)
            pen_len += max(0.0, (L_min - token_length) / L_min)


            test_case = LLMTestCase(
                input=query,
                actual_output=llm_response,
                retrieval_context=[gold_contexts],
            )
            try:
                metric = metric_factory()
                metric.measure(test_case)
                score = metric.score
                print(f" Faithfulness score: {score:.4f}")
            except Exception as e:
                score = 0.0
            overconf = overconf_norm(entropy, u_min_one_correct)
            reward = score - gamma * (1 - score) * overconf
            reward += metric_beta * metric_score
            reward -= len_eta * pen_len
            del metric
            return reward
    else:
        return 0.0