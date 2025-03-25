import os
import json
import time
import re
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
from together import Together
from openai import OpenAI
from anthropic import Anthropic
import groq
import logging

#################################
# LOGGING SETUP
#################################
os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#################################
# HELPER LOG FUNCTIONS
#################################

def log_function_call(func):
    """Decorator that logs calls and results."""
    def wrapper(*args, **kwargs):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        if func.__name__ in ["get_together_completion", "get_openai_completion", "get_claude_completion", "get_groq_completion"]:
            log_line = f"{timestamp} - {func.__name__} called with kwargs: {kwargs}\n"
        else:
            log_line = f"{timestamp} - {func.__name__} called with args truncated, kwargs: {kwargs}\n"

        with open("logs/function_calls.txt", "a", encoding="utf-8") as f:
            f.write(log_line)

        result = func(*args, **kwargs)

        if isinstance(result, str) and len(result) > 200:
            result_summary = result[:200] + "... [truncated]"
        else:
            result_summary = result
            
        log_line_result = f"{timestamp} - {func.__name__} returned: {result_summary}\n\n"
        with open("logs/function_calls.txt", "a", encoding="utf-8") as f:
            f.write(log_line_result)

        return result
    return wrapper

def log_error(error_message):
    """Log errors to error log file."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    with open("logs/error_logs.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] ERROR: {error_message}\n\n")
    logger.error(error_message)

def log_debug(debug_message):
    """Log debug information to debug log file."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    with open("logs/debug_logs.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] DEBUG: {debug_message}\n\n")
    logger.debug(debug_message)

def log_processed_prompt(prompt_name, processed_prompt):
    """Log the processed prompt after placeholder replacement."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    truncated_prompt = processed_prompt[:300] + "..." if len(processed_prompt) > 300 else processed_prompt
    
    with open("logs/processed_prompts.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] PROCESSED {prompt_name}:\n{truncated_prompt}\n\n")

#################################
# FILE READING HELPERS
#################################

@log_function_call
def read_prompt_template(file_path):
    """Read a prompt template from file."""
    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        error_msg = f"Error reading prompt template {file_path}: {str(e)}"
        print("\n" + error_msg)
        log_error(error_msg)
        return None

#################################
# RESPONSE PARSING & EXTRACTION
#################################

def extract_thinking(response):
    """
    Extract <think>...</think> tags from response.
    Returns (thinking_text, response_after_thinking)
    """
    if not response or not isinstance(response, str):
        return "", response
        
    think_pattern = r'<think>(.*?)</think>\s*'
    think_match = re.search(think_pattern, response, re.DOTALL)
    
    if think_match:
        thinking = f"<think>{think_match.group(1)}</think>"
        response_after_thinking = re.sub(think_pattern, '', response, count=1, flags=re.DOTALL).strip()
        return thinking, response_after_thinking
    else:
        return "", response

def extract_function_calls(response):
    """
    Extract <function> search_func(...) </function> calls.
    Returns (cleaned_response, [function_calls])
    """
    if not response or not isinstance(response, str):
        log_error(f"Invalid response passed to extract_function_calls: {type(response)}")
        return "", []
        
    patterns = [
        r'<function>\s*search_func\((.*?)\)\s*</function>',
        r'<function>search_func\((.*?)\)</function>',
        r'<function>\s*search_func\s*\((.*?)\)\s*</function>'
    ]
    
    function_calls = []
    clean_response = response
    
    for pattern in patterns:
        calls = re.findall(pattern, response, flags=re.DOTALL)
        if calls:
            function_calls = calls
            clean_response = re.sub(pattern, '', response, flags=re.DOTALL)
            log_debug(f"Extract function calls - Found pattern match: {pattern}")
            break
    
    log_debug(f"Extract function calls - Input length: {len(response)}")
    log_debug(f"Extract function calls - Found {len(function_calls)} function calls")
    
    clean_response = clean_response.strip()
    
    if not clean_response and function_calls:
        log_debug("Warning: After removing function calls, response is empty")
    
    return clean_response, function_calls

def create_detailed_message(thinking, response_after_thinking, final_response, search_history=None, critique=None):
    """Create an assistant message with optional fields: thinking, search_history, critique, etc."""
    msg = {
        "role": "assistant",
        "thinking": thinking
    }
    
    if response_after_thinking and response_after_thinking != final_response:
        msg["raw_response_after_thinking"] = response_after_thinking
        
    if search_history:
        msg["search_history"] = search_history
    
    if critique:
        msg["critique"] = critique
        
    msg["content"] = final_response
    return msg

@log_function_call
def save_search_results(search_params, search_result):
    """Append search info to search_history.json and return the record."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        search_record = {
            "timestamp": timestamp,
            "parameters": search_params,
            "results": search_result
        }
        try:
            with open('search_history.json', 'r', encoding='utf-8') as f:
                all_history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_history = []
        all_history.append(search_record)
        with open('search_history.json', 'w', encoding='utf-8') as f:
            json.dump(all_history, f, ensure_ascii=False, indent=2)
        return search_record
    except Exception as e:
        error_msg = f"Error saving search results: {str(e)}"
        print(f"\n{error_msg}")
        log_error(error_msg)
        return {"error": str(e)}

#################################
# CLIENT INITIALIZATIONS
#################################

def get_openai_client():
    """Initialize OpenAI client with API key."""
    try:
        with open("openai.key", "r", encoding="utf-8") as f:
            openai_key = f.read().strip()
        return OpenAI(api_key=openai_key)
    except Exception as e:
        error_msg = f"Error initializing OpenAI client: {str(e)}"
        print("\n" + error_msg)
        log_error(error_msg)
        return None

def get_together_client():
    """Initialize Together client with API key."""
    try:
        with open("together.key", "r", encoding="utf-8") as f:
            together_key = f.read().strip()
        return Together(api_key=together_key)
    except Exception as e:
        error_msg = f"Error initializing Together client: {str(e)}"
        print("\n" + error_msg)
        log_error(error_msg)
        return None

def get_claude_client():
    """Initialize Claude client with API key."""
    try:
        with open("claude.key", "r", encoding="utf-8") as f:
            claude_key = f.read().strip()
        return Anthropic(api_key=claude_key)
    except Exception as e:
        error_msg = f"Error initializing Claude client: {str(e)}"
        print("\n" + error_msg)
        log_error(error_msg)
        return None

def get_groq_client():
    """Initialize Groq client with API key."""
    try:
        with open("groq.key", "r", encoding="utf-8") as f:
            groq_key = f.read().strip()
        return groq.Client(api_key=groq_key)
    except Exception as e:
        error_msg = f"Error initializing Groq client: {str(e)}"
        print("\n" + error_msg)
        log_error(error_msg)
        return None

#################################
# COMPLETION CALLS (OPENAI / TOGETHER / CLAUDE / GROQ)
#################################

@log_function_call
def get_openai_completion(prompt, model="o3-mini"):
    """Use OpenAI for responses."""
    try:
        log_processed_prompt(f"OpenAI_{model}", prompt)
        
        client = get_openai_client()
        if not client:
            return None

        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content if completion.choices else None

    except Exception as e:
        error_msg = f"Error in get_openai_completion for model {model}: {str(e)}"
        print("\n" + error_msg)
        log_error(error_msg)
        return None

@log_function_call
def get_together_completion(prompt, include_thinking=False):
    """Use Together AI DeepSeek-R1 for responses."""
    try:
        log_processed_prompt("Together_DeepSeek-R1", prompt)
        
        client = get_together_client()
        if not client:
            return None
            
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            stream=False
        )
        final_text = completion.choices[0].message.content if completion.choices else ""
        
        if include_thinking:
            return final_text if final_text.strip() else None
        else:
            cleaned_text = re.sub(r'<think>.*?</think>\s*', '', final_text, flags=re.DOTALL)
            return cleaned_text if cleaned_text.strip() else None

    except Exception as e:
        error_msg = f"Error in get_together_completion: {str(e)}"
        print("\n" + error_msg)
        log_error(error_msg)
        return None

@log_function_call
def get_claude_completion(prompt, include_thinking=False):
    """Use Claude for responses."""
    try:
        log_processed_prompt("Claude_3-7-sonnet", prompt)
        
        client = get_claude_client()
        if not client:
            return None

        completion = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4000
        )
        final_text = completion.content[0].text if completion.content else ""
        return final_text if final_text.strip() else None

    except Exception as e:
        error_msg = f"Error in get_claude_completion: {str(e)}"
        print("\n" + error_msg)
        log_error(error_msg)
        return None

@log_function_call
def get_groq_completion(prompt, include_thinking=False):
    """Use Groq DeepSeek-R1-Distill-Qwen-32B for responses."""
    try:
        log_processed_prompt("Groq_DeepSeek", prompt)
        
        client = get_groq_client()
        if not client:
            return None

        completion = client.chat.completions.create(
            model="deepseek-r1-distill-qwen-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            top_p=0.95,
            stream=False
        )
        final_text = completion.choices[0].message.content if completion.choices else ""
        
        if include_thinking:
            return final_text if final_text.strip() else None
        else:
            cleaned_text = re.sub(r'<think>.*?</think>\s*', '', final_text, flags=re.DOTALL)
            return cleaned_text if cleaned_text.strip() else None

    except Exception as e:
        error_msg = f"Error in get_groq_completion: {str(e)}"
        print("\n" + error_msg)
        log_error(error_msg)
        return None

#################################
# THREE-PROMPT WORKFLOW FUNCTIONS
#################################

def extract_ner_from_conversation(conversation_history):
    """
    Extract named entities (hotel preferences) from the conversation using NER prompt.
    Returns a dictionary of extracted preferences.
    """
    try:
        simple_conversation = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in conversation_history
        ]
        
        ner_template = read_prompt_template("ner.md")
        if not ner_template:
            log_error("Failed to read ner.md template")
            return {}
            
        ner_prompt = ner_template.replace("{conv}", json.dumps(simple_conversation, ensure_ascii=False, indent=2))
        ner_response = get_openai_completion(ner_prompt, model="o3-mini")
        
        if not ner_response:
            log_error("No NER response generated")
            return {}
            
        dict_pattern = r'```python\s*({[\s\S]*?})\s*```'
        dict_match = re.search(dict_pattern, ner_response)
        
        if dict_match:
            try:
                preferences_dict = eval(dict_match.group(1))
                log_debug(f"Extracted preferences: {json.dumps(preferences_dict, ensure_ascii=False)}")
                return preferences_dict
            except Exception as e:
                log_error(f"Error parsing extracted preferences: {str(e)}")
                return {}
        else:
            # Try direct extraction if no code block
            dict_pattern = r'({[\s\S]*?})'
            dict_match = re.search(dict_pattern, ner_response)
            if dict_match:
                try:
                    preferences_dict = eval(dict_match.group(1))
                    log_debug(f"Extracted preferences (direct): {json.dumps(preferences_dict, ensure_ascii=False)}")
                    return preferences_dict
                except Exception as e:
                    log_error(f"Error with direct parsing: {str(e)}")
                    return {}
            else:
                log_error("No valid preferences dictionary found in NER response")
                return {}
    except Exception as e:
        error_msg = f"Error in NER extraction: {str(e)}"
        log_error(error_msg)
        return {}

def process_search_call(extracted_preferences):
    """
    Determine if a search should be triggered based on extracted preferences.
    Returns a string with the <function> search_func(...) call or "" if no search.
    """
    try:
        search_call_template = read_prompt_template("search_call.md")
        if not search_call_template:
            log_error("Failed to read search_call.md template")
            return ""
        
        search_call_prompt = search_call_template.replace(
            "{preferences}",
            json.dumps(extracted_preferences, ensure_ascii=False, indent=2)
        )
        
        search_call_response = get_openai_completion(search_call_prompt, model="o3-mini")
        if not search_call_response:
            log_error("No search call response generated")
            return ""
        
        search_call_response = search_call_response.strip()
        if "<function>" in search_call_response:
            return search_call_response
        return ""
    except Exception as e:
        error_msg = f"Error in search call processing: {str(e)}"
        log_error(error_msg)
        return ""

def debug_search_status(model_type, enable_search, previous_data):
    """Log the current search status for debugging."""
    log_debug(f"{model_type}: ENABLE_SEARCH={enable_search}")
    if previous_data:
        log_debug(f"{model_type}: Previous preferences count: {len(previous_data.get('preferences', {}))}")
        log_debug(f"{model_type}: Previous num_matches: {previous_data.get('num_matches', 'None')}")
    else:
        log_debug(f"{model_type}: No previous data available")

def process_search_simulation_openai(response_after_thinking, conversation_history):
    """
    Process the function calls in response_after_thinking, simulate search, 
    and return the search record (or None).
    """
    if not response_after_thinking or not response_after_thinking.strip():
        log_error("Empty response passed to search processing")
        return None
    
    log_debug(f"Processing search in response of length: {len(response_after_thinking)}")
    
    try:
        clean_response, function_calls = extract_function_calls(response_after_thinking)
        if not function_calls:
            log_debug("No function calls detected - returning None")
            return None
        
        log_debug(f"Found {len(function_calls)} function calls to process")
        function_call_content = function_calls[0]
        
        search_template = read_prompt_template("search_simulator.md")
        if not search_template:
            log_error("Failed to read search_simulator.md template")
            return None
        
        search_prompt = search_template.replace("{search_query}", function_call_content.strip())
        search_response = get_openai_completion(search_prompt)
        
        if not search_response:
            log_error("No search result received for query")
            return None
        
        log_debug(f"Search result received of length: {len(search_response)}")
        
        # Save to search_history.json
        search_record = save_search_results(function_call_content, search_response)
        return search_record
        
    except Exception as e:
        error_msg = f"Unexpected error in search processing: {str(e)}"
        log_error(error_msg)
        return None

def get_critic_evaluation(original_prompt, conversation_history, search_record, assistant_response):
    """
    Get a critique of the assistant's response using critic.md.
    Returns a JSON object with score and reason.
    """
    try:
        simple_conversation = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in conversation_history
        ]
        
        critic_template = read_prompt_template("critic.md")
        if not critic_template:
            log_error("Failed to read critic.md template")
            return None
        
        # Insert placeholders
        critic_prompt = critic_template.replace("{original_prompt}", original_prompt)
        critic_prompt = critic_prompt.replace("{conversation}", json.dumps(simple_conversation, ensure_ascii=False, indent=2))
        critic_prompt = critic_prompt.replace("{last_response}", assistant_response)
        
        # Only show search results if search_record AND they were shown to the assistant
        if search_record and search_record.get("show_results_to_actor", False) == True:
            critic_prompt = critic_prompt.replace("{search_history}", json.dumps(search_record["results"], ensure_ascii=False, indent=2))
        else:
            # Remove the entire block
            critic_prompt = critic_prompt.replace("<last_search_output>\n{search_history}\n</last_search_output>", "")
        
        critique_response = get_together_completion(critic_prompt)
        if not critique_response:
            log_error("No critique response generated")
            return None
        
        # Try to find JSON in the critique response
        json_pattern = r'(\{[\s\S]*\})'
        json_match = re.search(json_pattern, critique_response)
        if json_match:
            try:
                critique_json = json.loads(json_match.group(1))
                log_debug(f"Parsed critique: {json.dumps(critique_json, ensure_ascii=False)}")
                return critique_json
            except Exception as e:
                log_error(f"Error parsing critique JSON: {str(e)}")
                return None
        else:
            log_error("No valid JSON found in critique response")
            return None

    except Exception as e:
        error_msg = f"Error in critique evaluation: {str(e)}"
        log_error(error_msg)
        return None

def process_search_results(search_record):
    """
    Determine if search results are displayed to actor. 
    If > 50 matches, do not show; if <= 50, show.
    """
    if not search_record:
        return False, ""
    
    try:
        search_response = search_record.get("results", "")
        if not search_response:
            log_debug("Empty search results found")
            return False, ""
        
        num_matches = None
        patterns = [
            r'"Number of matches":\s*(\d+)',
            r'Number of matches:\s*(\d+)',
            r'Found (\d+) matches',
            r'(\d+) results found',
            r'(\d+) hotels match'
        ]
        for pattern in patterns:
            m = re.search(pattern, search_response, re.IGNORECASE)
            if m:
                try:
                    num_matches = int(m.group(1))
                    log_debug(f"Found {num_matches} matches in search results using pattern: {pattern}")
                    break
                except:
                    continue
        
        # If no match found, check if response says "no matches"
        if num_matches is None:
            no_matches_patterns = [
                r'no matches',
                r'no results',
                r'0 matches',
                r'0 results'
            ]
            for pattern in no_matches_patterns:
                if re.search(pattern, search_response, re.IGNORECASE):
                    log_debug("Search explicitly mentions no matches")
                    num_matches = 0
                    break
        
        # Fallback if still None
        if num_matches is None:
            hotel_name_count = len(re.findall(r'Hotel name:', search_response, re.IGNORECASE))
            if hotel_name_count > 0:
                num_matches = hotel_name_count
                log_debug(f"Fallback: Estimated {num_matches} matches by counting 'Hotel name:' occurrences")
            else:
                line_count = len([line for line in search_response.split('\n') if line.strip()])
                num_matches = line_count
                log_debug(f"Last resort: Setting matches to line count: {num_matches}")
        
        if num_matches is None:
            log_debug("Could not determine number of matches, defaulting to 100")
            num_matches = 100
        
        search_record["num_matches"] = num_matches
        
        log_debug(f"Search result comparison: num_matches={num_matches}, threshold=50, comparison={num_matches <= 50}")
        if num_matches > 50:
            log_debug(f"Will NOT show search results to actor ({num_matches} matches > 50)")
            return False, ""
        else:
            log_debug(f"Will show search results to actor ({num_matches} matches ≤ 50)")
            return True, search_response
        
    except Exception as e:
        error_msg = f"Error processing search results: {str(e)}"
        log_error(error_msg)
        return False, ""

def process_model_response(
    model_type: str,
    conversation: List[Dict[str, Any]],
    ENABLE_SEARCH: bool,
    EVALUATE_RESPONSES: bool,
    previous_data: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Process a single turn for the specified model (together/claude/groq).
    - NER extraction
    - Possibly search
    - Possibly show search results
    - Generate final response
    - Critique if EVALUATE_RESPONSES
    """
    print(f"Processing {model_type.upper()} response...")

    if previous_data is None:
        previous_data = {
            "preferences": {},
            "num_matches": 100
        }

    # Create a simpler conversation (role + content only) to feed into prompts
    simple_conversation = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in conversation
    ]

    extracted_preferences = {}
    search_record = None
    show_results_to_actor = False
    num_matches = previous_data["num_matches"]

    # STEP 1: Extract NER and decide whether to do a new search
    if ENABLE_SEARCH:
        extracted_preferences = extract_ner_from_conversation(simple_conversation)
        log_debug(f"{model_type}: Extracted preferences: {json.dumps(extracted_preferences, ensure_ascii=False)}")

        preferences_changed = (extracted_preferences != previous_data["preferences"])
        many_previous_results = (previous_data["num_matches"] > 10)

        log_debug(f"{model_type}: Preferences changed: {preferences_changed}")
        log_debug(f"{model_type}: Previous search had many results (> 10): {many_previous_results}")

        # Decide if we should trigger a new search
        should_trigger_search = many_previous_results or (preferences_changed and not many_previous_results)
        if should_trigger_search:
            if many_previous_results:
                log_debug(f"{model_type}: Search triggered because previous results > 10")
            else:
                log_debug(f"{model_type}: Search triggered because preferences changed and previous results <= 10")

            # Possibly generate a <function> search_func(...) call
            search_call_result = process_search_call(extracted_preferences)
            log_debug(f"{model_type}: Search call result length: {len(search_call_result) if search_call_result else 0}")

            # If a search call was indicated, simulate the search
            if search_call_result:
                try:
                    search_record = process_search_simulation_openai(search_call_result, simple_conversation)
                    if search_record and "num_matches" in search_record:
                        num_matches = search_record["num_matches"]
                except Exception as e:
                    error_msg = f"{model_type}: Error during search processing: {str(e)}"
                    log_error(error_msg)
                    print(f"WARNING: {error_msg}")
        else:
            log_debug(f"{model_type}: No new search triggered.")

    # STEP 2: If we have a search record, decide whether to show results
    if search_record:
        show_results_to_actor, search_text = process_search_results(search_record)
        search_record["show_results_to_actor"] = show_results_to_actor
        if "num_matches" in search_record:
            num_matches = search_record["num_matches"]
    else:
        search_text = ""

    # STEP 3: Build the final actor prompt
    agent_template = read_prompt_template("actor.md")
    if not agent_template:
        log_error(f"{model_type}: Failed to read actor.md template")
        return None

    if show_results_to_actor:
        # Show both the search listings and the numeric match count
        agent_prompt = (
            agent_template
            .replace("{conv}", json.dumps(simple_conversation, ensure_ascii=False, indent=2))
            .replace("{search}", search_text)
            .replace("{num_matches}", str(num_matches))
        )
    else:
        # Hide everything if > 50 matches
        agent_prompt = (
            agent_template
            .replace("{conv}", json.dumps(simple_conversation, ensure_ascii=False, indent=2))
            .replace("{search}", "")
            .replace("{num_matches}", "")
        )

    # STEP 4: Get assistant response
    assistant_response = None
    if model_type == 'together':
        assistant_response = get_together_completion(agent_prompt)
    elif model_type == 'claude':
        assistant_response = get_claude_completion(agent_prompt)
    elif model_type == 'groq':
        assistant_response = get_groq_completion(agent_prompt)

    if not assistant_response:
        log_debug(f"No {model_type} assistant response generated; skipping.")
        print(f"No {model_type} assistant response generated; skipping.")
        return None

    # STEP 5: Extract thinking and clean any function calls
    thinking, response_after_thinking = extract_thinking(assistant_response)
    final_response, _ = extract_function_calls(response_after_thinking)
    if not final_response.strip():
        final_response = response_after_thinking

    # STEP 6: Critique the response (if enabled)
    critique = None
    if EVALUATE_RESPONSES:
        original_prompt = read_prompt_template("actor.md")
        if original_prompt:
            original_prompt = original_prompt.replace("{conv}", "").replace("{search}", "").replace("{num_matches}", "").strip()
            critique = get_critic_evaluation(
                original_prompt,
                simple_conversation,
                search_record,
                final_response
            )
            if critique:
                log_debug(f"{model_type}: Critique score: {critique.get('score', 'N/A')} / total_score: {critique.get('total_score', 'N/A')}")

    # STEP 7: Create assistant message and add to conversation
    assistant_msg = create_detailed_message(
        thinking,
        response_after_thinking,
        final_response,
        search_record,
        critique
    )
    conversation.append(assistant_msg)

    # Print the assistant’s final response and critic score (if any)
    print(f"Assistant ({model_type}): {final_response}")
    if critique and "total_score" in critique:
        print(f"Original Critic Score: {critique['total_score']}")
    print()  # extra newline

    # Return next-step data (preferences, num_matches)
    return {
        "message": assistant_msg,
        "data": {
            "preferences": extracted_preferences,
            "num_matches": num_matches
        }
    }

#################################
# REGENERATION LOGIC
#################################

def get_together_completion_for_response(prompt, max_retries=3, retry_delay=5):
    """
    Similar to get_together_completion, but with retry logic
    and no trimming <think> from the final text.
    """
    for attempt in range(max_retries):
        try:
            client = get_together_client()
            if not client:
                return None
                
            completion = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                stream=False
            )
            final_text = completion.choices[0].message.content if completion.choices else ""
            cleaned_text = re.sub(r'<think>.*?</think>\s*', '', final_text, flags=re.DOTALL)
            return cleaned_text.strip() if cleaned_text.strip() else None

        except Exception as e:
            error_msg = f"Error in get_together_completion_for_response: {str(e)}"
            log_error(error_msg)
            if attempt < max_retries - 1:
                log_debug(f"Retrying in {retry_delay}s... Attempt {attempt+1} / {max_retries}")
                time.sleep(retry_delay)
    
    log_error("All retry attempts failed for get_together_completion_for_response")
    return None

def parse_critic_response(response_text):
    """Attempt to parse JSON from the response text with improved debugging."""
    if not response_text:
        log_debug("parse_critic_response: Empty response received")
        return {
            "raw_response": "",
            "total_score": None,
            "summary": "No response to parse"
        }

    # Log the first 500 chars of the raw response for debugging
    log_debug(f"parse_critic_response: Raw response (first 500 chars):\n{response_text[:500]}")
    
    # Try a simple big JSON pattern
    json_pattern = r'(\{[\s\S]*?\})'
    matches = list(re.finditer(json_pattern, response_text))
    log_debug(f"parse_critic_response: Found {len(matches)} potential JSON matches")
    
    for idx, m in enumerate(matches):
        candidate = m.group(1)
        log_debug(f"parse_critic_response: Examining JSON candidate #{idx+1} (length: {len(candidate)})")
        try:
            parsed = json.loads(candidate)
            # Log what keys are present in the parsed object
            log_debug(f"parse_critic_response: Successfully parsed JSON with keys: {list(parsed.keys())}")
            
            if "total_score" in parsed or "adherence_to_search" in parsed:
                log_debug(f"parse_critic_response: Found usable JSON with score info: {parsed.get('total_score')}")
                return parsed
            else:
                log_debug(f"parse_critic_response: JSON missing required score fields")
        except Exception as e:
            log_debug(f"parse_critic_response: Parse error for candidate #{idx+1}: {str(e)}")
            continue

    # Additional pattern attempts
    alternative_patterns = [
        r'"total_score"\s*:\s*(\d+\.?\d*)',
        r'total score:\s*(\d+\.?\d*)'
    ]
    
    for pattern in alternative_patterns:
        log_debug(f"parse_critic_response: Trying alternative pattern: {pattern}")
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                log_debug(f"parse_critic_response: Found score {score} using alternative pattern")
                return {
                    "raw_response": response_text,
                    "total_score": score,
                    "summary": "Score extracted via regex"
                }
            except:
                pass

    # If all parsing attempts fail
    log_debug("parse_critic_response: All parsing attempts failed, returning raw response")
    return {
        "raw_response": response_text,
        "total_score": None,
        "summary": "Failed to parse critic response"
    }

def regenerate_low_score_response(
    assistant_item: Dict[str, Any],
    conversation: List[Dict[str, Any]],
    score_threshold: float = 8.5
):
    """
    Regenerate a single assistant response in-memory 
    if it has total_score <= score_threshold.
    1) Build updated prompt from critic_regen.md
    2) Re-run the critic with critic.md using the same parsing as initial evaluation
    """
    if "critique" not in assistant_item:
        log_debug("regenerate_low_score_response: No critique found in assistant_item")
        return  # no critique means we can't do a score check
    
    if "total_score" not in assistant_item["critique"]:
        log_debug("regenerate_low_score_response: No total_score found in critique")
        return  # no total_score to compare
    
    total_score = assistant_item["critique"]["total_score"]
    if total_score is None or total_score > score_threshold:
        log_debug(f"regenerate_low_score_response: Score {total_score} is above threshold {score_threshold}")
        return  # no need to regenerate
    
    # We do have a low-scoring message => regenerate
    logger.info(f"Regenerating low-score response (score={total_score}).")

    # 1) Build conversation context
    idx = conversation.index(assistant_item)
    conversation_context = []
    for i in range(idx):
        role = conversation[i]["role"]
        content = conversation[i]["content"]
        conversation_context.append(f"{role}: {content}")
    conversation_context_str = "\n\n".join(conversation_context)

    # 2) Grab existing response
    existing_response = assistant_item["content"]

    # 3) Summarize the critic's reasons
    #    Original critique can have categories or summary
    critic_analysis = []
    for key, val in assistant_item["critique"].items():
        if key in ("score", "total_score"):
            continue
        if isinstance(val, dict):
            section_lines = [f"## {key}"]
            if "strengths" in val:
                section_lines.append(f"**Strengths**: {val['strengths']}")
            if "improvement_areas" in val:
                section_lines.append(f"**Improvement Areas**: {val['improvement_areas']}")
            if len(section_lines) > 1:
                critic_analysis.append("\n".join(section_lines))
        elif key == "summary":
            critic_analysis.append(f"## Summary\n{val}")

    critic_reason_str = "\n\n".join(critic_analysis)
    log_debug(f"regenerate_low_score_response: Critic analysis:\n{critic_reason_str}")

    # 4) Possibly pull search history if it was shown
    search_history_str = ""
    if "search_history" in assistant_item:
        sr = assistant_item["search_history"]
        if isinstance(sr, dict):
            # only if sr["show_results_to_actor"] == True and sr["num_matches"] <= 50
            if sr.get("show_results_to_actor") and sr.get("num_matches", 9999) <= 50:
                search_history_str = sr.get("results", "")
                log_debug(f"regenerate_low_score_response: Including search history of length {len(search_history_str)}")

    # 5) Read the response updater template (critic_regen.md)
    regen_template = read_prompt_template("critic_regen.md")
    if not regen_template:
        logger.error("Could not read critic_regen.md template; aborting regeneration.")
        return

    updated_prompt = (
        regen_template
        .replace("{conversation_context}", conversation_context_str)
        .replace("{last_response}", existing_response)
        .replace("{critic_reason}", critic_reason_str)
        .replace("{search_history}", search_history_str)
    )
    log_debug(f"regenerate_low_score_response: Regen prompt length: {len(updated_prompt)}")

    # 6) Call Together to regenerate
    logger.info("Calling Together to regenerate low-score response...")
    regenerated_response = get_together_completion_for_response(updated_prompt)
    if not regenerated_response:
        logger.error("Regeneration call returned None or empty.")
        return
    log_debug(f"regenerate_low_score_response: Generated new response of length {len(regenerated_response)}")

    # 7) Store in "regenerated_content"
    assistant_item["regenerated_content"] = regenerated_response

    # 8) Re-run the critic on the regenerated response using critic.md
    logger.info("Re-running the critic on the regenerated response...")
    
    # Prepare a new critic prompt
    original_prompt = read_prompt_template("actor.md")
    if original_prompt:
        # remove placeholders
        original_prompt = original_prompt.replace("{conv}", "").replace("{search}", "").replace("{num_matches}", "").strip()
    else:
        original_prompt = "Default Actor Prompt"

    critic_template = read_prompt_template("critic.md")
    if not critic_template:
        logger.error("Could not read critic.md for final evaluation.")
        return

    critic_prompt = critic_template.replace("{original_prompt}", original_prompt)
    critic_prompt = critic_prompt.replace("{conversation}", conversation_context_str)
    critic_prompt = critic_prompt.replace("{last_response}", regenerated_response)

    # Only include search history if it was shown
    if search_history_str.strip():
        critic_prompt = critic_prompt.replace("{search_history}", search_history_str)
    else:
        critic_prompt = critic_prompt.replace(
            "<last_search_output>\n{search_history}\n</last_search_output>", ""
        )
    log_debug(f"regenerate_low_score_response: Critic prompt length: {len(critic_prompt)}")

    # Use the same approach as in get_critic_evaluation
    regen_critic_result = get_together_completion(critic_prompt, include_thinking=False)
    if not regen_critic_result:
        logger.error("Failed to get critic evaluation for regenerated response.")
        return
    
    # Log the raw critic response for debugging
    log_debug(f"regenerate_low_score_response: Raw critic response (first 1000 chars):\n{regen_critic_result[:1000]}")

    # 9) Parse the critic using the SAME approach as the initial evaluation
    json_pattern = r'(\{[\s\S]*\})'
    json_match = re.search(json_pattern, regen_critic_result)
    
    regen_critic_evaluation = None
    if json_match:
        try:
            regen_critic_evaluation = json.loads(json_match.group(1))
            log_debug(f"regenerate_low_score_response: Successfully parsed JSON with keys: {list(regen_critic_evaluation.keys())}")
        except Exception as e:
            log_error(f"Error parsing critique JSON for regenerated response: {str(e)}")
            regen_critic_evaluation = {"error": str(e), "raw_response": regen_critic_result[:500] + "..."}
    else:
        log_error("No valid JSON found in critique response for regenerated content")
        regen_critic_evaluation = {"error": "No JSON found", "raw_response": regen_critic_result[:500] + "..."}
    
    assistant_item["regenerated_content_critic_evaluation"] = regen_critic_evaluation
    
    # 10) Ensure the new score is displayed in the console
    new_score = regen_critic_evaluation.get("total_score") if regen_critic_evaluation else None
    if new_score is not None:
        logger.info(f"Regeneration improved score from {total_score} to {new_score}")
    else:
        logger.warning(f"Regeneration completed but could not determine new score")

    logger.info(f"Regeneration + new critic complete for this low-scoring response. (score was {total_score})")
    
#################################
# MAIN - Merged Code
#################################

def main():
    ENABLE_SEARCH = True
    EVALUATE_RESPONSES = True

    log_debug("Starting merged main process with search and immediate regen of low-scoring answers.")
    
    # We keep a single conversation
    conversation = []
    
    # For tracking preferences + search results
    previous_data = {
        "preferences": {},
        "num_matches": 10  # so that a search can happen on first user input
    }
    
    model_type = 'together'  # Hard-coded to use the Together model, as an example.
    
    print("========== HOTEL SEARCH ASSISTANT ==========")
    print("Enter 'exit' or 'quit' to end the conversation.\n")

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting.")
            break

        # Add user message to conversation
        user_msg = {"role": "user", "content": user_input}
        conversation.append(user_msg)

        # Process with our multi-step approach
        result = process_model_response(model_type, conversation, ENABLE_SEARCH, EVALUATE_RESPONSES, previous_data)
        if result and "data" in result:
            previous_data = result["data"]
        
        # If there's a new assistant message with a critique, check if it's low score
        if conversation and conversation[-1]["role"] == "assistant":
            assistant_item = conversation[-1]
            # If it has a critique with total_score, see if we need to regenerate
            if "critique" in assistant_item:
                crit = assistant_item["critique"]
                old_score = crit.get("total_score")
                if old_score is not None and old_score <= 8.5:
                    # Call immediate regeneration
                    regenerate_low_score_response(assistant_item, conversation, score_threshold=8.5)
                    
                    # If there's a new regenerated text, show it along with the new critic score
                    if "regenerated_content" in assistant_item:
                        print("\n--- REGENERATED LOW-SCORE RESPONSE ---")
                        print(f"Original Assistant Text:\n{assistant_item['content']}")
                        print(f"(Original Critic Score: {old_score})\n")
                        print(f"Assistant (regenerated): {assistant_item['regenerated_content']}")

                        if "regenerated_content_critic_evaluation" in assistant_item:
                            new_eval = assistant_item["regenerated_content_critic_evaluation"]
                            new_score = new_eval.get("total_score")
                            
                            if new_score is not None:
                                print(f"(New Critic Score: {new_score})")
                                
                                # Display summary or improvement areas if available
                                if "summary" in new_eval:
                                    print(f"\nCritic Summary: {new_eval['summary']}")
                                elif "improvement_areas" in new_eval:
                                    print(f"\nImprovement Areas: {new_eval['improvement_areas']}")
                            else:
                                # If no new score was found, show a warning and some of the raw response
                                print("\n(WARNING: Could not parse new critic score)")
                                raw_resp = new_eval.get("raw_response", "")
                                if raw_resp:
                                    # Show first 200 chars of raw response for debugging
                                    print(f"Critic raw response (excerpt): {raw_resp[:200]}...")
                        else:
                            print("\n(WARNING: No critic evaluation available for regenerated content)")
                        print("---------------------------------------\n")


if __name__ == "__main__":
    main()