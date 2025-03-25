from flask import Flask, request, jsonify, render_template, send_file
import threading
import time
import uuid
import json
import os
import logging
from datetime import datetime
import concurrent.futures

# Import functions from your existing code
from paste import (
    extract_ner_from_conversation,
    process_search_call,
    process_search_simulation_openai,
    get_critic_evaluation,
    get_together_completion,
    get_claude_completion,
    get_openai_completion,
    get_groq_completion,
    extract_thinking,
    extract_function_calls,
    read_prompt_template
)

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory storage for conversations and tasks
conversations = {}
tasks = {}
task_results = {}

# Directory for saved conversations
SAVED_CONVS_DIR = "saved_conversations"
os.makedirs(SAVED_CONVS_DIR, exist_ok=True)

def init_conversation():
    """Initialize a new conversation with empty state."""
    return {
        "messages": [],
        "chosen_responses": [],
        "rejected_responses": [],
        "preferences": {},
        "num_matches": 100,
        "workflow_state": {
            "ner": {"status": "pending", "result": None},
            "search_decision": {"status": "pending", "result": None},
            "search": {"status": "pending", "result": None},
            "model1_response": {"status": "pending", "result": None, "model": "together"},
            "model2_response": {"status": "pending", "result": None, "model": "claude"},
            "critique": {"status": "pending", "result": None},
            "regeneration": {"status": "pending", "result": None}
        }
    }

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/conversation', methods=['POST'])
def create_conversation():
    """Create a new conversation."""
    data = request.json or {}
    model1 = data.get('model1', 'together')
    model2 = data.get('model2', 'claude')
    
    conversation_id = str(uuid.uuid4())
    conversations[conversation_id] = init_conversation()
    
    # Set the models
    conversations[conversation_id]["workflow_state"]["model1_response"]["model"] = model1
    conversations[conversation_id]["workflow_state"]["model2_response"]["model"] = model2
    
    return jsonify({"conversation_id": conversation_id})

@app.route('/api/conversation/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """Get the current state of a conversation."""
    if conversation_id not in conversations:
        return jsonify({"error": "Conversation not found"}), 404
    
    return jsonify(conversations[conversation_id])

@app.route('/api/conversation/<conversation_id>/message', methods=['POST'])
def add_message(conversation_id):
    """Add a user message and trigger the processing workflow."""
    if conversation_id not in conversations:
        return jsonify({"error": "Conversation not found"}), 404
    
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "Message is required"}), 400
    
    user_message = data['message']
    enable_search = data.get('enable_search', True)
    evaluate_responses = data.get('evaluate_responses', True)
    
    # Add user message to conversation
    conversation = conversations[conversation_id]
    user_msg = {"role": "user", "content": user_message}
    conversation["messages"].append(user_msg)
    
    # Create a task to process the message
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "conversation_id": conversation_id,
        "status": "processing",
        "created_at": datetime.now().isoformat()
    }
    
    # Start background processing
    thread = threading.Thread(
        target=process_message_dual_models,
        args=(task_id, conversation_id, enable_search, evaluate_responses)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "task_id": task_id,
        "status": "processing"
    })

@app.route('/api/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Check the status of a task."""
    if task_id not in tasks:
        return jsonify({"error": "Task not found"}), 404
    
    task = tasks[task_id]
    result = {
        "status": task["status"],
        "conversation_id": task["conversation_id"]
    }
    
    # Include results if completed
    if task["status"] == "completed" and task_id in task_results:
        result["results"] = task_results[task_id]
    
    return jsonify(result)

@app.route('/api/conversation/<conversation_id>/choose', methods=['POST'])
def choose_response(conversation_id):
    """Choose which model's response to use."""
    if conversation_id not in conversations:
        return jsonify({"error": "Conversation not found"}), 404
    
    data = request.json
    if not data or 'model' not in data:
        return jsonify({"error": "Model choice is required"}), 400
    
    chosen_model = data['model']
    conversation = conversations[conversation_id]
    
    # Get the two most recent model responses
    workflow_state = conversation["workflow_state"]
    model1_data = workflow_state["model1_response"]
    model2_data = workflow_state["model2_response"]
    
    if model1_data["status"] != "completed" or model2_data["status"] != "completed":
        return jsonify({"error": "Both model responses must be completed"}), 400
    
    model1_response = model1_data["result"]["final_response"]
    model2_response = model2_data["result"]["final_response"]
    model1_name = model1_data["model"]
    model2_name = model2_data["model"]
    
    # Record the chosen and rejected responses
    if chosen_model == model1_name:
        chosen_response = {
            "model": model1_name,
            "content": model1_response,
            "timestamp": datetime.now().isoformat()
        }
        rejected_response = {
            "model": model2_name,
            "content": model2_response,
            "timestamp": datetime.now().isoformat()
        }
        # Add the chosen response to the conversation
        assistant_msg = {
            "role": "assistant",
            "content": model1_response,
            "model": model1_name
        }
    else:
        chosen_response = {
            "model": model2_name,
            "content": model2_response,
            "timestamp": datetime.now().isoformat()
        }
        rejected_response = {
            "model": model1_name,
            "content": model1_response,
            "timestamp": datetime.now().isoformat()
        }
        # Add the chosen response to the conversation
        assistant_msg = {
            "role": "assistant",
            "content": model2_response,
            "model": model2_name
        }
    
    conversation["chosen_responses"].append(chosen_response)
    conversation["rejected_responses"].append(rejected_response)
    conversation["messages"].append(assistant_msg)
    
    return jsonify({
        "status": "success",
        "conversation": conversation
    })

@app.route('/api/conversation/<conversation_id>/save', methods=['POST'])
def save_conversation(conversation_id):
    """Save the conversation to a file."""
    if conversation_id not in conversations:
        return jsonify({"error": "Conversation not found"}), 404
    
    conversation = conversations[conversation_id]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_{timestamp}.json"
    filepath = os.path.join(SAVED_CONVS_DIR, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            "status": "success",
            "filename": filename,
            "filepath": filepath
        })
    except Exception as e:
        logger.error(f"Error saving conversation: {str(e)}")
        return jsonify({"error": f"Error saving: {str(e)}"}), 500

@app.route('/api/conversation/<conversation_id>/clear', methods=['POST'])
def clear_conversation(conversation_id):
    """Clear the conversation and start fresh."""
    if conversation_id not in conversations:
        return jsonify({"error": "Conversation not found"}), 404
    
    # Get the current models
    current_models = [
        conversations[conversation_id]["workflow_state"]["model1_response"]["model"],
        conversations[conversation_id]["workflow_state"]["model2_response"]["model"]
    ]
    
    # Reset the conversation
    conversations[conversation_id] = init_conversation()
    
    # Restore the models
    conversations[conversation_id]["workflow_state"]["model1_response"]["model"] = current_models[0]
    conversations[conversation_id]["workflow_state"]["model2_response"]["model"] = current_models[1]
    
    return jsonify({
        "status": "success",
        "conversation_id": conversation_id
    })

@app.route('/api/saved_conversations', methods=['GET'])
def list_saved_conversations():
    """List all saved conversations."""
    files = [f for f in os.listdir(SAVED_CONVS_DIR) if f.endswith('.json')]
    files.sort(reverse=True)  # Sort by newest first
    
    conversations_list = []
    for filename in files:
        filepath = os.path.join(SAVED_CONVS_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract basic info
            message_count = len(data.get("messages", []))
            first_message = data.get("messages", [{}])[0].get("content", "") if message_count > 0 else ""
            timestamp = filename.split("_")[1].split(".")[0]  # Extract timestamp from filename
            
            conversations_list.append({
                "filename": filename,
                "timestamp": timestamp,
                "message_count": message_count,
                "preview": first_message[:100] + "..." if len(first_message) > 100 else first_message
            })
        except Exception as e:
            logger.error(f"Error reading saved conversation {filename}: {str(e)}")
    
    return jsonify(conversations_list)

@app.route('/api/saved_conversations/<filename>', methods=['GET'])
def download_saved_conversation(filename):
    """Download a saved conversation file."""
    filepath = os.path.join(SAVED_CONVS_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    return send_file(filepath, as_attachment=True)

def generate_model_response(conversation, model_type, agent_prompt,timeout=600):
    """Generate a response from the specified model."""
    try:
        if model_type == 'together':
            return get_together_completion(agent_prompt)
        elif model_type == 'claude':
            return get_claude_completion(agent_prompt)
        elif model_type == 'groq':
            return get_groq_completion(agent_prompt)
        elif model_type == 'openai':
            return get_openai_completion(agent_prompt, model="o3-mini")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        logger.error(f"Error generating {model_type} response: {str(e)}")
        return None

def process_message_dual_models(task_id, conversation_id, enable_search, evaluate_responses):
    """Process a user message through the entire workflow with dual model responses."""
    conversation = conversations[conversation_id]
    previous_data = {
        "preferences": conversation.get("preferences", {}),
        "num_matches": conversation.get("num_matches", 100)
    }
    
    try:
        # Reset workflow state
        conversation["workflow_state"] = {
            "ner": {"status": "pending", "result": None},
            "search_decision": {"status": "pending", "result": None},
            "search": {"status": "pending", "result": None},
            "model1_response": {"status": "pending", "result": None, "model": conversation["workflow_state"]["model1_response"]["model"]},
            "model2_response": {"status": "pending", "result": None, "model": conversation["workflow_state"]["model2_response"]["model"]},
            "critique": {"status": "pending", "result": None},
            "regeneration": {"status": "pending", "result": None}
        }
        
        # Create a simple conversation (role + content only)
        simple_conversation = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in conversation["messages"]
        ]
        
        extracted_preferences = {}
        search_record = None
        show_results_to_actor = False
        
        # STEP 1: Extract NER
        conversation["workflow_state"]["ner"]["status"] = "processing"
        if enable_search:
            extracted_preferences = extract_ner_from_conversation(simple_conversation)
            conversation["workflow_state"]["ner"]["status"] = "completed"
            conversation["workflow_state"]["ner"]["result"] = extracted_preferences
            
            # Update conversation preferences
            conversation["preferences"] = extracted_preferences
            
            # STEP 2: Decide whether to trigger a search
            conversation["workflow_state"]["search_decision"]["status"] = "processing"
            preferences_changed = (extracted_preferences != previous_data["preferences"])
            many_previous_results = (previous_data["num_matches"] > 10)
            
            should_trigger_search = many_previous_results or (preferences_changed and not many_previous_results)
            conversation["workflow_state"]["search_decision"]["status"] = "completed"
            conversation["workflow_state"]["search_decision"]["result"] = {
                "should_trigger_search": should_trigger_search,
                "reason": "Many previous results" if many_previous_results else "Preferences changed"
            }
            
            # STEP 3: Perform search if needed
            if should_trigger_search:
                conversation["workflow_state"]["search"]["status"] = "processing"
                search_call_result = process_search_call(extracted_preferences)
                
                if search_call_result:
                    search_record = process_search_simulation_openai(search_call_result, simple_conversation)
                    if search_record and "num_matches" in search_record:
                        conversation["num_matches"] = search_record["num_matches"]
                
                conversation["workflow_state"]["search"]["status"] = "completed"
                conversation["workflow_state"]["search"]["result"] = search_record
        
        # STEP 4: Determine whether to show search results
        search_text = ""
        if search_record:
            if search_record.get("num_matches", 100) <= 50:
                show_results_to_actor = True
                search_text = search_record.get("results", "")
                search_record["show_results_to_actor"] = True
            else:
                search_record["show_results_to_actor"] = False
        
        # STEP 5: Prepare the actor prompt
        agent_template = read_prompt_template("actor.md")
        if not agent_template:
            raise Exception("Failed to read actor.md template")
            
        if show_results_to_actor:
            agent_prompt = (
                agent_template
                .replace("{conv}", json.dumps(simple_conversation, ensure_ascii=False, indent=2))
                .replace("{search}", search_text)
                .replace("{num_matches}", str(conversation["num_matches"]))
            )
        else:
            agent_prompt = (
                agent_template
                .replace("{conv}", json.dumps(simple_conversation, ensure_ascii=False, indent=2))
                .replace("{search}", "")
                .replace("{num_matches}", "")
            )
        
        # STEP 6: Generate responses from both models in parallel
        model1 = conversation["workflow_state"]["model1_response"]["model"]
        model2 = conversation["workflow_state"]["model2_response"]["model"]

        conversation["workflow_state"]["model1_response"]["status"] = "processing"
        conversation["workflow_state"]["model2_response"]["status"] = "processing"

        # Create a dictionary to store futures
        futures = {}
        model_futures = {}  # Map futures to their model names
        final_response1 = None
        final_response2 = None

        # Use ThreadPoolExecutor to run the model calls in parallel with timeouts
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            futures[model1] = executor.submit(generate_model_response, conversation, model1, agent_prompt, timeout=90)
            futures[model2] = executor.submit(generate_model_response, conversation, model2, agent_prompt, timeout=90)
            
            # Create reverse mapping from future to model
            model_futures = {future: model for model, future in futures.items()}
            
            # Process results as they complete
            model1_response = None
            model2_response = None
            
            try:
                # Wait for both futures to complete, but with a maximum wait time
                # This prevents one slow model from blocking the entire process
                for completed_future in concurrent.futures.as_completed(list(futures.values()), timeout=120):
                    try:
                        result = completed_future.result()
                        model = model_futures[completed_future]
                        
                        if model == model1:
                            model1_response = result
                        elif model == model2:
                            model2_response = result
                    except Exception as e:
                        model = model_futures[completed_future]
                        logger.error(f"Error getting result from {model}: {str(e)}")
                        if model == model1:
                            conversation["workflow_state"]["model1_response"]["status"] = "error"
                            conversation["workflow_state"]["model1_response"]["error"] = str(e)
                        elif model == model2:
                            conversation["workflow_state"]["model2_response"]["status"] = "error"
                            conversation["workflow_state"]["model2_response"]["error"] = str(e)
            except concurrent.futures.TimeoutError:
                logger.error("Parallel model execution timed out")
                # Mark any incomplete futures as timed out
                for model, future in futures.items():
                    if not future.done():
                        conversation["workflow_state"][f"{'model1' if model == model1 else 'model2'}_response"]["status"] = "error"
                        conversation["workflow_state"][f"{'model1' if model == model1 else 'model2'}_response"]["error"] = f"{model} model timed out"
        
        # Process model1 response
        if model1_response:
            thinking1, response_after_thinking1 = extract_thinking(model1_response)
            final_response1, _ = extract_function_calls(response_after_thinking1)
            if not final_response1 or not final_response1.strip():
                final_response1 = response_after_thinking1
                
            conversation["workflow_state"]["model1_response"]["status"] = "completed"
            conversation["workflow_state"]["model1_response"]["result"] = {
                "thinking": thinking1,
                "response_after_thinking": response_after_thinking1,
                "final_response": final_response1
            }
        else:
            if conversation["workflow_state"]["model1_response"]["status"] != "error":
                conversation["workflow_state"]["model1_response"]["status"] = "error"
                conversation["workflow_state"]["model1_response"]["error"] = f"Failed to generate {model1} response"
        
        # Process model2 response
        if model2_response:
            thinking2, response_after_thinking2 = extract_thinking(model2_response)
            final_response2, _ = extract_function_calls(response_after_thinking2)
            if not final_response2 or not final_response2.strip():
                final_response2 = response_after_thinking2
                
            conversation["workflow_state"]["model2_response"]["status"] = "completed"
            conversation["workflow_state"]["model2_response"]["result"] = {
                "thinking": thinking2,
                "response_after_thinking": response_after_thinking2,
                "final_response": final_response2
            }
        else:
            if conversation["workflow_state"]["model2_response"]["status"] != "error":
                conversation["workflow_state"]["model2_response"]["status"] = "error"
                conversation["workflow_state"]["model2_response"]["error"] = f"Failed to generate {model2} response"
        
        # STEP 7: Critique both responses if evaluation is enabled
        if evaluate_responses:
            conversation["workflow_state"]["critique"]["status"] = "processing"
            critiques = {}
            
            if final_response1:
                original_prompt = agent_template.replace("{conv}", "").replace("{search}", "").replace("{num_matches}", "").strip()
                try:
                    critique1 = get_critic_evaluation(
                        original_prompt,
                        simple_conversation,
                        search_record,
                        final_response1
                    )
                    critiques[model1] = critique1
                except Exception as e:
                    logger.error(f"Error generating critique for {model1}: {str(e)}")
                    critiques[model1] = {"error": str(e)}
            
            if final_response2:
                original_prompt = agent_template.replace("{conv}", "").replace("{search}", "").replace("{num_matches}", "").strip()
                try:
                    critique2 = get_critic_evaluation(
                        original_prompt,
                        simple_conversation,
                        search_record,
                        final_response2
                    )
                    critiques[model2] = critique2
                except Exception as e:
                    logger.error(f"Error generating critique for {model2}: {str(e)}")
                    critiques[model2] = {"error": str(e)}
            
            conversation["workflow_state"]["critique"]["status"] = "completed"
            conversation["workflow_state"]["critique"]["result"] = critiques
        
        # Mark task as completed - even if some models failed, we want to show what we have
        tasks[task_id]["status"] = "completed"
        task_results[task_id] = {
            "conversation_id": conversation_id,
            "workflow_state": conversation["workflow_state"]
        }
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        # Update task status
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"] = str(e)
        # Update workflow state
        for step in conversation["workflow_state"]:
            if conversation["workflow_state"][step]["status"] == "processing":
                conversation["workflow_state"][step]["status"] = "error"
                conversation["workflow_state"][step]["error"] = str(e)

def generate_model_response(conversation, model_type, agent_prompt, timeout=60):
    """Generate a response from the specified model with timeout."""
    try:
        # Create a Future with the model call
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_call_model_api, model_type, agent_prompt)
            
            try:
                # Wait for the result with a timeout
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                logger.error(f"Model call to {model_type} timed out after {timeout} seconds")
                return f"The {model_type} model timed out. Please try again or choose a different model."
    except Exception as e:
        logger.error(f"Error generating {model_type} response: {str(e)}")
        return None

def _call_model_api(model_type, prompt):
    """Internal function to call the appropriate model API."""
    if model_type == 'together':
        return get_together_completion(prompt)
    elif model_type == 'claude':
        return get_claude_completion(prompt)
    elif model_type == 'groq':
        return get_groq_completion(prompt)
    elif model_type == 'openai':
        return get_openai_completion(prompt, model="o3-mini")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs("prompts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5100)