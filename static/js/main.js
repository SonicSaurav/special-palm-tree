// State management
let state = {
    conversationId: null,
    taskId: null,
    polling: false,
    pollingInterval: null,
    messages: [],
    waitingForChoice: false,
    model1: 'together',
    model2: 'claude'
};

// DOM elements
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const model1Select = document.getElementById('model1-select');
const model2Select = document.getElementById('model2-select');
const searchToggle = document.getElementById('search-toggle');
const critiqueToggle = document.getElementById('critique-toggle');
const saveButton = document.getElementById('save-button');
const clearButton = document.getElementById('clear-button');

// Model selection buttons
const chooseModel1Btn = document.getElementById('choose-model1-btn');
const chooseModel2Btn = document.getElementById('choose-model2-btn');

// Model headers
const model1Name = document.getElementById('model1-name');
const model2Name = document.getElementById('model2-name');

// Modal elements
const saveDialog = document.getElementById('save-dialog');
const clearDialog = document.getElementById('clear-dialog');
const closeButtons = document.querySelectorAll('.close');
const downloadSaveBtn = document.getElementById('download-save-btn');
const closeSaveBtn = document.getElementById('close-save-btn');
const confirmClearBtn = document.getElementById('confirm-clear-btn');
const cancelClearBtn = document.getElementById('cancel-clear-btn');
const savedFilename = document.getElementById('saved-filename');

// Templates
const userMessageTemplate = document.getElementById('user-message-template');
const assistantMessageTemplate = document.getElementById('assistant-message-template');
const modelComparisonTemplate = document.getElementById('model-comparison-template');

// Status elements
const workflowStatusElements = {
    ner: document.getElementById('ner-status'),
    search_decision: document.getElementById('search-decision-status'),
    search: document.getElementById('search-status'),
    model1: document.getElementById('model1-status'),
    model2: document.getElementById('model2-status'),
    critique: document.getElementById('critique-status')
};

// Content elements
const workflowContentElements = {
    ner: document.getElementById('ner-content').querySelector('pre'),
    search_decision: document.getElementById('search-decision-content').querySelector('pre'),
    search: document.getElementById('search-content').querySelector('pre'),
    model1_thinking: document.getElementById('model1-thinking-content'),
    model1_response: document.getElementById('model1-response-content'),
    model2_thinking: document.getElementById('model2-thinking-content'),
    model2_response: document.getElementById('model2-response-content'),
    model1_critique: document.querySelector('#model1-critique pre'),
    model2_critique: document.querySelector('#model2-critique pre')
};

// Initialize the application
async function initApp() {
    try {
        // Set event listeners for model selection
        model1Select.addEventListener('change', () => {
            state.model1 = model1Select.value;
            updateModelDisplay();
        });
        
        model2Select.addEventListener('change', () => {
            state.model2 = model2Select.value;
            updateModelDisplay();
        });
        
        // Set initial model display
        updateModelDisplay();
        
        // Set event listeners for model choice buttons
        chooseModel1Btn.addEventListener('click', () => {
            if (state.waitingForChoice) {
                chooseModelResponse(state.model1);
            }
        });
        
        chooseModel2Btn.addEventListener('click', () => {
            if (state.waitingForChoice) {
                chooseModelResponse(state.model2);
            }
        });
        
        // Set event listeners for tab buttons
        const tabButtons = document.querySelectorAll('.tab-btn');
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const target = button.dataset.target;
                
                // Remove active class from all tabs and contents
                document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
                
                // Add active class to clicked tab and corresponding content
                button.classList.add('active');
                document.getElementById(target).classList.add('active');
            });
        });
        
        // Modal event listeners
        saveButton.addEventListener('click', saveConversation);
        clearButton.addEventListener('click', () => {
            clearDialog.classList.add('active');
        });
        
        closeButtons.forEach(button => {
            button.addEventListener('click', () => {
                saveDialog.classList.remove('active');
                clearDialog.classList.remove('active');
            });
        });
        
        downloadSaveBtn.addEventListener('click', downloadSavedConversation);
        closeSaveBtn.addEventListener('click', () => {
            saveDialog.classList.remove('active');
        });
        
        confirmClearBtn.addEventListener('click', clearConversation);
        cancelClearBtn.addEventListener('click', () => {
            clearDialog.classList.remove('active');
        });
        
        // Create a new conversation
        const response = await fetch('/api/conversation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model1: state.model1,
                model2: state.model2
            })
        });
        
        if (!response.ok) {
            throw new Error(`Failed to create conversation: ${response.statusText}`);
        }
        
        const data = await response.json();
        state.conversationId = data.conversation_id;
        console.log(`Conversation created with ID: ${state.conversationId}`);
        
        // Set event listeners for message input
        sendButton.addEventListener('click', sendMessage);
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        // Ready to use
        userInput.focus();
        
    } catch (error) {
        console.error('Error initializing app:', error);
        alert('Failed to initialize the application. Please refresh the page.');
    }
}

// Update model display based on selections
function updateModelDisplay() {
    // Update model headers
    model1Name.textContent = `${getModelDisplayName(state.model1)} Response`;
    model2Name.textContent = `${getModelDisplayName(state.model2)} Response`;
    
    // Update model response section color classes
    const model1Section = model1Name.closest('.model-response');
    const model2Section = model2Name.closest('.model-response');
    
    // Remove existing model classes
    model1Section.classList.remove('model-together', 'model-claude', 'model-groq', 'model-openai');
    model2Section.classList.remove('model-together', 'model-claude', 'model-groq', 'model-openai');
    
    // Add appropriate model classes
    model1Section.classList.add(`model-${state.model1}`);
    model2Section.classList.add(`model-${state.model2}`);
}

// Get display name for model
function getModelDisplayName(model) {
    switch (model) {
        case 'together':
            return 'Together (DeepSeek-R1)';
        case 'claude':
            return 'Claude (3-7-sonnet)';
        case 'groq':
            return 'Groq (DeepSeek-R1-Distill)';
        case 'openai':
            return 'OpenAI (o3-mini)';
        default:
            return model.charAt(0).toUpperCase() + model.slice(1);
    }
}

// Send a user message
async function sendMessage() {
    // Don't send if waiting for model choice
    if (state.waitingForChoice) {
        alert('Please choose one of the model responses to continue.');
        return;
    }
    
    const messageText = userInput.value.trim();
    if (!messageText) return;
    
    // Clear input
    userInput.value = '';
    
    // Add to UI
    addUserMessage(messageText);
    
    // Update state
    state.messages.push({
        role: 'user',
        content: messageText
    });
    
    // Reset workflow status
    resetWorkflowStatus();
    
    try {
        // Send message to API
        const response = await fetch(`/api/conversation/${state.conversationId}/message`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: messageText,
                enable_search: searchToggle.checked,
                evaluate_responses: critiqueToggle.checked
            })
        });
        
        if (!response.ok) {
            throw new Error(`Failed to send message: ${response.statusText}`);
        }
        
        const data = await response.json();
        state.taskId = data.task_id;
        
        // Start polling for task updates
        startPolling();
        
    } catch (error) {
        console.error('Error sending message:', error);
        alert('Failed to send message. Please try again.');
    }
}

// Start polling for task updates
function startPolling() {
    if (state.polling) return;
    
    state.polling = true;
    state.pollingStartTime = Date.now();
    pollTaskStatus();
    
    // Set a timeout to stop polling after 2 minutes
    setTimeout(() => {
        if (state.polling && (Date.now() - state.pollingStartTime) > 120000) {
            stopPolling();
            
            // Show error message
            alert('The request timed out. Some models may take longer to respond. You can try again or switch to faster models.');
            
            // Enable the input field again
            sendButton.disabled = false;
            userInput.disabled = false;
        }
    }, 720000);
}
// Stop polling
function stopPolling() {
    state.polling = false;
    if (state.pollingInterval) {
        clearTimeout(state.pollingInterval);
        state.pollingInterval = null;
    }
}

// Poll for task status
async function pollTaskStatus() {
    if (!state.polling || !state.taskId) return;
    
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout for fetch
        
        const response = await fetch(`/api/task/${state.taskId}`, {
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            throw new Error(`Failed to poll task: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Update workflow status
        if (data.results && data.results.workflow_state) {
            updateWorkflowStatus(data.results.workflow_state);
            
            // Check if any model has completed - if so, we can show partial results
            const model1Status = data.results.workflow_state.model1_response.status;
            const model2Status = data.results.workflow_state.model2_response.status;
            
            // If at least one model has completed or errored, we can stop waiting
            // if the polling has been going on for more than 60 seconds
            const pollingDuration = Date.now() - state.pollingStartTime;
            if (pollingDuration > 60000 && 
                ((model1Status === 'completed' || model1Status === 'error') || 
                 (model2Status === 'completed' || model2Status === 'error'))) {
                
                // Show whatever results we have so far
                displayPartialResults(data.results.workflow_state);
                stopPolling();
                return;
            }
        }
        
        if (data.status === 'completed') {
            // Task completed, display model comparison and stop polling
            displayModelComparison(data.results.workflow_state);
            stopPolling();
        } else if (data.status === 'error') {
            console.error('Task error:', data.error);
            alert(`Error processing message: ${data.error || 'Unknown error'}`);
            stopPolling();
        } else {
            // Still processing, continue polling after a delay
            const backoffDelay = Math.min(2000 * Math.pow(1.5, Math.floor((Date.now() - state.pollingStartTime) / 10000)), 10000);
            state.pollingInterval = setTimeout(pollTaskStatus, backoffDelay);
        }
        
    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('Fetch aborted due to timeout');
            // Try again with exponential backoff
            const backoffDelay = Math.min(5000 * Math.pow(1.5, Math.floor((Date.now() - state.pollingStartTime) / 10000)), 15000);
            state.pollingInterval = setTimeout(pollTaskStatus, backoffDelay);
        } else {
            console.error('Error polling task status:', error);
            state.pollingInterval = setTimeout(pollTaskStatus, 5000);
        }
    }
}

// Display partial results when at least one model has responded
function displayPartialResults(workflowState) {
    const model1Data = workflowState.model1_response;
    const model2Data = workflowState.model2_response;
    
    let completedModels = 0;
    
    // Check if model1 is completed
    if (model1Data.status === 'completed' && model1Data.result && model1Data.result.final_response) {
        completedModels++;
        chooseModel1Btn.style.display = 'block';
    } else {
        // Show error or timeout message for model1
        workflowContentElements.model1_response.textContent = 
            "This model is taking longer than expected. You can choose the other model's response or wait for both to complete.";
    }
    
    // Check if model2 is completed
    if (model2Data.status === 'completed' && model2Data.result && model2Data.result.final_response) {
        completedModels++;
        chooseModel2Btn.style.display = 'block';
    } else {
        // Show error or timeout message for model2
        workflowContentElements.model2_response.textContent = 
            "This model is taking longer than expected. You can choose the other model's response or wait for both to complete.";
    }
    
    // If at least one model completed, enable choosing
    if (completedModels > 0) {
        state.waitingForChoice = true;
        sendButton.disabled = true;
        
        // Show a message to the user
        const message = document.createElement('div');
        message.className = 'system-message';
        message.textContent = `${completedModels === 1 ? 'One model has' : 'Both models have'} responded. Please select a response to continue.`;
        chatMessages.appendChild(message);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    } else {
        // No models completed, allow user to try again
        alert('Both models took too long to respond. Please try again or switch to faster models.');
        sendButton.disabled = false;
    }
}

// Display model comparison UI in chat
function displayModelComparison(workflowState) {
    const model1Data = workflowState.model1_response;
    const model2Data = workflowState.model2_response;
    
    // If either model failed, just use the successful one
    if (model1Data.status === 'error' && model2Data.status === 'completed') {
        addAssistantMessage({
            role: 'assistant',
            content: model2Data.result.final_response,
            model: model2Data.model
        });
        return;
    } else if (model2Data.status === 'error' && model1Data.status === 'completed') {
        addAssistantMessage({
            role: 'assistant',
            content: model1Data.result.final_response,
            model: model1Data.model
        });
        return;
    } else if (model1Data.status === 'error' && model2Data.status === 'error') {
        alert('Both models failed to generate responses. Please try again.');
        return;
    }
    
    // Show the choice buttons
    chooseModel1Btn.style.display = 'block';
    chooseModel2Btn.style.display = 'block';
    
    // Set waiting for choice flag
    state.waitingForChoice = true;
    
    // Disable send button until choice is made
    sendButton.disabled = true;
}

// Choose a model response
async function chooseModelResponse(modelName) {
    try {
        const response = await fetch(`/api/conversation/${state.conversationId}/choose`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: modelName
            })
        });
        
        if (!response.ok) {
            throw new Error(`Failed to choose response: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Get the chosen response from the response
        const messages = data.conversation.messages;
        const assistantMessage = messages[messages.length - 1];
        
        // Add the chosen response to the chat
        addAssistantMessage(assistantMessage);
        
        // Hide the choice buttons
        chooseModel1Btn.style.display = 'none';
        chooseModel2Btn.style.display = 'none';
        
        // Reset waiting for choice flag
        state.waitingForChoice = false;
        
        // Enable send button
        sendButton.disabled = false;
        
    } catch (error) {
        console.error('Error choosing response:', error);
        alert('Failed to choose response. Please try again.');
    }
}

// Save the current conversation
async function saveConversation() {
    try {
        const response = await fetch(`/api/conversation/${state.conversationId}/save`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`Failed to save conversation: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Show the save dialog
        savedFilename.textContent = data.filename;
        saveDialog.classList.add('active');
        
    } catch (error) {
        console.error('Error saving conversation:', error);
        alert('Failed to save conversation. Please try again.');
    }
}

// Download a saved conversation
function downloadSavedConversation() {
    const filename = savedFilename.textContent;
    if (!filename) return;
    
    window.open(`/api/saved_conversations/${filename}`, '_blank');
}

// Clear the current conversation
async function clearConversation() {
    try {
        const response = await fetch(`/api/conversation/${state.conversationId}/clear`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`Failed to clear conversation: ${response.statusText}`);
        }
        
        // Clear the UI
        chatMessages.innerHTML = '';
        resetWorkflowStatus();
        
        // Hide dialogs
        clearDialog.classList.remove('active');
        
        // Reset waiting for choice flag
        state.waitingForChoice = false;
        
        // Enable send button
        sendButton.disabled = false;
        
        // Hide choice buttons
        chooseModel1Btn.style.display = 'none';
        chooseModel2Btn.style.display = 'none';
        
        // Update state
        state.messages = [];
        
    } catch (error) {
        console.error('Error clearing conversation:', error);
        alert('Failed to clear conversation. Please try again.');
    }
}

// Add a user message to the UI
function addUserMessage(content) {
    const messageNode = userMessageTemplate.content.cloneNode(true);
    const messageContent = messageNode.querySelector('.message-content');
    messageContent.textContent = content;
    
    chatMessages.appendChild(messageNode);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Add an assistant message to the UI
function addAssistantMessage(message) {
    const messageNode = assistantMessageTemplate.content.cloneNode(true);
    const messageContent = messageNode.querySelector('.message-content');
    messageContent.textContent = message.content;
    
    // Add model name if available
    const modelName = messageNode.querySelector('.model-name');
    if (message.model) {
        modelName.textContent = getModelDisplayName(message.model);
        
        // Add model-specific class
        messageNode.querySelector('.assistant-message').classList.add(`model-${message.model}`);
    } else {
        modelName.textContent = 'Unknown';
    }
    
    // Add score if available
    const scoreValue = messageNode.querySelector('.score-value');
    if (message.critique && message.critique.total_score !== undefined) {
        scoreValue.textContent = message.critique.total_score.toFixed(1) + '/10';
        
        // Color based on score
        if (message.critique.total_score < 7) {
            scoreValue.style.color = 'var(--error-color)';
        } else if (message.critique.total_score < 8.5) {
            scoreValue.style.color = 'var(--warning-color)';
        } else {
            scoreValue.style.color = 'var(--success-color)';
        }
    } else {
        scoreValue.parentElement.style.display = 'none';
    }
    
    chatMessages.appendChild(messageNode);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Reset workflow status UI
function resetWorkflowStatus() {
    for (const key in workflowStatusElements) {
        workflowStatusElements[key].textContent = 'Waiting for input';
        workflowStatusElements[key].className = 'status-indicator status-pending';
    }
    
    for (const key in workflowContentElements) {
        workflowContentElements[key].textContent = 'No data yet';
    }
    
    // Hide choice buttons
    chooseModel1Btn.style.display = 'none';
    chooseModel2Btn.style.display = 'none';
}

// Update workflow status UI based on current state
function updateWorkflowStatus(workflowState) {
    // Update NER
    updateStepStatus('ner', workflowState.ner);
    if (workflowState.ner.result) {
        workflowContentElements.ner.textContent = JSON.stringify(workflowState.ner.result, null, 2);
    }
    
    // Update Search Decision
    updateStepStatus('search_decision', workflowState.search_decision);
    if (workflowState.search_decision.result) {
        workflowContentElements.search_decision.textContent = JSON.stringify(workflowState.search_decision.result, null, 2);
    }
    
    // Update Search
    updateStepStatus('search', workflowState.search);
    if (workflowState.search.result) {
        workflowContentElements.search.textContent = workflowState.search.result.results || 'No search results';
    }
    
    // Update Model 1 Response
    updateStepStatus('model1', workflowState.model1_response);
    if (workflowState.model1_response.result) {
        if (workflowState.model1_response.result.thinking) {
            workflowContentElements.model1_thinking.textContent = workflowState.model1_response.result.thinking;
        }
        
        if (workflowState.model1_response.result.final_response) {
            workflowContentElements.model1_response.textContent = workflowState.model1_response.result.final_response;
        }
    }
    
    // Update Model 2 Response
    updateStepStatus('model2', workflowState.model2_response);
    if (workflowState.model2_response.result) {
        if (workflowState.model2_response.result.thinking) {
            workflowContentElements.model2_thinking.textContent = workflowState.model2_response.result.thinking;
        }
        
        if (workflowState.model2_response.result.final_response) {
            workflowContentElements.model2_response.textContent = workflowState.model2_response.result.final_response;
        }
    }
    
    // Update Critique
    updateStepStatus('critique', workflowState.critique);
    if (workflowState.critique.result) {
        const model1Name = workflowState.model1_response.model;
        const model2Name = workflowState.model2_response.model;
        
        if (workflowState.critique.result[model1Name]) {
            workflowContentElements.model1_critique.textContent = JSON.stringify(workflowState.critique.result[model1Name], null, 2);
        }
        
        if (workflowState.critique.result[model2Name]) {
            workflowContentElements.model2_critique.textContent = JSON.stringify(workflowState.critique.result[model2Name], null, 2);
        }
    }
}

// Update a single workflow step status
function updateStepStatus(step, stepState) {
    const statusElement = workflowStatusElements[step];
    
    if (!statusElement) return;
    
    statusElement.className = 'status-indicator';
    
    switch (stepState.status) {
        case 'pending':
            statusElement.textContent = 'Pending';
            statusElement.classList.add('status-pending');
            break;
        case 'processing':
            statusElement.textContent = 'Processing...';
            statusElement.classList.add('status-processing');
            break;
        case 'completed':
            statusElement.textContent = 'Completed';
            statusElement.classList.add('status-completed');
            break;
        case 'error':
            statusElement.textContent = 'Error';
            statusElement.classList.add('status-error');
            break;
        default:
            statusElement.textContent = stepState.status;
            statusElement.classList.add('status-pending');
    }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', initApp);