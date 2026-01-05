// DOM Elements
const ticketText = document.getElementById('ticketText');
const thresholdSlider = document.getElementById('threshold');
const thresholdValue = document.getElementById('thresholdValue');
const classifyBtn = document.getElementById('classifyBtn');
const resultsSection = document.getElementById('resultsSection');
const loading = document.getElementById('loading');
const errorMessage = document.getElementById('errorMessage');
const exampleBtns = document.querySelectorAll('.example-btn');

// Update threshold value display
thresholdSlider.addEventListener('input', (e) => {
    thresholdValue.textContent = parseFloat(e.target.value).toFixed(2);
});

// Example button clicks
exampleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        ticketText.value = btn.dataset.text;
        ticketText.focus();
    });
});

// Classify button click
classifyBtn.addEventListener('click', async () => {
    const text = ticketText.value.trim();
    const threshold = parseFloat(thresholdSlider.value);
    
    if (!text) {
        showError('Please enter some text');
        return;
    }
    
    // Show loading
    loading.style.display = 'block';
    resultsSection.style.display = 'none';
    errorMessage.style.display = 'none';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text, threshold })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Prediction failed');
        }
        
        const result = await response.json();
        displayResults(result);
        
    } catch (error) {
        showError(error.message);
    } finally {
        loading.style.display = 'none';
    }
});

// Display results
function displayResults(result) {
    // Show results section
    resultsSection.style.display = 'block';
    
    // Update action badge
    const actionBadge = document.getElementById('actionBadge');
    actionBadge.textContent = result.action;
    actionBadge.className = `badge ${result.action_class}`;
    
    // Update department name
    document.getElementById('departmentName').textContent = result.predicted_label;
    
    // Update confidence bar
    const confidenceFill = document.getElementById('confidenceFill');
    const confidencePercent = (result.confidence * 100).toFixed(1);
    confidenceFill.style.width = confidencePercent + '%';
    
    // Update confidence text
    document.getElementById('confidenceValue').textContent = confidencePercent + '%';
    
    // Update reason
    document.getElementById('actionReason').textContent = result.reason;
    
    // Display all probabilities
    const probabilitiesList = document.getElementById('probabilitiesList');
    probabilitiesList.innerHTML = '';
    
    result.all_probabilities.forEach(([dept, prob]) => {
        const probPercent = (prob * 100).toFixed(1);
        
        const probItem = document.createElement('div');
        probItem.className = 'prob-item';
        
        probItem.innerHTML = `
            <div class="prob-label">${dept}</div>
            <div class="prob-bar-container">
                <div class="prob-bar" style="width: ${probPercent}%"></div>
            </div>
            <div class="prob-value">${probPercent}%</div>
        `;
        
        probabilitiesList.appendChild(probItem);
    });
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    setTimeout(() => {
        errorMessage.style.display = 'none';
    }, 5000);
}

// Allow Enter key to submit (Ctrl+Enter for new line)
ticketText.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        classifyBtn.click();
    }
});
