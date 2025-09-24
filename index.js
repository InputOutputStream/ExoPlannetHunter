// Global variables
let currentMode = 'novice';
let modelConfig = {
    n_estimators: 200,
    learning_rate: 0.1,
    max_depth: 8,
    min_samples_split: 5
};

//mock model metrics
let modelMetrics = {
    accuracy: 0.968,
    precision: 0.942,
    recall: 0.981,
    f1score: 0.961
};
let trainingData = [];
let isTraining = false;

// Machine Learning Model Simulation
class ExoplanetML {
    constructor() {
        this.weights = {
            period: 0.25,
            duration: 0.20,
            depth: 0.30,
            stellar_radius: 0.15,
            stellar_temp: 0.05,
            impact: 0.05
        };
        this.threshold_confirmed = 0.8;
        this.threshold_candidate = 0.5;
    }

    preprocess(features) {
        // Normalize features
        const normalized = {};
        normalized.period = Math.log(features.period || 10) / 10;
        normalized.duration = (features.duration || 5) / 24;
        normalized.depth = Math.log(features.depth || 100) / 15;
        normalized.stellar_radius = (features.stellar_radius || 1) / 5;
        normalized.stellar_temp = ((features.stellar_temp || 5778) - 3000) / 7000;
        normalized.impact = features.impact || 0.5;
        return normalized;
    }

    predict(features) {
        const norm = this.preprocess(features);
        
        // Simulate ensemble prediction
        let score = 0;
        for (let [key, weight] of Object.entries(this.weights)) {
            if (norm[key] !== undefined) {
                score += norm[key] * weight;
            }
        }
        
        // Add some realistic noise
        score += (Math.random() - 0.5) * 0.1;
        score = Math.max(0, Math.min(1, score));
        
        // Classify
        let classification, confidence;
        if (score >= this.threshold_confirmed) {
            classification = 'CONFIRMED';
            confidence = score;
        } else if (score >= this.threshold_candidate) {
            classification = 'CANDIDATE';
            confidence = score;
        } else {
            classification = 'FALSE POSITIVE';
            confidence = 1 - score;
        }
        
        return {
            classification,
            confidence: confidence,
            score: score,
            features: norm
        };
    }

    retrain(hyperparams) {
        // Simulate retraining with new hyperparameters
        Object.assign(modelConfig, hyperparams);
        
        // Simulate performance changes
        const variation = (Math.random() - 0.5) * 0.02;
        modelMetrics.accuracy = Math.max(0.9, Math.min(0.99, 0.968 + variation));
        modelMetrics.precision = Math.max(0.9, Math.min(0.99, 0.942 + variation));
        modelMetrics.recall = Math.max(0.9, Math.min(0.99, 0.981 + variation));
        modelMetrics.f1score = Math.max(0.9, Math.min(0.99, 0.961 + variation));
        
        updateMetricsDisplay();
    }
}

const model = new ExoplanetML();

// Mode switching
function switchMode(mode) {
    currentMode = mode;
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    if (mode === 'novice') {
        document.body.classList.add('novice-mode');
        document.getElementById('tutorial').style.display = 'block';
    } else {
        document.body.classList.remove('novice-mode');
        document.getElementById('tutorial').style.display = 'none';
    }
}

// Hyperparameter updates
function updateHyperparameter(param, value) {
    modelConfig[param] = parseFloat(value);
    document.getElementById(param.replace('_', '-') + '-value').textContent = 
        param === 'learning_rate' ? parseFloat(value).toFixed(2) : value;
}

// Model retraining
function retrainModel() {
    if (isTraining) return;
    
    isTraining = true;
    const btn = document.getElementById('retrain-btn');
    const status = document.getElementById('status');
    
    btn.disabled = true;
    btn.textContent = '🔄 Training...';
    status.textContent = 'Retraining Model...';
    status.classList.add('processing');
    
    setTimeout(() => {
        model.retrain(modelConfig);
        btn.disabled = false;
        btn.textContent = '🔄 Retrain Model';
        status.textContent = 'Model Updated';
        status.classList.remove('processing');
        isTraining = false;
        
        generateROCCurve();
        generateFeatureImportance();
    }, 2000);
}

// File upload handling
function handleFileUpload(event) {
    const files = event.target.files;
    const status = document.getElementById('status');
    
    status.textContent = 'Processing files...';
    status.classList.add('processing');
    
    setTimeout(() => {
        // Simulate file processing
        const mockResults = [];
        for (let i = 0; i < Math.min(files.length * 10, 100); i++) {
            mockResults.push(generateMockData());
        }
        
        displayBatchResults(mockResults);
        status.textContent = `Processed ${files.length} file(s)`;
        status.classList.remove('processing');
    }, 1500);
}

// Drag and drop
const uploadArea = document.getElementById('upload-area');
uploadArea.ondragover = (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
};
uploadArea.ondragleave = () => uploadArea.classList.remove('dragover');
uploadArea.ondrop = (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFileUpload(e);
};

// Single prediction
function predictSingle() {
    const features = {
        period: parseFloat(document.getElementById('period').value) || null,
        duration: parseFloat(document.getElementById('duration').value) || null,
        depth: parseFloat(document.getElementById('depth').value) || null,
        stellar_radius: parseFloat(document.getElementById('stellar_radius').value) || null,
        stellar_temp: parseFloat(document.getElementById('stellar_temp').value) || null,
        impact: parseFloat(document.getElementById('impact').value) || null
    };
    
    if (!features.period && !features.duration && !features.depth) {
        alert('Please enter at least orbital period, transit duration, or transit depth.');
        return;
    }
    
    const result = model.predict(features);
    displaySingleResult(result, features);
    generateLightCurve(features);
}

// Batch prediction
function batchPredict() {
    const mockData = [];
    for (let i = 0; i < 50; i++) {
        mockData.push(generateMockData());
    }
    displayBatchResults(mockData);
}

// Generate mock data
function generateMockData() {
    const features = {
        period: Math.random() * 1000 + 1,
        duration: Math.random() * 24 + 0.5,
        depth: Math.random() * 5000 + 10,
        stellar_radius: Math.random() * 3 + 0.5,
        stellar_temp: Math.random() * 4000 + 3000,
        impact: Math.random()
    };
    
    return {
        features,
        result: model.predict(features)
    };
}

// Display single result
function displaySingleResult(result, features) {
    const resultsDiv = document.getElementById('results');
    const className = result.classification === 'CONFIRMED' ? 'confirmed' : 
                        result.classification === 'CANDIDATE' ? 'candidate' : 'false-positive';
    
    resultsDiv.innerHTML = `
        <div class="panel">
            <h3>🎯 Prediction Result</h3>
            <div class="prediction-result ${className}">
                <h2>${result.classification}</h2>
                <p>Confidence: ${(result.confidence * 100).toFixed(1)}%</p>
                <p>Model Score: ${result.score.toFixed(3)}</p>
            </div>
            <div class="metrics">
                <div class="metric"><div class="metric-value">${features.period?.toFixed(2) || 'N/A'}</div><div class="metric-label">Period (days)</div></div>
                <div class="metric"><div class="metric-value">${features.duration?.toFixed(2) || 'N/A'}</div><div class="metric-label">Duration (hrs)</div></div>
                <div class="metric"><div class="metric-value">${features.depth?.toFixed(0) || 'N/A'}</div><div class="metric-label">Depth (ppm)</div></div>
                <div class="metric"><div class="metric-value">${features.stellar_radius?.toFixed(2) || 'N/A'}</div><div class="metric-label">Stellar R☉</div></div>
            </div>
        </div>
    `;
}

// Display batch results
function displayBatchResults(results) {
    const confirmed = results.filter(r => r.result.classification === 'CONFIRMED').length;
    const candidates = results.filter(r => r.result.classification === 'CANDIDATE').length;
    const falsePos = results.filter(r => r.result.classification === 'FALSE POSITIVE').length;
    
    document.getElementById('results').innerHTML = `
        <div class="panel">
            <h3>📊 Batch Analysis Results</h3>
            <div class="metrics">
                <div class="metric"><div class="metric-value" style="color: #00ff00">${confirmed}</div><div class="metric-label">Confirmed</div></div>
                <div class="metric"><div class="metric-value" style="color: #ffa500">${candidates}</div><div class="metric-label">Candidates</div></div>
                <div class="metric"><div class="metric-value" style="color: #ff0000">${falsePos}</div><div class="metric-label">False Positives</div></div>
                <div class="metric"><div class="metric-value">${results.length}</div><div class="metric-label">Total Objects</div></div>
            </div>
            <div class="visualization" id="batch-results-chart"></div>
        </div>
    `;
    
    generateBatchChart(results);
}

// Update metrics display
function updateMetricsDisplay() {
    document.getElementById('accuracy').textContent = (modelMetrics.accuracy * 100).toFixed(1) + '%';
    document.getElementById('precision').textContent = (modelMetrics.precision * 100).toFixed(1) + '%';
    document.getElementById('recall').textContent = (modelMetrics.recall * 100).toFixed(1) + '%';
    document.getElementById('f1score').textContent = (modelMetrics.f1score * 100).toFixed(1) + '%';
}

// Generate ROC curve
function generateROCCurve() {
    const x = [];
    const y = [];
    for (let i = 0; i <= 100; i++) {
        const fpr = i / 100;
        const tpr = Math.min(1, modelMetrics.recall * (1 - Math.exp(-fpr * 5)));
        x.push(fpr);
        y.push(tpr);
    }
    
    const trace = {
        x: x,
        y: y,
        type: 'scatter',
        mode: 'lines',
        name: 'ROC Curve',
        line: { color: '#00d4ff', width: 3 }
    };
    
    const layout = {
        title: 'ROC Curve',
        xaxis: { title: 'False Positive Rate' },
        yaxis: { title: 'True Positive Rate' },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#ffffff' }
    };
    
    Plotly.newPlot('roc-curve', [trace], layout);
}

// Generate feature importance chart
function generateFeatureImportance() {
    const features = ['Transit Depth', 'Orbital Period', 'Transit Duration', 'Stellar Radius', 'Impact Param', 'Stellar Temp'];
    const importance = [0.30, 0.25, 0.20, 0.15, 0.05, 0.05];
    
    const trace = {
        x: importance,
        y: features,
        type: 'bar',
        orientation: 'h',
        marker: { color: '#00d4ff' }
    };
    
    const layout = {
        title: 'Feature Importance',
        xaxis: { title: 'Importance' },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#ffffff' }
    };
    
    Plotly.newPlot('feature-importance', [trace], layout);
}

// Generate light curve
function generateLightCurve(features) {
    const time = [];
    const flux = [];
    const period = features.period || 10;
    const depth = features.depth || 1000;
    const duration = features.duration || 6;
    
    for (let i = 0; i < 200; i++) {
        const t = i * period / 200;
        let f = 1.0;
        
        // Add transit
        const phase = (t % period) / period;
        if (Math.abs(phase - 0.5) < (duration/24) / (2 * period)) {
            f -= depth / 1000000;
        }
        
        // Add noise
        f += (Math.random() - 0.5) * 0.0001;
        
        time.push(t);
        flux.push(f);
    }
    
    const trace = {
        x: time,
        y: flux,
        type: 'scatter',
        mode: 'lines',
        name: 'Light Curve',
        line: { color: '#00d4ff', width: 2 }
    };
    
    const layout = {
        title: 'Simulated Light Curve',
        xaxis: { title: 'Time (days)' },
        yaxis: { title: 'Relative Flux' },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#ffffff' }
    };
    
    Plotly.newPlot('light-curve', [trace], layout);
}

// Generate batch results chart
function generateBatchChart(results) {
    const classifications = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'];
    const counts = classifications.map(c => 
        results.filter(r => r.result.classification === c).length
    );
    const colors = ['#00ff00', '#ffa500', '#ff0000'];
    
    const trace = {
        labels: classifications,
        values: counts,
        type: 'pie',
        marker: { colors: colors },
        textfont: { color: '#ffffff' }
    };
    
    const layout = {
        title: 'Classification Distribution',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#ffffff' }
    };
    
    Plotly.newPlot('batch-results-chart', [trace], layout);
}

// Initialize visualizations
window.onload = function() {
    generateROCCurve();
    generateFeatureImportance();
    
    // Generate sample light curve
    generateLightCurve({ period: 365, depth: 1000, duration: 13 });
};