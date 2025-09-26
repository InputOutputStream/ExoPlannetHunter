import { ExoplanetDetectorApp } from './app.js';

// Global app instance
let app = null;

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    try {
        console.log('🚀 Starting ExoPlanet AI Detector...');
        
        // Create and initialize application
        app = new ExoplanetDetectorApp();
        await app.initialize();
        
        console.log('✅ Application ready!');
        
        // Make app globally available for debugging
        window.exoplanetApp = app;
        
    } catch (error) {
        console.error('❌ Application initialization failed:', error);
        showInitializationError(error);
    }
});

// Handle window resize for responsive charts
window.addEventListener('resize', () => {
    if (app && app.visualizer) {
        app.visualizer.resizePlots();
    }
});

// Handle before unload for cleanup
window.addEventListener('beforeunload', () => {
    if (app) {
        app.dispose();
    }
});

// Error handling
function showInitializationError(error) {
    const container = document.querySelector('.container');
    if (container) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'init-error';
        errorDiv.innerHTML = `
            <div class="error-panel">
                <h2>🚨 Initialization Error</h2>
                <p>Failed to initialize the ExoPlanet AI Detector:</p>
                <pre>${error.message}</pre>
                <p>Please check the console for more details and refresh the page.</p>
                <button onclick="location.reload()" class="retry-btn">Retry</button>
            </div>
        `;
        container.innerHTML = '';
        container.appendChild(errorDiv);
    }
}

// Export for module usage
export { app };