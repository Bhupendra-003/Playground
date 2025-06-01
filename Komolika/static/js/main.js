// Modern JavaScript for Image Caption Generator

class ImageCaptionGenerator {
    constructor() {
        this.selectedModel = null;
        this.selectedFile = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.selectFirstAvailableModel();
    }

    setupEventListeners() {
        // Model selection
        document.querySelectorAll('.model-card').forEach(card => {
            card.addEventListener('click', (e) => this.selectModel(e.currentTarget));
        });

        // File upload
        const uploadArea = document.getElementById('uploadArea');
        const imageInput = document.getElementById('imageInput');
        const browseText = document.querySelector('.browse-text');

        // Click to browse
        uploadArea.addEventListener('click', () => imageInput.click());
        browseText.addEventListener('click', (e) => {
            e.stopPropagation();
            imageInput.click();
        });

        // File input change
        imageInput.addEventListener('change', (e) => this.handleFileSelect(e.target.files[0]));

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        uploadArea.addEventListener('drop', (e) => this.handleDrop(e));

        // Remove image
        document.getElementById('removeImage').addEventListener('click', () => this.removeImage());

        // Generate caption
        document.getElementById('generateBtn').addEventListener('click', () => this.generateCaption());

        // Copy caption
        document.getElementById('copyBtn').addEventListener('click', () => this.copyCaption());

        // Share caption
        document.getElementById('shareBtn').addEventListener('click', () => this.shareCaption());

        // Close error
        document.getElementById('closeError').addEventListener('click', () => this.hideError());
    }

    selectFirstAvailableModel() {
        const firstModel = document.querySelector('.model-card');
        if (firstModel) {
            this.selectModel(firstModel);
        }
    }

    selectModel(modelCard) {
        // Remove previous selection
        document.querySelectorAll('.model-card').forEach(card => {
            card.classList.remove('selected');
        });

        // Select new model
        modelCard.classList.add('selected');
        this.selectedModel = modelCard.dataset.model;
        
        this.updateGenerateButton();
    }

    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFileSelect(files[0]);
        }
    }

    handleFileSelect(file) {
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showError('Please select a valid image file.');
            return;
        }

        // Validate file size (16MB max)
        if (file.size > 16 * 1024 * 1024) {
            this.showError('File size must be less than 16MB.');
            return;
        }

        this.selectedFile = file;
        this.showImagePreview(file);
        this.updateGenerateButton();
    }

    showImagePreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const previewImg = document.getElementById('previewImg');
            const imageName = document.getElementById('imageName');
            const imagePreview = document.getElementById('imagePreview');
            const uploadArea = document.getElementById('uploadArea');

            previewImg.src = e.target.result;
            imageName.textContent = file.name;
            
            uploadArea.style.display = 'none';
            imagePreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    removeImage() {
        this.selectedFile = null;
        
        const uploadArea = document.getElementById('uploadArea');
        const imagePreview = document.getElementById('imagePreview');
        const imageInput = document.getElementById('imageInput');
        
        uploadArea.style.display = 'block';
        imagePreview.style.display = 'none';
        imageInput.value = '';
        
        this.updateGenerateButton();
        this.hideResults();
    }

    updateGenerateButton() {
        const generateBtn = document.getElementById('generateBtn');
        const canGenerate = this.selectedModel && this.selectedFile;
        
        generateBtn.disabled = !canGenerate;
    }

    async generateCaption() {
        if (!this.selectedModel || !this.selectedFile) {
            this.showError('Please select a model and upload an image.');
            return;
        }

        this.showLoading();
        this.hideError();
        this.hideResults();

        try {
            const formData = new FormData();
            formData.append('image', this.selectedFile);
            formData.append('model', this.selectedModel);

            const response = await fetch('/api/generate', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to generate caption');
            }

            this.showResults(data);
        } catch (error) {
            console.error('Error generating caption:', error);
            this.showError(error.message || 'Failed to generate caption. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    showResults(data) {
        const resultsSection = document.getElementById('resultsSection');
        const captionText = document.getElementById('captionText');
        const modelUsed = document.getElementById('modelUsed');
        const vocabInfo = document.getElementById('vocabInfo');

        captionText.textContent = data.caption;
        modelUsed.textContent = data.model === 'simple' ? 'Simple Model' : '8K Model';
        vocabInfo.textContent = `Vocabulary: ${data.vocab_size.toLocaleString()} words`;

        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    hideResults() {
        const resultsSection = document.getElementById('resultsSection');
        resultsSection.style.display = 'none';
    }

    async copyCaption() {
        const captionText = document.getElementById('captionText').textContent;
        
        try {
            await navigator.clipboard.writeText(captionText);
            this.showSuccess('Caption copied to clipboard!');
        } catch (error) {
            console.error('Failed to copy caption:', error);
            this.showError('Failed to copy caption to clipboard.');
        }
    }

    shareCaption() {
        const captionText = document.getElementById('captionText').textContent;
        
        if (navigator.share) {
            navigator.share({
                title: 'Generated Image Caption',
                text: captionText
            }).catch(error => {
                console.error('Error sharing:', error);
            });
        } else {
            // Fallback: copy to clipboard
            this.copyCaption();
        }
    }

    showLoading() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        loadingOverlay.style.display = 'flex';
    }

    hideLoading() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        loadingOverlay.style.display = 'none';
    }

    showError(message) {
        const errorMessage = document.getElementById('errorMessage');
        const errorText = document.getElementById('errorText');
        
        errorText.textContent = message;
        errorMessage.style.display = 'flex';
        
        // Auto-hide after 5 seconds
        setTimeout(() => this.hideError(), 5000);
    }

    hideError() {
        const errorMessage = document.getElementById('errorMessage');
        errorMessage.style.display = 'none';
    }

    showSuccess(message) {
        // Create a temporary success message
        const successDiv = document.createElement('div');
        successDiv.className = 'error-message';
        successDiv.style.background = 'var(--accent-success)';
        successDiv.innerHTML = `
            <i class="fas fa-check-circle"></i>
            <span>${message}</span>
        `;
        
        document.body.appendChild(successDiv);
        
        // Remove after 3 seconds
        setTimeout(() => {
            if (successDiv.parentNode) {
                successDiv.parentNode.removeChild(successDiv);
            }
        }, 3000);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ImageCaptionGenerator();
});
