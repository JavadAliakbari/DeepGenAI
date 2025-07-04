# Paper Citation Network Builder

An AI-powered appli### Configuration

### API Key Setup
The application uses Semantic Scholar API. Update the API key in the code:
```python
HEADERS = {"X-API-KEY": "your_api_key_here"}
```

### Model Configuration
The application automatically downloads the Gemma-2-2b-it model from Hugging Face on first run.
- **Default location**: `./saved_models/gemma-2-2b-it/`
- **Model size**: ~5GB
- **Source**: `google/gemma-2-2b-it` on Hugging Face

If you want to use a different model or location, update these variables in the code:
```python
LOCAL_PATH = "./path/to/your/model"
HF_MODEL_NAME = "your_preferred_model"
```
builds and visualizes citation networks for academic papers using Semantic Scholar API and local language models for intelligent filtering.

## Features

- 🔍 **Smart Paper Search**: Search papers by exact title using Semantic Scholar API
- 🤖 **AI-Powered Filtering**: Uses local Gemma-2-2b model to filter papers by topic relevance
- 📊 **Interactive Visualization**: D3.js-powered interactive network graphs
- 🕸️ **Multi-level Crawling**: Recursively explore citation networks with configurable depth
- 🎯 **Topic-Focused**: Filter citations to show only papers relevant to your research topic

## Prerequisites

- Python 3.8+
- GPU with CUDA support (optional, for faster AI processing)
- Apple Silicon Mac (MPS support available)
- Internet connection for Semantic Scholar API
- **Hugging Face account** (free) with access to Gemma 2B model

## Installation

1. **Clone or download the project files**

2. **Set up Hugging Face access**:
   
   **a) Create a Hugging Face account:**
   - Go to [huggingface.co](https://huggingface.co) and create a free account
   
   **b) Accept Gemma 2B license:**
   - Visit the [Gemma 2B model page](https://huggingface.co/google/gemma-2-2b-it)
   - Click "Agree and access repository" to accept the license terms
   - **Note**: You must accept the license to download the model
   
   **c) Get your Hugging Face token:**
   - Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - Click "New token" and create a token with "Read" permissions
   - Copy the token (starts with `hf_`)
   
   **d) Add your token to the code:**
   - Open `gradio_paper_graph_app.py`
   - Find the line: `HUGGINGFACE_TOKEN = ""`
   - Replace it with: `HUGGINGFACE_TOKEN = "your_hf_token_here"`

3. **Set up a virtual environment** (recommended):
   
   **Using venv (Python 3.3+):**
   ```bash
   # Create virtual environment
   python -m venv paper_graph_env
   
   # Activate virtual environment
   # On macOS/Linux:
   source paper_graph_env/bin/activate
   # On Windows:
   # paper_graph_env\Scripts\activate
   ```
   
   **Using conda:**
   ```bash
   # Create virtual environment with Python 3.8+
   conda create -n paper_graph_env python=3.9
   
   # Activate virtual environment
   conda activate paper_graph_env
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

**Note:** The Gemma model will be automatically downloaded from Hugging Face on first run (~5GB). Make sure you have a stable internet connection and sufficient disk space.

## Configuration

### Required Setup

**Hugging Face Token:**
The application requires a Hugging Face token to download the Gemma model. Make sure you've completed step 2 above and added your token:
```python
HUGGINGFACE_TOKEN = "hf_your_token_here"
```

**API Key Setup:**
The application uses Semantic Scholar API. Update the API key in the code:
```python
HEADERS = {"X-API-KEY": "your_api_key_here"}
```

### Model Configuration
The application automatically downloads the Gemma-2-2b-it model from Hugging Face on first run.
- **Default location**: `./saved_models/gemma-2-2b-it/`
- **Model size**: ~5GB
- **Source**: `google/gemma-2-2b-it` on Hugging Face

If you want to use a different model or location, update these variables in the code:
```python
LOCAL_PATH = "./path/to/your/model"
HF_MODEL_NAME = "your_preferred_model"
```

## Usage

1. **Activate your virtual environment** (if using one):
   ```bash
   # For venv:
   source paper_graph_env/bin/activate  # macOS/Linux
   # paper_graph_env\Scripts\activate   # Windows
   
   # For conda:
   conda activate paper_graph_env
   ```

2. **Start the application**:
   ```bash
   python gradio_paper_graph_app.py
   ```

3. **Access the web interface**:
   Open your browser and go to `http://localhost:8080`

3. **Build a citation network**:
   - Enter the exact title of your seed paper
   - Specify a topic for filtering (e.g., "Graph Neural Networks")
   - Adjust search depth (1-3 levels)
   - Set maximum references per paper (5-20)
   - Click "Build Citation Network"

4. **Interact with the visualization**:
   - Drag nodes to rearrange the network
   - Zoom in/out with mouse wheel
   - Hover over nodes to see paper details
   - Use "Reset Zoom" and "Restart Layout" buttons

## Example Usage

**Seed Paper**: "Decoupled Subgraph Federated Learning"
**Topic**: "Graph Neural Networks"
**Depth**: 2
**Max References**: 10

This will:
1. Find the seed paper on Semantic Scholar
2. Fetch its references
3. Use AI to filter references related to "Graph Neural Networks"
4. Recursively explore 2 levels deep
5. Generate an interactive visualization

## Features in Detail

### AI Filtering
The application uses a local Gemma-2-2b model to determine if each paper's abstract is relevant to your specified topic. This ensures the citation network focuses on papers actually related to your research area.

### Visualization
- **Seed papers**: Red circles (larger)
- **Referenced papers**: Teal circles
- **Connections**: Directed arrows showing citation relationships
- **Interactive**: Drag, zoom, and hover for details

### Performance
- **Device Detection**: Automatically uses GPU (CUDA/MPS) if available
- **Model Caching**: Model downloads once and stays cached locally
- **Auto-download**: First run automatically downloads model from Hugging Face
- **Rate Limiting**: Respects API rate limits with retry logic

## Troubleshooting

### Hugging Face Issues

1. **License not accepted**:
   - Go to [Gemma 2B model page](https://huggingface.co/google/gemma-2-2b-it)
   - Make sure you've clicked "Agree and access repository"
   - You must accept the license before downloading

2. **Invalid token errors**:
   - Check your token is correctly added to `HUGGINGFACE_TOKEN`
   - Verify token has "Read" permissions
   - Generate a new token if needed at [HF Settings](https://huggingface.co/settings/tokens)

3. **Authentication failed**:
   - Ensure you're logged into Hugging Face account
   - Token should start with `hf_`
   - Check internet connection to huggingface.co

### Virtual Environment Issues

1. **Virtual environment not found**:
   - Make sure you've activated the virtual environment before running the app
   - Verify the environment exists: `conda env list` or check if the folder exists

2. **Package not found errors**:
   - Ensure you're in the correct virtual environment
   - Reinstall requirements: `pip install -r requirements.txt`

3. **Deactivating virtual environment**:
   ```bash
   # For venv:
   deactivate
   
   # For conda:
   conda deactivate
   ```

### Common Issues

1. **Model loading errors**:
   - First run: Model will auto-download (~5GB), ensure stable internet
   - Check available disk space (need ~10GB free for download + extraction)
   - Verify torch installation matches your hardware
   - If download fails, delete `./saved_models/` folder and try again

2. **Download issues**:
   - Check internet connection stability
   - Ensure firewall allows access to huggingface.co
   - For slow connections, consider downloading manually:
     ```python
     from transformers import AutoTokenizer, AutoModelForCausalLM
     model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
     tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
     model.save_pretrained("./saved_models/gemma-2-2b-it")
     tokenizer.save_pretrained("./saved_models/gemma-2-2b-it")
     ```

3. **API errors**:
   - Check your Semantic Scholar API key
   - Verify internet connection
   - Some papers may not be found (try exact titles)

4. **Visualization not showing**:
   - Check browser console for JavaScript errors
   - Ensure D3.js is loading (check debug info)
   - Try the test mode by entering "test" as paper title

### Debug Mode
Enter "test" as the paper title to generate a simple test graph and verify the visualization is working.

## File Structure

```
.
├── gradio_paper_graph_app.py    # Main application
├── requirements.txt             # Python dependencies
├── README.md                   # This file
└── saved_models/
    └── gemma-2-2b-it/          # Gemma model files
        ├── config.json
        ├── model-*.safetensors
        └── tokenizer files
```

## Dependencies

- **gradio**: Web interface framework
- **torch**: Deep learning framework
- **transformers**: Hugging Face model library
- **networkx**: Graph processing
- **requests**: HTTP API calls

## Security Note

⚠️ **Important**: This code currently has the Hugging Face token hardcoded in the source file. For production use, consider:

- Using environment variables: `os.getenv('HUGGINGFACE_TOKEN')`
- Using `.env` files with `python-dotenv`
- Passing tokens as command line arguments

Never commit your actual tokens to version control.

## License

This project is for educational and research purposes. Please respect the terms of use for:
- Semantic Scholar API
- Gemma model (Google)
- Any papers and data accessed through the APIs

## Contributing

Feel free to submit issues and enhancement requests!

## Acknowledgments

- Semantic Scholar for providing the citation data API
- Google for the Gemma language model
- D3.js for visualization capabilities
- Gradio for the web interface framework
