# Paper Citation Network Builder

An AI-powered application that builds and visualizes citation networks for academic papers using Semantic Scholar API and local language models for intelligent filtering.

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

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Gemma model** (if not already present):
   The application expects the Gemma-2-2b-it model in `./saved_models/gemma-2-2b-it/`
   
   You can download it using Hugging Face transformers:
   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM
   
   model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
   tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
   
   model.save_pretrained("./saved_models/gemma-2-2b-it")
   tokenizer.save_pretrained("./saved_models/gemma-2-2b-it")
   ```

## Configuration

### API Key Setup
The application uses Semantic Scholar API. Update the API key in the code:
```python
HEADERS = {"X-API-KEY": "your_api_key_here"}
```

### Model Path
If your Gemma model is in a different location, update:
```python
LOCAL_PATH = "./path/to/your/gemma-model"
```

## Usage

1. **Start the application**:
   ```bash
   python gradio_paper_graph_app.py
   ```

2. **Access the web interface**:
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
- **Caching**: Model loads once and stays in memory
- **Rate Limiting**: Respects API rate limits with retry logic

## Troubleshooting

### Common Issues

1. **Model loading errors**:
   - Ensure the Gemma model is properly downloaded
   - Check available memory (model requires ~5GB)
   - Verify torch installation matches your hardware

2. **API errors**:
   - Check your Semantic Scholar API key
   - Verify internet connection
   - Some papers may not be found (try exact titles)

3. **Visualization not showing**:
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
