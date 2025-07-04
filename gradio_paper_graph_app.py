import json
import time
import requests
import networkx as nx
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import html
import os

# Configuration
SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
REFERENCES_URL = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
LOCAL_PATH = "./saved_models/gemma-2-2b-it"
HF_MODEL_NAME = "google/gemma-2-2b-it"  # Hugging Face model identifier
HEADERS = {"X-API-KEY": ""}
HUGGINGFACE_TOKEN = "hf_your_token_here"

# Global variables to store loaded model and tokenizer
model = None
tokenizer = None
device = None


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def check_and_download_model():
    """Check if model exists locally, if not download from Hugging Face"""
    # Check if local model directory exists and has required files
    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    model_files_exist = os.path.exists(LOCAL_PATH) and all(
        os.path.exists(os.path.join(LOCAL_PATH, f)) for f in required_files
    )

    if not model_files_exist:
        print(f"Model not found locally at {LOCAL_PATH}")
        print(f"Downloading {HF_MODEL_NAME} from Hugging Face...")

        try:
            # Create directory if it doesn't exist
            login(HUGGINGFACE_TOKEN)
            os.makedirs(LOCAL_PATH, exist_ok=True)

            # Download model and tokenizer from Hugging Face
            print("Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
            tokenizer.save_pretrained(LOCAL_PATH)

            print("Downloading model (this may take a while - ~5GB)...")
            model = AutoModelForCausalLM.from_pretrained(
                HF_MODEL_NAME, torch_dtype=torch.bfloat16
            )
            model.save_pretrained(LOCAL_PATH)

            print(f"Model successfully downloaded and saved to {LOCAL_PATH}")
            return True

        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            print("Please check your internet connection and try again.")
            return False
    else:
        print(f"Model found locally at {LOCAL_PATH}")
        return True


def load_model():
    """Load the model and tokenizer once at startup"""
    global model, tokenizer, device

    if model is None:
        # First check and download model if needed
        if not check_and_download_model():
            raise Exception("Failed to download model from Hugging Face")

        device = get_device()
        print(f"Loading model on device: {device}")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                LOCAL_PATH, torch_dtype=torch.bfloat16
            )
            tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)
            model = model.to(device)
            model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # If loading fails, try downloading again
            print("Attempting to re-download model...")
            if os.path.exists(LOCAL_PATH):
                import shutil

                shutil.rmtree(LOCAL_PATH)
            if check_and_download_model():
                model = AutoModelForCausalLM.from_pretrained(
                    LOCAL_PATH, torch_dtype=torch.bfloat16
                )
                tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)
                model = model.to(device)
                model.eval()
                print("Model loaded successfully after re-download!")
            else:
                raise Exception("Failed to load model after multiple attempts")


def get_response(url, params=None):
    """Helper function to make GET requests with retries."""
    for _ in range(10):
        resp = requests.get(url, params=params, headers=HEADERS)
        if resp.status_code == 200:
            return resp.json()
        time.sleep(2)
    resp.raise_for_status()
    return resp.json()


def search_paper_by_title(title):
    """Search for the exact paper title, return the top hit."""
    params = {"query": title, "limit": 1, "fields": "paperId,title,abstract"}
    resp = get_response(SEARCH_URL, params=params)
    data = resp.get("data", [])
    if not data:
        raise ValueError(f"No paper found for title: {title}")
    return data[0]


def get_references(paper_id, max_refs=20):
    """Fetch references (papers cited by this paper)."""
    url = REFERENCES_URL.format(paper_id=paper_id)
    params = {
        "fields": "citedPaper.paperId,citedPaper.title",
        "limit": max_refs,
    }
    resp = get_response(url, params=params)
    items = resp.get("data", [])
    refs = []
    if items is None:
        return refs
    for item in items:
        ref = item.get("citedPaper", {})
        pid = ref.get("paperId")
        title = ref.get("title")
        if pid and title:
            refs.append({"paperId": pid, "title": title})
    return refs


def filter_with_llm(papers, topic, tokenizer, model, device):
    """Filter list of papers by asking the LLM if each abstract is about topic."""
    relevant = []
    for p in papers:
        if not p.get("abstract"):
            continue

        prompt = (
            f"Question: Is this paper mainly about {topic}? Reply Yes or No.\n"
            f"Abstract: {p.get('abstract', '')}\nAnswer:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=3)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ans = text.split("Answer:")[-1].strip().lower()
        if ans.startswith("yes"):
            relevant.append(p)
    return relevant


def crawl_references(
    paper, topic, depth, max_refs, G, tokenizer, model, device, visited
):
    """Recursively crawl references up to depth levels, filtering by topic."""
    if depth == 0:
        return

    parent_id = paper["paperId"]
    visited.add(parent_id)

    # Fetch references
    refs = get_references(parent_id, max_refs)

    # Enrich refs with abstract
    for r in refs:
        try:
            params = {"query": r["title"], "limit": 1, "fields": "abstract"}
            resp = get_response(SEARCH_URL, params=params)
            data = resp.get("data", [])
            r["abstract"] = data[0].get("abstract", "") if data else ""
        except:
            r["abstract"] = ""

    # Filter with LLM
    filtered = filter_with_llm(refs, topic, tokenizer, model, device)

    # Add to graph and recurse
    for ref in filtered:
        rid = ref["paperId"]
        if not G.has_node(rid):
            G.add_node(rid, title=ref["title"])
        if not G.has_edge(parent_id, rid):
            G.add_edge(parent_id, rid)

        if rid not in visited:
            crawl_references(
                ref, topic, depth - 1, max_refs, G, tokenizer, model, device, visited
            )


def create_graph_data(G):
    """Convert a NetworkX graph to JSON data compatible with D3.js"""
    seed_id = list(G.nodes())[0]

    nodes = []
    for node_id in G.nodes():
        nodes.append(
            {
                "id": node_id,
                "title": G.nodes[node_id].get("title", "Unknown"),
                "isSeed": node_id == seed_id,
            }
        )

    links = []
    for source, target in G.edges():
        links.append({"source": source, "target": target})

    return {"nodes": nodes, "links": links}


def export_graph_to_json(G, output_file):
    """Export a NetworkX graph to a JSON file compatible with D3.js"""
    seed_id = list(G.nodes())[0]

    nodes = []
    for node_id in G.nodes():
        nodes.append(
            {
                "id": node_id,
                "title": G.nodes[node_id].get("title", "Unknown"),
                "isSeed": node_id == seed_id,
            }
        )

    links = []
    for source, target in G.edges():
        links.append({"source": source, "target": target})

    graph_data = {"nodes": nodes, "links": links}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)


def build_paper_graph(paper_title, topic, depth=2, max_refs=10):
    """Main function to build the paper graph and return HTML visualization"""
    try:
        # Special test mode
        if paper_title.lower() == "test":
            test_data = {
                "nodes": [
                    {"id": "test1", "title": "Test Paper 1", "isSeed": True},
                    {"id": "test2", "title": "Test Paper 2", "isSeed": False},
                    {"id": "test3", "title": "Test Paper 3", "isSeed": False},
                ],
                "links": [
                    {"source": "test1", "target": "test2"},
                    {"source": "test1", "target": "test3"},
                ],
            }
            html_content = create_d3_html(test_data)
            return "Test graph created: 3 nodes, 2 edges", html_content

        # Check if model needs to be downloaded (inform user)
        if not os.path.exists(LOCAL_PATH) or not os.path.exists(
            os.path.join(LOCAL_PATH, "config.json")
        ):
            # return (
            # "Downloading AI model from Hugging Face (~5GB). This may take several minutes on first run...",
            # "<p>Please wait while the model is being downloaded...</p>",
            # )
            pass

        # Load model if not already loaded
        load_model()

        # Search for seed paper
        seed = search_paper_by_title(paper_title)

        # Build graph
        G = nx.DiGraph()
        visited = set()
        G.add_node(seed["paperId"], title=seed["title"])

        crawl_references(
            seed, topic, depth, max_refs, G, tokenizer, model, device, visited
        )

        # Convert graph to JSON data
        graph_data = create_graph_data(G)

        # Create HTML visualization with embedded data
        html_content = create_d3_html(graph_data)

        # Return statistics and HTML
        stats = f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"

        return stats, html_content

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, "<p>Error occurred while building the graph.</p>"


def create_d3_html(graph_data):
    """Create the HTML content with D3 visualization and embedded graph data"""
    # Convert graph data to JSON string for embedding
    graph_json = json.dumps(graph_data)

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Paper Citation Network</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 10px;
                background-color: #f5f5f5;
            }}
            #graph {{
                width: 100%;
                height: 600px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                border: 2px solid #ddd; /* Debug border */
            }}
            .tooltip {{
                position: absolute;
                padding: 10px;
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 4px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.2);
                max-width: 400px;
                font-size: 12px;
                pointer-events: none;
                z-index: 10;
            }}
            .controls {{
                display: flex;
                justify-content: center;
                margin: 10px 0;
                gap: 10px;
            }}
            button {{
                padding: 8px 15px;
                border: none;
                border-radius: 4px;
                background-color: #007bff;
                color: white;
                cursor: pointer;
            }}
            button:hover {{
                background-color: #0056b3;
            }}
            .debug-info {{
                background-color: #e7f3ff;
                padding: 10px;
                margin: 10px 0;
                border-left: 4px solid #007bff;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="debug-info">
            <strong>Debug Info:</strong>
            <span id="debug-status">Loading...</span>
        </div>
        
        <div class="controls">
            <button onclick="resetZoom()">Reset Zoom</button>
            <button onclick="restartSimulation()">Restart Layout</button>
        </div>
        <div id="graph"></div>
        
        <script>
            console.log("Script starting...");
            
            // Debug function
            function updateDebug(message) {{
                const debugEl = document.getElementById('debug-status');
                if (debugEl) {{
                    debugEl.innerHTML = message;
                }}
                console.log("DEBUG:", message);
            }}
            
            try {{
                updateDebug("D3 version: " + (typeof d3 !== 'undefined' ? d3.version : 'NOT LOADED'));
                
                // Embedded graph data
                const graph = {graph_json};
                console.log("Graph data loaded:", graph);
                updateDebug(`Graph loaded: ${{graph.nodes.length}} nodes, ${{graph.links.length}} links`);
                
                const graphContainer = document.getElementById('graph');
                if (!graphContainer) {{
                    throw new Error("Graph container not found!");
                }}
                
                const width = graphContainer.clientWidth || 800;
                const height = graphContainer.clientHeight || 600;
                updateDebug(`Container size: ${{width}}x${{height}}`);

                const svg = d3.select('#graph')
                    .append('svg')
                    .attr('width', width)
                    .attr('height', height);

                const g = svg.append('g');

                const zoom = d3.zoom()
                    .scaleExtent([0.1, 4])
                    .on('zoom', (event) => {{
                        g.attr('transform', event.transform);
                    }});

                svg.call(zoom);

                const simulation = d3.forceSimulation()
                    .force('link', d3.forceLink().id(d => d.id).distance(100))
                    .force('charge', d3.forceManyBody().strength(-400))
                    .force('center', d3.forceCenter(width / 2, height / 2))
                    .force('x', d3.forceX(width / 2).strength(0.05))
                    .force('y', d3.forceY(height / 2).strength(0.05));

                // Create arrow markers
                svg.append('defs').append('marker')
                    .attr('id', 'arrowhead')
                    .attr('viewBox', '-0 -5 10 10')
                    .attr('refX', 25)
                    .attr('refY', 0)
                    .attr('orient', 'auto')
                    .attr('markerWidth', 6)
                    .attr('markerHeight', 6)
                    .append('path')
                    .attr('d', 'M 0,-5 L 10,0 L 0,5')
                    .attr('fill', '#666');

                // Create links
                const link = g.append('g')
                    .selectAll('line')
                    .data(graph.links)
                    .join('line')
                    .attr('stroke', '#666')
                    .attr('stroke-width', 2)
                    .attr('stroke-opacity', 0.6)
                    .attr('marker-end', 'url(#arrowhead)');

                console.log("Links created:", link.size());

                // Create nodes
                const node = g.append('g')
                    .selectAll('circle')
                    .data(graph.nodes)
                    .join('circle')
                    .attr('r', d => d.isSeed ? 12 : 8)
                    .attr('fill', d => d.isSeed ? '#ff6b6b' : '#4ecdc4')
                    .attr('stroke', '#fff')
                    .attr('stroke-width', 2)
                    .call(d3.drag()
                        .on('start', dragstarted)
                        .on('drag', dragged)
                        .on('end', dragended));

                console.log("Nodes created:", node.size());

                // Add node labels
                const label = g.append('g')
                    .selectAll('text')
                    .data(graph.nodes)
                    .join('text')
                    .text(d => d.title.length > 30 ? d.title.substring(0, 30) + '...' : d.title)
                    .attr('font-size', 10)
                    .attr('font-family', 'Arial, sans-serif')
                    .attr('text-anchor', 'middle')
                    .attr('dy', 25)
                    .attr('fill', '#333');

                console.log("Labels created:", label.size());

                // Add tooltip - simplified
                const tooltip = d3.select('body').append('div')
                    .attr('class', 'tooltip')
                    .style('opacity', 0);

                node.on('mouseover', (event, d) => {{
                    tooltip.transition()
                        .duration(200)
                        .style('opacity', .9);
                    tooltip.html(`<strong>${{d.title}}</strong><br/>ID: ${{d.id}}`)
                        .style('left', (event.pageX + 10) + 'px')
                        .style('top', (event.pageY - 28) + 'px');
                }})
                .on('mouseout', () => {{
                    tooltip.transition()
                        .duration(500)
                        .style('opacity', 0);
                }});

                // Update simulation
                simulation.nodes(graph.nodes);
                simulation.force('link').links(graph.links);

                simulation.on('tick', () => {{
                    link
                        .attr('x1', d => d.source.x)
                        .attr('y1', d => d.source.y)
                        .attr('x2', d => d.target.x)
                        .attr('y2', d => d.target.y);

                    node
                        .attr('cx', d => d.x)
                        .attr('cy', d => d.y);

                    label
                        .attr('x', d => d.x)
                        .attr('y', d => d.y);
                }});

                function dragstarted(event, d) {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                }}

                function dragged(event, d) {{
                    d.fx = event.x;
                    d.fy = event.y;
                }}

                function dragended(event, d) {{
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }}

                // Global functions for buttons
                window.resetZoom = function() {{
                    svg.transition().duration(750).call(
                        zoom.transform,
                        d3.zoomIdentity
                    );
                }};

                window.restartSimulation = function() {{
                    simulation.alpha(1).restart();
                }};
                
                updateDebug(`Visualization complete! Nodes: ${{node.size()}}, Links: ${{link.size()}}`);
                
            }} catch (error) {{
                console.error("Error creating visualization:", error);
                updateDebug(`ERROR: ${{error.message}}`);
                document.getElementById('graph').innerHTML = `<div style="padding: 20px; color: red; text-align: center;"><h3>Error:</h3><p>${{error.message}}</p></div>`;
            }}
        </script>
    </body>
    </html>
    """

    # Escape HTML for srcdoc
    escaped = html.escape(html_content)

    # Wrap in iframe
    iframe = f"""
    <iframe
      srcdoc="{escaped}"
      style="width:100%; height:650px; border:none;"
    ></iframe>
    """
    return iframe


# Create Gradio interface
def create_gradio_interface():
    with gr.Blocks(
        title="Paper Citation Network Builder", theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("# 📚 Paper Citation Network Builder")
        gr.Markdown(
            "Enter a paper title and topic to build and visualize a citation network using AI filtering."
        )

        with gr.Row():
            with gr.Column(scale=1):
                paper_title = gr.Textbox(
                    label="Paper Title",
                    placeholder="Enter the exact title of the seed paper",
                    value="Decoupled Subgraph Federated Learning",
                )
                topic = gr.Textbox(
                    label="Topic for Filtering",
                    placeholder="Enter the topic to filter relevant papers",
                    value="Graph Neural Networks",
                )

                with gr.Row():
                    depth = gr.Slider(
                        label="Search Depth",
                        minimum=1,
                        maximum=3,
                        value=2,
                        step=1,
                        info="How many levels deep to search",
                    )
                    max_refs = gr.Slider(
                        label="Max References per Paper",
                        minimum=5,
                        maximum=20,
                        value=10,
                        step=1,
                        info="Maximum references to fetch per paper",
                    )

                build_btn = gr.Button("🔍 Build Citation Network", variant="primary")

                stats_output = gr.Textbox(
                    label="Build Statistics", interactive=False, lines=2
                )

        with gr.Column(scale=2):
            html_output = gr.HTML(
                label="Citation Network Visualization",
                value="<p>Enter paper details and click 'Build Citation Network' to generate the visualization.</p>",
            )

        build_btn.click(
            fn=build_paper_graph,
            inputs=[paper_title, topic, depth, max_refs],
            outputs=[stats_output, html_output],
        )

        gr.Markdown(
            """
        ### Instructions:
        1. Enter the exact title of your seed paper
        2. Specify the topic for AI-based filtering of relevant papers
        3. Adjust search depth and maximum references as needed
        4. Click "Build Citation Network" to generate the interactive visualization
        
        **Note:** 
        - The first run will automatically download the AI model (~5GB) from Hugging Face
        - This may take several minutes depending on your internet connection
        - Subsequent runs will be much faster as the model is cached locally
        """
        )

    return demo


if __name__ == "__main__":
    # Create and launch the Gradio app
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=8080, share=False, debug=False)
