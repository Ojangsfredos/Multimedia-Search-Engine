import torch
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import gradio as gr
import time
from sklearn.metrics.pairwise import cosine_similarity
import cv2  # Library for video processing (OpenCV)

# Setup ChromaDB
client = chromadb.Client()

# Create a new collection for storing image and video embeddings
collection = client.create_collection("media_collection")  # Renamed for image/video

# Load CLIP model and processor for generating image, text, and video embeddings
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load and preprocess images
image_paths = [
    "img/beautiful_beach.jpg",  # Beautiful Beach
    "img/hiking_forest.jpg",  # Hiking forest
    "img/red_flower.jpg",  # Red flower
    "img/sunset_mountains.jpg"  # Sunset mountains
]

# Preprocess images and generate embeddings
images = [Image.open(image_path) for image_path in image_paths]
image_inputs = processor(images=images, return_tensors="pt", padding=True)

# Measure image ingestion time
start_ingestion_time = time.time()

with torch.no_grad():
    image_embeddings = model.get_image_features(**image_inputs).numpy()

# Convert numpy arrays to lists
image_embeddings = [embedding.tolist() for embedding in image_embeddings]

# Measure total ingestion time
end_ingestion_time = time.time()
ingestion_time = end_ingestion_time - start_ingestion_time

# Add image embeddings to the collection with metadata and display ingestion time
collection.add(
    embeddings=image_embeddings,
    metadatas=[{"type": "image", "path": image_path} for image_path in image_paths],
    ids=[str(i) for i in range(len(image_paths))]
)

# Log the ingestion performance for images
print(f"Image Data ingestion time: {ingestion_time:.4f} seconds")

# Add support for loading videos
video_paths = [
    "video/Starship.mp4",  # Sample Video 1
]

def process_video(video_path, frames_to_extract=5):
    """Extracts key frames from a video and generates embeddings for them."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or count >= frames_to_extract:
            break
        # Convert frame to PIL image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame)
        frames.append(pil_image)
        count += 1
    cap.release()
    
    # Preprocess video frames and generate embeddings
    video_inputs = processor(images=frames, return_tensors="pt", padding=True)
    with torch.no_grad():
        video_embeddings = model.get_image_features(**video_inputs).numpy()
    
    # Average frame embeddings to get a single video embedding
    video_embedding = video_embeddings.mean(axis=0).tolist()
    
    return video_embedding

# Preprocess and add video embeddings
for video_path in video_paths:
    video_embedding = process_video(video_path)
    collection.add(
        embeddings=[video_embedding],
        metadatas=[{"type": "video", "path": video_path}],
        ids=[str(len(image_paths) + video_paths.index(video_path))]
    )

# Create a function to calculate "accuracy" score based on cosine similarity
def calculate_accuracy(embedding, query_embedding):
    similarity = cosine_similarity([embedding], [query_embedding])[0][0]
    return similarity

# Define Gradio function for searching both images and videos
def search_media(query):
    # Simple validation: if the query is empty, show an error message
    if not query.strip():
        return None, None, "Oops! You forgot to type something on the query input!"

    print(f"\nQuery: {query}")  # Only show the query
    
    # Start measuring the query processing time
    start_time = time.time()
    
    # Generate an embedding for the query text
    inputs = processor(text=query, return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embedding = model.get_text_features(**inputs).numpy()

    query_embedding = query_embedding.tolist()[0]  # Ensure it's a single list

    # Perform a vector search in the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    # Retrieve the matched media (image or video)
    if 'metadatas' in results and results['metadatas'] and results['metadatas'][0]:
        result_metadata = results['metadatas'][0][0]  # Access metadata
        matched_media_path = result_metadata['path']  # Access path
        matched_media_type = result_metadata['type']  # Access type (image/video)
        
        # Retrieve appropriate embedding
        matched_media_index = int(results['ids'][0][0])
        matched_embedding = image_embeddings[matched_media_index] if matched_media_type == "image" else video_embedding
        
        # Calculate accuracy score based on cosine similarity
        accuracy_score = calculate_accuracy(matched_embedding, query_embedding)
    else:
        return None, None, "No results found. Please try a different query."

    end_time = time.time()
    query_time = end_time - start_time

    # Display result: either image or video path
    if matched_media_type == "image":
        result_image = Image.open(matched_media_path)
        file_name = matched_media_path.split('/')[-1]
        return result_image, None, f"Accuracy score: {accuracy_score:.4f}\nQuery time: {query_time:.4f} seconds", file_name
    else:
        return None, matched_media_path, f"Matched video: {matched_media_path}\nAccuracy score: {accuracy_score:.4f}\nQuery time: {query_time:.4f} seconds", matched_media_path.split('/')[-1]

# Suggested queries
queries = [
    "A beautiful beach with waves crashing",
    "A group of people hiking in the forest",
    "A close-up shot of a red flower",
    "A scenic view of mountains during sunset",
    "A video of starship"
]

# Function to populate the query input box with the suggested query
def populate_query(suggested_query):
    return suggested_query

# Gradio Interface Layout
with gr.Blocks() as gr_interface:
    gr.Markdown("# Multimedia Search Engine using ChromaDB")
    with gr.Row():
        # Left Panel
        with gr.Column():
            # Display the ingestion time of image embeddings
            gr.Markdown(f"**Image Ingestion Time**: {ingestion_time:.4f} seconds")
            
            gr.Markdown("### Input Panel")
            custom_query = gr.Textbox(placeholder="Enter your custom query here", label="What are you looking for?")

            with gr.Row():
                submit_button = gr.Button("Submit Query")
                cancel_button = gr.Button("Cancel")

            # Suggested search phrases
            gr.Markdown("#### Suggested Search Phrases")
            with gr.Row(elem_id="button-container"):
                for query in queries:
                    gr.Button(query).click(fn=lambda q=query: populate_query(q), outputs=custom_query)

        # Right Panel
        with gr.Column():
            gr.Markdown("### Retrieved Media")
            image_output = gr.Image(type="pil", label="Result Image")  # For image results
            video_output = gr.Video(label="Result Video")  # For video results
            accuracy_output = gr.Textbox(label="Performance")

        submit_button.click(fn=search_media, inputs=custom_query, outputs=[image_output, video_output, accuracy_output])
        cancel_button.click(fn=lambda: (None, None, ""), outputs=[image_output, video_output, accuracy_output])

# Launch the Gradio interface
gr_interface.launch(share=True)
