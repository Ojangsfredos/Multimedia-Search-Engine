
# Multimedia Search Engine using ChromaDB

## Project Purpose

The Multimedia Search Engine is designed to facilitate efficient retrieval of images and videos based on textual descriptions. Leveraging the power of ChromaDB, this application generates embeddings from both text and media files, allowing for high-quality, similarity-based searches. The project aims to simplify the process of finding relevant multimedia content, making it a valuable tool for users who need quick access to visual information based on text queries.

## Key Features
- **Text-to-Image and Text-to-Video Search**: Users can input descriptive text and retrieve the most relevant images or videos.
- **Efficient Embedding Storage**: Media and text embeddings are stored in ChromaDB, enabling fast and accurate retrieval.
- **User-Friendly Interface**: The application provides a straightforward interface for searching and displaying results.

## Project Flow
1. **User Input**: The user enters a text query describing the desired media.
2. **Embedding Generation**: The application generates embeddings for the text input and stored media.
3. **Similarity Search**: ChromaDB performs a search for the closest matching embeddings.
4. **Results Display**: The application displays the retrieved images or videos based on the user's query.

## Installation Guide

To set up and run the application locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd multimedia-search-engine
   ```

2. **Create a Python virtual environment** to manage the project's dependencies:
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Directory Structure

- **img/**: This folder contains three sample images used in the project for text-to-image comparison and analysis.
- **requirements.txt**: This file lists the dependencies necessary for the project, including ChromaDB, image processing libraries (e.g., Pillow), and other tools needed for text-to-image functionality.
- **text-to-image.py**: This Python script implements the text-to-image functionality using ChromaDB. It handles tasks such as creating embeddings from text and images, storing them in a collection, and performing queries to find the most relevant matches.

## Video Demonstration

![Multimedia Search Engine Demo](https://img.youtube.com/vi/emZToJkCZPs/0.jpg)
[Watch the demonstration on YouTube](https://youtu.be/emZToJkCZPs)

## Conclusion

This project showcases the capabilities of ChromaDB in building a multimedia search engine that allows users to efficiently retrieve images and videos based on text descriptions. It solves the problem of finding relevant multimedia content quickly, providing a robust solution for various applications.


```

