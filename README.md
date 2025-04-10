Resume Data Extraction Project
This project provides a machine learning-based solution for extracting structured information from resumes using Natural Language Processing (NLP). The model uses BERT for token classification to extract important details like names, emails, skills, education, and work experience from resumes, with an interactive UI built using Streamlit and a Flask API to serve the model.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The Resume Data Extraction project is aimed at automating the extraction of key information from resumes to aid in recruitment and human resource management. The main components of the project are:

- BERT-based Token Classification Model: Trained on labeled resume data to identify entities like name, email, skills, education, and experience.
- Flask API: Exposes the trained model to be accessed via RESTful API for predictions.
- Streamlit UI: Allows users to upload resume files and get instant predictions through a web interface.
- PyMuPDF: Library for parsing PDF files to extract text content from resumes.

## Technologies Used

- BERT: A transformer-based NLP model pre-trained by Hugging Face for token classification tasks.
- TensorFlow: Deep learning framework used for model training and deployment.
- Keras: High-level neural networks API used with TensorFlow for model development.
- Streamlit: Python library for building interactive web applications quickly and easily.
- Flask: Lightweight web framework for serving the model through an API.
- PyMuPDF: A library for extracting text from PDF documents, used to parse resumes.
- Transformers: Hugging Face library for easy access to pre-trained BERT models and tokenizers.
- Pandas: Used for data manipulation and processing.

## Setup and Installation

### Prerequisites:

- Python 3.12 (or compatible)
- TensorFlow 2.17.0
- Keras 3.2.1
- Streamlit

### Installation Steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/alishad846/Resume-Data-Extraction.git
   cd Resume-Data-Extraction
