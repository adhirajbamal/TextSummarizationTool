#REQUIREMENTS
transformers>=4.0.0
torch>=1.7.0
nltk>=3.5

#USAGE
Run the summarization tool:
python summarizer.py
Follow the on-screen instructions to:
Enter text directly or load from a file
View the generated summary

#EXAMPLE 
Input Text:
The COVID-19 pandemic has caused unprecedented disruption to education systems worldwide. School closures affected more than 1.5 billion students across 190 countries. The crisis has exacerbated existing educational inequalities, with students from disadvantaged backgrounds being particularly affected due to lack of access to digital learning resources. Many institutions rapidly transitioned to online learning platforms, though the effectiveness of this approach varied significantly. Experts warn that the long-term impacts on student learning outcomes and mental health may persist for years after the pandemic ends.


Generated Summary:
The COVID-19 pandemic has caused unprecedented disruption to education systems worldwide, with school closures affecting more than 1.5 billion students. The crisis has exacerbated educational inequalities, with disadvantaged students particularly affected. Many institutions transitioned to online learning, though effectiveness varied.




# TextSummarizationTool
 Python script for a text summarization tool
# Text Summarization Tool

A Python tool for summarizing lengthy articles using Natural Language Processing techniques.

## Features

- Abstractive summarization using BART model from Hugging Face Transformers
- Extractive summarization option (selecting key sentences)
- Interactive command-line interface
- File input support

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/text-summarization-tool.git
   cd text-summarization-tool
 2.  Install dependencies:
   pip install -r requirements.txt
