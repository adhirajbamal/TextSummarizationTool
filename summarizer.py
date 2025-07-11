import nltk
from transformers import BartForConditionalGeneration, BartTokenizer
from nltk.tokenize import sent_tokenize

# Download NLTK punkt tokenizer if not already present
nltk.download('punkt')

class TextSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """
        Initialize the summarizer with BART model
        :param model_name: name of the pre-trained model
        """
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        
    def summarize(self, text, max_length=150, min_length=30):
        """
        Generate a summary of the input text
        :param text: input text to summarize
        :param max_length: maximum length of the summary
        :param min_length: minimum length of the summary
        :return: generated summary
        """
        # Tokenize and generate summary
        inputs = self.tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
        summary_ids = self.model.generate(
            inputs['input_ids'], 
            num_beams=4, 
            max_length=max_length, 
            min_length=min_length, 
            early_stopping=True
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def summarize_by_sentences(self, text, num_sentences=3):
        """
        Alternative summarization approach - extractive summarization by selecting top sentences
        :param text: input text to summarize
        :param num_sentences: number of sentences to include in summary
        :return: generated summary
        """
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
        
        # Simple approach: take first few sentences (more sophisticated approaches could use TF-IDF)
        return ' '.join(sentences[:num_sentences])

def main():
    print("CODTECH TEXT SUMMARIZATION TOOL")
    print("--------------------------------\n")
    
    summarizer = TextSummarizer()
    
    while True:
        print("\nOptions:")
        print("1. Enter text to summarize")
        print("2. Load text from file")
        print("3. Exit")
        
        choice = input("Select an option (1-3): ")
        
        if choice == '3':
            print("Exiting summarization tool...")
            break
            
        if choice == '1':
            text = input("\nEnter the text to summarize:\n")
        elif choice == '2':
            file_path = input("Enter file path: ")
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            except Exception as e:
                print(f"Error reading file: {e}")
                continue
        else:
            print("Invalid choice. Please try again.")
            continue
            
        # Get summary using BART model
        summary = summarizer.summarize(text)
        
        print("\nGenerated Summary:")
        print("------------------")
        print(summary)
        print("\n")

if __name__ == "__main__":
    main()
