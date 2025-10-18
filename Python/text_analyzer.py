import string

def analyze_text():
    text = input("Enter text to analyze: ")
    words = text.split()
    sentences = text.split('.')

    print(f"Total words: {len(words)}")    
    print(f"Total sentences: {len(sentences)}")    

analyze_text()
