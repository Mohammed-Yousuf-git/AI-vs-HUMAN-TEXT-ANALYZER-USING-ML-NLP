"""
Text rewriting module using T5 paraphraser model
This module provides functionality to rewrite/humanize AI-generated text
"""

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
from typing import List, Optional

class T5Paraphraser:
    def __init__(self, model_name: str = "ramsrigouthamg/t5_paraphraser"):
        """Initialize the T5 paraphraser model"""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
    
    def _load_model(self):
        """Load the T5 model and tokenizer"""
        try:
            print(f"Loading T5 model: {self.model_name}")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def paraphrase(self, text: str, max_length: int = 512, num_return_sequences: int = 1, 
                   temperature: float = 1.0, do_sample: bool = True) -> List[str]:
        """
        Paraphrase the input text using T5 model
        
        Args:
            text: Input text to paraphrase
            max_length: Maximum length of generated text
            num_return_sequences: Number of paraphrases to generate
            temperature: Sampling temperature (higher = more random)
            do_sample: Whether to use sampling or greedy decoding
        
        Returns:
            List of paraphrased texts
        """
        try:
            # Prepare input
            input_text = f"paraphrase: {text}"
            inputs = self.tokenizer.encode_plus(
                input_text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate paraphrase
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode outputs
            paraphrases = []
            for output in outputs:
                paraphrase = self.tokenizer.decode(output, skip_special_tokens=True)
                paraphrases.append(paraphrase)
            
            return paraphrases
            
        except Exception as e:
            print(f"Error during paraphrasing: {e}")
            return [text]  # Return original text if paraphrasing fails

# Global paraphraser instance
paraphraser = None

def initialize_paraphraser():
    """Initialize the global paraphraser instance"""
    global paraphraser
    if paraphraser is None:
        try:
            paraphraser = T5Paraphraser()
        except Exception as e:
            print(f"Failed to initialize paraphraser: {e}")
            paraphraser = None

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences for better processing"""
    # Simple sentence splitting - you might want to use a more sophisticated approach
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def rewrite_text(text: str, chunk_size: int = 3) -> str:
    """
    Rewrite text to make it more human-like
    
    Args:
        text: Input text to rewrite
        chunk_size: Number of sentences to process at once
    
    Returns:
        Rewritten text
    """
    global paraphraser
    
    # Initialize paraphraser if not already done
    if paraphraser is None:
        initialize_paraphraser()
    
    # If paraphraser failed to initialize, return modified text with simple transformations
    if paraphraser is None:
        return simple_humanize(text)
    
    try:
        # Split text into sentences
        sentences = split_into_sentences(text)
        
        if not sentences:
            return text
        
        rewritten_sentences = []
        
        # Process sentences in chunks
        for i in range(0, len(sentences), chunk_size):
            chunk = sentences[i:i + chunk_size]
            chunk_text = ' '.join(chunk)
            
            # Skip very short chunks
            if len(chunk_text.split()) < 3:
                rewritten_sentences.extend(chunk)
                continue
            
            # Paraphrase the chunk
            paraphrases = paraphraser.paraphrase(
                chunk_text, 
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True
            )
            
            if paraphrases and paraphrases[0].strip():
                rewritten_sentences.append(paraphrases[0].strip())
            else:
                rewritten_sentences.extend(chunk)
        
        # Join and clean up the result
        result = ' '.join(rewritten_sentences)
        result = clean_text(result)
        
        return result
        
    except Exception as e:
        print(f"Error in rewrite_text: {e}")
        return simple_humanize(text)

def simple_humanize(text: str) -> str:
    """
    Simple text humanization using rule-based transformations
    This is a fallback when the T5 model is not available
    """
    # Add some variations and make it sound more natural
    transformations = [
        (r'\bIn conclusion\b', 'To wrap up'),
        (r'\bFurthermore\b', 'Also'),
        (r'\bMoreover\b', 'What\'s more'),
        (r'\bHowever\b', 'But'),
        (r'\bTherefore\b', 'So'),
        (r'\bNevertheless\b', 'Still'),
        (r'\bConsequently\b', 'As a result'),
        (r'\bAdditionally\b', 'Plus'),
        (r'\bIt is important to note that\b', 'Worth mentioning is that'),
        (r'\bIt should be noted that\b', 'Keep in mind that'),
    ]
    
    result = text
    for pattern, replacement in transformations:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result

def clean_text(text: str) -> str:
    """Clean up the generated text"""
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.!?,:;])', r'\1', text)
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    
    return text.strip()

# Test function
if __name__ == "__main__":
    # Test the rewriting functionality
    test_text = "This is a test sentence. It should be rewritten to sound more natural and human-like."
    result = rewrite_text(test_text)
    print(f"Original: {test_text}")
    print(f"Rewritten: {result}")