import torch
import numpy as np
from transformers import BertTokenizerFast as BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import re

class BertWSD:
    def __init__(self, model_name='bert-base-uncased'):
        print(f"Loading {model_name}...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name) # WordPiece
        self.model = BertModel.from_pretrained(model_name, output_hidden_states=True)
        
        # Turn off dropout 
        self.model.eval()
        print("Model loaded.")

    def get_bert_embeddings(self, text, target_word_idx=None):
        """
        This method is retired.
        Runs BERT on the text and returns embeddings.
        If target_word_idx is provided, returns the embedding for that specific word (handling subwords).
        Otherwise, returns the word embedding from the definitions.
        """
        marked_text = "[CLS] " + text + " [SEP]"
        """unfriendly →['un', '##friend', '##ly']"""
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]

        # Stack the last 4 layers, common practice for non-cls tasks
        token_embeddings = torch.stack(hidden_states[-4:], dim=0) 
        # Shape: [4, 1, seq_len, 768]
        
        # Remove batch dimension -> [4, seq_len, 768]
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        
        # Permute to [seq_len, 4, 768]
        token_embeddings = token_embeddings.permute(1, 0, 2)
        vectors = []
        for token in token_embeddings:
            sum_vec = torch.sum(token, dim=0)
            vectors.append(sum_vec)
        
        # Convert to tensor [seq_len, 768]
        vectors = torch.stack(vectors)

        if target_word_idx is not None:
            return vectors[target_word_idx].reshape(1, -1)
        else:
            if len(vectors) > 2:
                return torch.mean(vectors[1:-1], dim=0).reshape(1, -1)
            else:
                return torch.mean(vectors, dim=0).reshape(1, -1)

    def process_text(self, text, target_word, definition_embedding, window_size=10, threshold=0.6):
        """
        Scans the text for the target word, extracts context, computes similarity, and returns alerts.
        """
        alerts = []
        
        # Find all start indices of the target word
        # Using regex to find whole words
        matches = [m for m in re.finditer(r'\b' + re.escape(target_word) + r'\b', text, re.IGNORECASE)]
        
        words = text.split()
        
        for match in matches:
            start_char = match.start()
            end_char = match.end()
            
            # Create a window of words around the match
            # and map character index to word index.
            
            # Robust approach: Use the tokenizer's offset mapping.
            inputs = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
            input_ids = inputs['input_ids']
            offset_mapping = inputs['offset_mapping']
            
            # Find which tokens correspond to our character match
            target_token_indices = []
            for idx, (start, end) in enumerate(offset_mapping):
                if start >= start_char and end <= end_char:
                    target_token_indices.append(idx)
                elif start < start_char and end > start_char: # Overlap start
                    target_token_indices.append(idx)
                elif start < end_char and end > end_char: # Overlap end
                    target_token_indices.append(idx)
            
            if not target_token_indices:
                continue
                
            # Define window in terms of TOKENS, not words, to be easier with BERT
            # but still centering around the target tokens
            center_idx = target_token_indices[0] # Start of the word
            window_start = max(0, center_idx - window_size)
            window_end = min(len(input_ids), target_token_indices[-1] + window_size + 1)
            
            # Extract window tokens
            window_input_ids = input_ids[window_start:window_end]
            
            # New index of target in the window
            new_target_idx = target_token_indices[0] - window_start
            
            # Convert back to tensor
            tokens_tensor = torch.tensor([window_input_ids])
            segments_tensors = torch.tensor([[1] * len(window_input_ids)])
            
            # Run BERT on window
            with torch.no_grad():
                outputs = self.model(tokens_tensor, segments_tensors)
                hidden_states = outputs[2]
            
            # Stack and sum layers
            token_embeddings = torch.stack(hidden_states[-4:], dim=0)
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            token_embeddings = token_embeddings.permute(1, 0, 2)
            vectors = torch.sum(token_embeddings, dim=1) # [seq_len, 768]
            
            # Handle subwords: average the vectors for the target word tokens
            # The target word might span multiple tokens in the window
            num_subwords = len(target_token_indices)
            # In the window, these are at new_target_idx, new_target_idx+1, ...
            target_vecs = vectors[new_target_idx : new_target_idx + num_subwords]
            target_embedding = torch.mean(target_vecs, dim=0).reshape(1, -1)
            
            # Compute similarity
            sim = cosine_similarity(target_embedding, definition_embedding)[0][0]
            
            if sim >= threshold:
                # Reconstruct context string for display
                # We can use the offset mapping to get the original string slice if we kept it,
                # or decode the tokens.
                context_str = self.tokenizer.decode(window_input_ids, skip_special_tokens=True)
                
                alerts.append({
                    #"doc_id": TBD
                    "word": target_word,
                    "context": context_str,
                    "similarity": float(sim),
                    "match_index": start_char
                })
                
        return alerts

    def get_embedding_for_word(self, text, target_word):
        """
        Finds the target word in the text and returns its embedding.
        Uses the first occurrence found.
        """
        # Tokenize and get offsets
        inputs = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids']
        offset_mapping = inputs['offset_mapping']
        
        # Find the character span of the target word using regex
        match = re.search(r'\b' + re.escape(target_word) + r'\b', text, re.IGNORECASE)
        if not match:
            # Fallback: if regex fails (e.g. special chars), try simple string find
            start = text.lower().find(target_word.lower())
            if start == -1:
                return None
            end = start + len(target_word)
        else:
            start, end = match.span()
            
        # Map character span to token indices
        target_indices = []
        for idx, (t_start, t_end) in enumerate(offset_mapping):
            # Skip special tokens (0,0)
            if t_start == 0 and t_end == 0:
                continue
            # Check overlap
            if t_start < end and t_end > start:
                target_indices.append(idx)
                
        if not target_indices:
            return None

        # Run BERT
        tokens_tensor = torch.tensor([input_ids])
        segments_tensors = torch.tensor([[1] * len(input_ids)])
        
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
            
        # Stack last 4 layers
        token_embeddings = torch.stack(hidden_states[-4:], dim=0) # [4, 1, seq, 768]
        token_embeddings = torch.squeeze(token_embeddings, dim=1) # [4, seq, 768]
        token_embeddings = token_embeddings.permute(1, 0, 2)      # [seq, 4, 768]
        
        # Sum last 4 layers to get vectors for each token
        vectors = torch.sum(token_embeddings, dim=1) # [seq, 768]
        
        # Extract target tokens and average them
        target_vecs = vectors[target_indices]
        final_embedding = torch.mean(target_vecs, dim=0).reshape(1, -1)
        
        return final_embedding

    def scan_dataset(self, dataset, target_word, definition, window_size=10, threshold=0.6):
        """
        Main entry point.
        dataset: List of strings (documents)
        target_word: String
        definition: String
        """
        print(f"Computing embedding for definition: '{definition}'")
        
        # STRATEGY: Constructed Context
        # We create a sentence where the target word is defined. 
        # This forces the target word embedding to align with the definition's meaning, because in this way they would share the same spanned space.
        definition_context = f"{target_word} means {definition}"
        
        def_embedding = self.get_embedding_for_word(definition_context, target_word)
        
        if def_embedding is None:
            print("Warning: Could not extract target word embedding from definition context. Falling back to sentence mean.")
            def_embedding = self.get_bert_embeddings(definition)
        
        all_alerts = []
        print(f"Scanning {len(dataset)} documents for '{target_word}'...")
        
        for i, text in enumerate(dataset):
            alerts = self.process_text(text, target_word, def_embedding, window_size, threshold)
            for alert in alerts:
                alert['doc_id'] = i
                all_alerts.append(alert)
                
        print(f"Done Scanning {len(dataset)} documents for '{target_word}'")
        return all_alerts
