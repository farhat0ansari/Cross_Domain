import numpy as np
import re
from collections import defaultdict
from transformers import BertTokenizer, BertModel
import torch

# Load BERT tokenizer globally
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def remove_stopwords(text):
    """Simple stopword removal"""
    stopwords = set([
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she',
        'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
        'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
        'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
        'against', 'between', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
        'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
        'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're',
        've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn',
        'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn',
        'wasn', 'weren', 'won', 'wouldn'
    ])
    words = text.lower().split()
    return [w for w in words if w not in stopwords and w.isalpha()]

def calculate_word_frequencies(reviews):
    """Calculate word frequencies in a set of reviews"""
    freq = defaultdict(int)
    for review in reviews:
        words = remove_stopwords(str(review))
        for word in set(words):  # unique words per review
            freq[word] += 1
    return freq

def calculate_affinity_score(word, domain_freq, other_freq, total_domain, total_other):
    """Calculate domain affinity score for a word - Equation 1 from paper"""
    # P(d|w) - probability of domain given word
    word_in_domain = domain_freq.get(word, 0)
    word_in_other = other_freq.get(word, 0)
    total_word = word_in_domain + word_in_other

    if total_word == 0:
        return 0

    p_domain_given_word = word_in_domain / total_word

    # H(w|d) - entropy
    p1 = word_in_domain / total_word if total_word > 0 else 0
    p2 = word_in_other / total_word if total_word > 0 else 0

    entropy = 0
    if p1 > 0:
        entropy -= p1 * np.log2(p1)
    if p2 > 0:
        entropy -= p2 * np.log2(p2)

    # N = 2 domain classes
    N = 2
    affinity = p_domain_given_word * (1 - entropy / np.log2(N + 1e-10))
    return affinity

def heuristic_masking(source_reviews, target_reviews, threshold_percentile=50):
    """
    Phase I - Step 1: Heuristic Masking
    Masks words that are more associated with source domain than target domain
    Based on Equation 3 from the paper: M1(w,S,T) = rho(w,S) - rho(w,T)
    """
    print("  Running Heuristic Masking...")

    # Calculate word frequencies
    source_freq = calculate_word_frequencies(source_reviews)
    target_freq = calculate_word_frequencies(target_reviews)
    total_source = len(source_reviews)
    total_target = len(target_reviews)

    # Calculate M1 scores for all words
    all_words = set(list(source_freq.keys()) + list(target_freq.keys()))
    m1_scores = {}
    for word in all_words:
        rho_source = calculate_affinity_score(word, source_freq, target_freq,
                                               total_source, total_target)
        rho_target = calculate_affinity_score(word, target_freq, source_freq,
                                               total_target, total_source)
        m1_scores[word] = rho_source - rho_target

    # Set threshold
    scores = list(m1_scores.values())
    threshold = np.percentile(scores, threshold_percentile)

    # Words to mask (strongly associated with source)
    words_to_mask = {w for w, s in m1_scores.items() if s > threshold}

    # Apply masking
    masked_reviews = []
    for review in source_reviews:
        words = str(review).split()
        masked = []
        for word in words:
            clean_word = re.sub(r'[^a-zA-Z]', '', word).lower()
            if clean_word in words_to_mask:
                masked.append('[MASK]')
            else:
                masked.append(word)
        masked_reviews.append(' '.join(masked))

    print(f"  Heuristic masking done. Words masked: {len(words_to_mask)}")
    return masked_reviews

def contextual_masking(masked_reviews, threshold_percentile=50):
    """
    Phase I - Step 2: Contextual Masking
    Uses BERT attention weights to mask domain-specific words
    Based on Equation 4,5,6 from the paper
    """
    print("  Running Contextual Masking (this may take a few minutes)...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    model = model.to(device)
    model.eval()

    final_masked = []
    batch_size = 16

    for i in range(0, len(masked_reviews), batch_size):
        batch = masked_reviews[i:i+batch_size]
        if i % 100 == 0:
            print(f"    Processing reviews {i}/{len(masked_reviews)}...")

        for review in batch:
            try:
                # Tokenize
                inputs = tokenizer(
                    review,
                    return_tensors='pt',
                    max_length=128,
                    truncation=True,
                    padding=True
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)

                # Get attention weights - average across all heads and layers
                attentions = outputs.attentions  # tuple of tensors
                # Use last layer attention
                last_attention = attentions[-1]  # shape: (batch, heads, seq, seq)
                # Average over heads and over attending tokens
                avg_attention = last_attention[0].mean(dim=0).mean(dim=0)  # shape: (seq,)
                avg_attention = avg_attention.cpu().numpy()

                # Get tokens
                tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

                # Calculate threshold for this review
                threshold = np.percentile(avg_attention, threshold_percentile)

                # Reconstruct review with additional masking
                words = review.split()
                word_tokens = tokenizer.tokenize(review)

                # Simple approach: mask words whose tokens have high attention
                new_words = []
                token_idx = 1  # skip [CLS]

                for word in words:
                    if word == '[MASK]':
                        new_words.append('[MASK]')
                        token_idx += 1
                        continue

                    word_toks = tokenizer.tokenize(word)
                    if token_idx < len(avg_attention) - 1:
                        word_attention = avg_attention[token_idx]
                        if word_attention >= threshold:
                            new_words.append('[MASK]')
                        else:
                            new_words.append(word)
                        token_idx += len(word_toks)
                    else:
                        new_words.append(word)

                final_masked.append(' '.join(new_words))

            except Exception as e:
                final_masked.append(review)

    # Free memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"  Contextual masking done.")
    return final_masked

def two_step_masking(source_reviews, target_reviews,
                     heuristic_threshold=50, contextual_threshold=50):
    """
    Complete Phase I: Two-Step Masking
    """
    print("\nPhase I: Two-Step Masking")
    print("-" * 40)

    # Step 1: Heuristic masking
    after_heuristic = heuristic_masking(
        source_reviews, target_reviews, heuristic_threshold
    )

    # Step 2: Contextual masking
    after_contextual = contextual_masking(after_heuristic, contextual_threshold)

    print(f"Phase I complete. {len(after_contextual)} reviews masked.")
    return after_contextual
