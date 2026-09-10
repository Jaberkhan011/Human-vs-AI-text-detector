import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def count_syllables(word):
    """Simple rule-based syllable estimator."""
    word = re.sub(r"[^a-zA-Z]", "", word.lower())

    if not word:
        return 0

    # Handle silent 'e'
    word = re.sub(r"e$", "", word)

    # Count vowel groups
    groups = re.findall(r"[aeiouy]+", word)
    syllables = len(groups)

    return max(1, syllables)


def extract_text_features(raw_text):
    """
    Extract the model-free text features used by the classifier.

    Parameters
    ----------
    raw_text : str
        User-provided text.

    Returns
    -------
    pandas.DataFrame
        One-row DataFrame containing the features expected by the ML pipeline.
    """
    text = str(raw_text).strip()

    # WORDS
    words = re.findall(r"\b[a-zA-Z]+(?:['-][a-zA-Z]+)*\b", text)

    clean_words = [
        re.sub(r"[^a-zA-Z]", "", word).lower()
        for word in words
    ]
    clean_words = [w for w in clean_words if w]

    word_count = len(clean_words)

    # CHARACTERS
    character_count = len(text)

    # SENTENCES
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    sentence_count = len(sentences)

    sentence_lengths = []

    for sentence in sentences:
        sentence_words = re.findall(
            r"\b[a-zA-Z]+(?:['-][a-zA-Z]+)*\b",
            sentence
        )
        sentence_lengths.append(len(sentence_words))

    # AVERAGE WORD LENGTH
    if word_count > 0:
        average_word_length = (
            sum(len(word) for word in clean_words) / word_count
        )
    else:
        average_word_length = 0

    # AVERAGE SENTENCE LENGTH
    if sentence_count > 0:
        average_sentence_length = (
            sum(sentence_lengths) / sentence_count
        )
    else:
        average_sentence_length = 0

    # UNIQUE WORDS / LEXICAL DIVERSITY
    unique_words_count = len(set(clean_words))

    if word_count > 0:
        lexical_diversity = unique_words_count / word_count
    else:
        lexical_diversity = 0

    # PUNCTUATION
    punctuation_count = len(
        re.findall(r'[.,!?;:"\'()\[\]{}\-]', text)
    )

    if character_count > 0:
        punctuation_density = punctuation_count / character_count
    else:
        punctuation_density = 0

    # SPECIAL CHARACTERS
    special_char_count = len(
        re.findall(
            r'[^a-zA-Z0-9\s.,!?;:"\'()\[\]{}\-]',
            text
        )
    )

    # STOPWORDS
    stopword_count = sum(
        1 for word in clean_words
        if word in ENGLISH_STOP_WORDS
    )

    if word_count > 0:
        stopword_density = stopword_count / word_count
    else:
        stopword_density = 0

    # SYLLABLES
    syllable_counts = [
        count_syllables(word)
        for word in clean_words
    ]

    if word_count > 0:
        average_syllables = (
            sum(syllable_counts) / word_count
        )
    else:
        average_syllables = 0

    # FLESCH-KINCAID GRADE LEVEL
    if word_count > 0 and sentence_count > 0:
        avg_words_sentence = word_count / sentence_count
        avg_syllables_word = average_syllables

        flesch_kincaid = (
            0.39 * avg_words_sentence
            + 11.8 * avg_syllables_word
            - 15.59
        )
    else:
        flesch_kincaid = 0

    # SMOG INDEX
    complex_words = sum(
        1 for s in syllable_counts
        if s >= 3
    )

    if sentence_count > 0:
        smog_index = (
            1.043
            * np.sqrt(complex_words * (30 / sentence_count))
            + 3.1291
        )
    else:
        smog_index = 0

    # CAPITAL LETTERS
    capital_letter_count = sum(
        1 for char in text
        if char.isupper()
    )

    # BURSTINESS
    if sentence_count > 1 and average_sentence_length > 0:
        sentence_std = np.std(sentence_lengths)

        burstiness_score = (
            sentence_std / average_sentence_length
        ) * 100
    else:
        burstiness_score = 0

    # CODE DETECTION
    code_patterns = [
        r"```",
        r"\bdef\s+\w+\s*\(",
        r"\bclass\s+\w+",
        r"\bimport\s+\w+",
        r"\bfrom\s+\w+\s+import\b",
        r"#include\s*[<\"]",
        r"<\?php",
        r"=>",
        r"console\.log\s*\(",
    ]

    contains_code = int(
        any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in code_patterns
        )
    )

    # FINAL FEATURES
    features = {
        "Raw_Text": text,
        "Word_Count": word_count,
        "Character_Count_Total": character_count,
        "Sentence_Count": sentence_count,
        "Average_Word_Length": average_word_length,
        "Average_Sentence_Length_Words": average_sentence_length,
        "Unique_Words_Count": unique_words_count,
        "Lexical_Diversity_Score": lexical_diversity,
        "Punctuation_Count": punctuation_count,
        "Punctuation_Density": punctuation_density,
        "Special_Char_Count": special_char_count,
        "Stopword_Count": stopword_count,
        "Stopword_Density": stopword_density,
        "Flesch_Kincaid_Grade_Level": flesch_kincaid,
        "Smog_Index": smog_index,
        "Average_Syllables_Per_Word": average_syllables,
        "Contains_Code_Snippet": contains_code,
        "Capital_Letter_Count": capital_letter_count,
        "Burstiness_Score": burstiness_score,
    }

    return pd.DataFrame([features])


if __name__ == "__main__":
    # Quick test
    sample_text = (
        "Artificial intelligence is changing modern technology. "
        "It can automate repetitive tasks and help people analyze data."
    )

    result = extract_text_features(sample_text)
    print(result.to_string(index=False))
