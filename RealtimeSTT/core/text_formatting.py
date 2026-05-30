"""Internal text formatting helpers for recorder output."""

import re


def format_number(num):
    """
    Formats a number as the legacy two-digit timestamp fragment.
    """
    num_str = f"{num:.10f}"  # Ensure precision is sufficient
    integer_part, decimal_part = num_str.split('.')
    result = f"{integer_part[-2:]}.{decimal_part[:2]}"
    return result


def preprocess_output(
    text,
    preview=False,
    ensure_sentence_starting_uppercase=True,
    ensure_sentence_ends_with_period=True,
):
    """
    Normalizes recorder output text for display.

    Args:
    - text: Text to normalize.
    - preview: Leaves final punctuation unchanged for preview text.
    - ensure_sentence_starting_uppercase: Uppercases the first character.
    - ensure_sentence_ends_with_period: Adds sentence punctuation when needed.
    """
    text = re.sub(r'\s+', ' ', text.strip())

    if ensure_sentence_starting_uppercase:
        if text:
            text = text[0].upper() + text[1:]

    if not preview:
        if ensure_sentence_ends_with_period:
            if text and text[-1].isalnum():
                text += '.'

    return text


def find_tail_match_in_text(text1, text2, length_of_match=10):
    """
    Finds where the tail of one text appears in another text.

    Args:
    - text1: Text whose trailing characters should be matched.
    - text2: Text to search from the end.
    - length_of_match: Number of trailing characters to compare.

    Returns:
        Start position in text2, or -1 when no tail match exists.
    """

    if len(text1) < length_of_match or len(text2) < length_of_match:
        return -1

    target_substring = text1[-length_of_match:]

    for i in range(len(text2) - length_of_match + 1):
        current_substring = text2[len(text2) - i - length_of_match:
                                  len(text2) - i]

        if current_substring == target_substring:
            return len(text2) - i

    return -1
