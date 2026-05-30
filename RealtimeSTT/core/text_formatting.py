"""Internal text formatting helpers for recorder output."""

import re


def format_number(num):
    # Convert the number to a string
    num_str = f"{num:.10f}"  # Ensure precision is sufficient
    # Split the number into integer and decimal parts
    integer_part, decimal_part = num_str.split('.')
    # Take the last two digits of the integer part and the first two digits of the decimal part
    result = f"{integer_part[-2:]}.{decimal_part[:2]}"
    return result


def preprocess_output(
    text,
    preview=False,
    ensure_sentence_starting_uppercase=True,
    ensure_sentence_ends_with_period=True,
):
    """
    Preprocesses the output text by removing any leading or trailing
    whitespace, converting all whitespace sequences to a single space
    character, and capitalizing the first character of the text.

    Args:
        text (str): The text to be preprocessed.

    Returns:
        str: The preprocessed text.
    """
    text = re.sub(r'\s+', ' ', text.strip())

    if ensure_sentence_starting_uppercase:
        if text:
            text = text[0].upper() + text[1:]

    # Ensure the text ends with a proper punctuation
    # if it ends with an alphanumeric character
    if not preview:
        if ensure_sentence_ends_with_period:
            if text and text[-1].isalnum():
                text += '.'

    return text


def find_tail_match_in_text(text1, text2, length_of_match=10):
    """
    Find the position where the last 'n' characters of text1
    match with a substring in text2.

    This method takes two texts, extracts the last 'n' characters from
    text1 (where 'n' is determined by the variable 'length_of_match'), and
    searches for an occurrence of this substring in text2, starting from
    the end of text2 and moving towards the beginning.

    Parameters:
    - text1 (str): The text containing the substring that we want to find
      in text2.
    - text2 (str): The text in which we want to find the matching
      substring.
    - length_of_match(int): The length of the matching string that we are
      looking for

    Returns:
    int: The position (0-based index) in text2 where the matching
      substring starts. If no match is found or either of the texts is
      too short, returns -1.
    """

    # Check if either of the texts is too short
    if len(text1) < length_of_match or len(text2) < length_of_match:
        return -1

    # The end portion of the first text that we want to compare
    target_substring = text1[-length_of_match:]

    # Loop through text2 from right to left
    for i in range(len(text2) - length_of_match + 1):
        # Extract the substring from text2
        # to compare with the target_substring
        current_substring = text2[len(text2) - i - length_of_match:
                                  len(text2) - i]

        # Compare the current_substring with the target_substring
        if current_substring == target_substring:
            # Position in text2 where the match starts
            return len(text2) - i

    return -1
