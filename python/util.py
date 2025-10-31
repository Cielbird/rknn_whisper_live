"""
Utility functions for live whisper
"""
import numpy as np


def detect_any_repetition_loop(tokens: list[int], max_seq_len: int, min_seq_reps: int) -> bool:
    """
    Checks for repeated sequences at the end of the tokens list
    """
    for seq_len in range(1, max_seq_len + 1):
        if detect_repetition_loop(tokens, seq_len, min_seq_reps):
            return True
    return False

def detect_repetition_loop(tokens: list[int], seq_len: int, min_seq_reps: int) -> bool:
    """
    Checks for a repeated sequence of a fixed size at the end of the tokens list
    """
    return False # TODO remove this, deal with this: do i keep it or not
    num_tokens = len(tokens)
    for i in range(seq_len):
        char = None
        for rep in range(min_seq_reps):
            idx = num_tokens - 1 - (rep * seq_len) - i
            if not char:
                char = tokens[idx]
            elif char != tokens[idx]:
                return False
    return True

def token_to_timestamp(token: int) -> float | None:
    """Converts a timestamp token to the seconds as a float."""
    # 50364 is the |<0.00>| timestamp. they are spaced 20ms apart.
    if token < 50364:
        return None
    return (token -  50364) * 0.02


def get_char_index(c):
    """
    Returns the 6-bit integer index corresponding to a Base64 character.

    Args:
        c (str): A single Base64 character ('A'-'Z', 'a'-'z', '0'-'9', '+', or '/').

    Returns:
        int: The integer value (0–63) representing the given Base64 character.

    Raises:
        SystemExit: If the character is not a valid Base64 symbol.
    """
    if "A" <= c <= "Z":
        return ord(c) - ord("A")
    elif "a" <= c <= "z":
        return ord(c) - ord("a") + (ord("Z") - ord("A") + 1)
    elif "0" <= c <= "9":
        return ord(c) - ord("0") + (ord("Z") - ord("A")) + (ord("z") - ord("a")) + 2
    elif c == "+":
        return 62
    elif c == "/":
        return 63
    else:
        print(f"Unknown character {ord(c)}, {c}")
        exit(-1)


def base64_decode(encoded_string):
    """
    Decodes a Base64-encoded string into its original UTF-8 representation.

    Args:
        encoded_string (str): The Base64-encoded string to decode.

    Returns:
        str: The decoded UTF-8 string. Returns a single space (" ") if padding ('=') is encountered.

    Raises:
        SystemExit: If the input string is empty.

    Note:
        This function assumes the existence of a helper function `get_char_index(char)`
        that maps a Base64 character to its corresponding 6-bit integer value.
    """

    if not encoded_string:
        print("Empty string!")
        exit(-1)

    output_length = len(encoded_string) // 4 * 3
    decoded_string = bytearray(output_length)

    index = 0
    output_index = 0
    while index < len(encoded_string):
        if encoded_string[index] == "=":
            return " "

        first_byte = (get_char_index(encoded_string[index]) << 2) + (
            (get_char_index(encoded_string[index + 1]) & 0x30) >> 4
        )
        decoded_string[output_index] = first_byte

        if index + 2 < len(encoded_string) and encoded_string[index + 2] != "=":
            second_byte = ((get_char_index(encoded_string[index + 1]) & 0x0F) << 4) + (
                (get_char_index(encoded_string[index + 2]) & 0x3C) >> 2
            )
            decoded_string[output_index + 1] = second_byte

            if index + 3 < len(encoded_string) and encoded_string[index + 3] != "=":
                third_byte = (
                    (get_char_index(encoded_string[index + 2]) & 0x03) << 6
                ) + get_char_index(encoded_string[index + 3])
                decoded_string[output_index + 2] = third_byte
                output_index += 3
            else:
                output_index += 2
        else:
            output_index += 1

        index += 4

    return decoded_string.decode("utf-8", errors="replace")


def read_vocab(vocab_path: str) -> dict:
    """
    Reads the vocab for a whisper model
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = {}
        for line in f:
            if len(line.strip().split(" ")) < 2:
                key = line.strip().split(" ")[0]
                value = ""
            else:
                key, value = line.strip().split(" ")
            vocab[key] = value
    return vocab

def load_array_from_file(filename: str) -> np.array:
    """
    Read a np.array from a file
    """
    with open(filename, "r", encoding="utf-8") as file:
        data = file.readlines()

    array = []
    for line in data:
        row = [float(num) for num in line.split()]
        array.extend(row)

    return np.array(array).reshape((80, 2000))
