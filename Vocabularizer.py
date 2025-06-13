from config import VOCAB_PATH, CODE_LENGTH, CODE_LENGTH2, ZERO, VOCAB_SIZE, EMPTY_TOKEN, UNK_TOKEN

def Vocabularizer():
    text = open(VOCAB_PATH).read().lower()

    vocabulary = []

    char_array = list(text)
    chars_unique = list(set(char_array))
    vocabulary.extend(chars_unique)

    encoded_array = []
    for char in char_array:
        encoded = str(chars_unique.index(char))
        encoded = "0" * (CODE_LENGTH - len(encoded)) + encoded
        encoded_array.append(encoded)
    encoded_string = "".join(encoded_array)

    next_replacement = 100
    replaced_pairs = []
    replacements = []

    print(len(encoded_string))

    # Not really vocab size, since only top-level replacements survive
    while len(replacements) < VOCAB_SIZE:
        pairs = []
        scores = []

        for i in range(int(len(encoded_string) / CODE_LENGTH) - 1):
            pair = encoded_string[CODE_LENGTH*i : CODE_LENGTH*(i+2)]

            if pair[0 : CODE_LENGTH] == ZERO or pair[CODE_LENGTH : CODE_LENGTH2] == ZERO:
                continue

            if pair in pairs:
                scores[pairs.index(pair)] += 1
            else:
                pairs.append(pair)
                scores.append(1)

        highest_pair = pairs[scores.index(max(scores))]
        replacement = str(next_replacement)
        next_replacement += 1

        replaced_pairs.append(highest_pair)
        replacements.append(replacement)

        new_encoded_array = []
        i = 0
        while i < int(len(encoded_string) / CODE_LENGTH) - 1:
            if encoded_string[CODE_LENGTH*i : CODE_LENGTH*(i+2)] == highest_pair:
                new_encoded_array.append(replacement)
                i += 2
            else:
                new_encoded_array.append(encoded_string[CODE_LENGTH*i : CODE_LENGTH*(i+1)])
                i += 1

        encoded_string = "".join(new_encoded_array)

    print(len(encoded_string))

    def replaced_back(replacement):
        if replacement not in replacements:
            return replacement
        replaced_pair = replaced_pairs[replacements.index(replacement)]
        return replaced_back(replaced_pair[0 : CODE_LENGTH]) + replaced_back(replaced_pair[CODE_LENGTH : CODE_LENGTH2])

    for replacement in replacements:
        # Check if top-level
        top = True
        for replaced_pair in replaced_pairs:
            if replacement == replaced_pair[0 : CODE_LENGTH] or replacement == replaced_pair[CODE_LENGTH : CODE_LENGTH2]:
                top = False
                break

        if top:
            expanded = replaced_back(replacement)
            token = "".join([chars_unique[int(expanded[CODE_LENGTH*j : CODE_LENGTH*(j+1)])] for j in range(int(len(expanded) / CODE_LENGTH))])
            vocabulary.append(token)

    vocabulary.sort(key=len, reverse=True)

    vocabulary.append(EMPTY_TOKEN)
    vocabulary.append(UNK_TOKEN)

    return vocabulary

def Tokenize(text, vocabulary, num_tokens):
    # Remove all digits
    encoded_text = text.lower()
    # Vocabulary is sorted in descending length, so longer ones are replaced first, last two are EMPTY and UNK
    for i in range(len(vocabulary) - 2):
        encoded_text = encoded_text.replace(vocabulary[i], "B" + str(i) + "E")

    tokens = []
    current_read = ""
    state = 0
    # 0 = waiting, 1 = reading
    for i in range(len(encoded_text)):

        if state == 0:
            if encoded_text[i] == "B":
                state = 1
            else:
                tokens.append(UNK_TOKEN)

        if state == 1:
            if encoded_text[i].isdigit():
                current_read = current_read + encoded_text[i]
            elif encoded_text[i] == "E":
                token = vocabulary[int(current_read)]
                tokens.append(token)
                current_read = ""
                state = 0

    if num_tokens != None:
        if len(tokens) > num_tokens - 1:
            tokens = tokens[len(tokens) - (num_tokens - 1) : len(tokens)]

        elif len(tokens) < num_tokens - 1:
            tokens = [EMPTY_TOKEN] * (num_tokens - 1 - len(tokens)) + tokens

    return tokens
