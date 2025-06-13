import json
import os
from Vocabularizer import Vocabularizer
from config import SAVE_PATH

final_vocabulary = Vocabularizer()

try:
    os.mkdir(SAVE_PATH)
    print(f"Folder {SAVE_PATH} created successfully.")
except:
    print("")

with open(SAVE_PATH + "/vocabulary.json", "w") as file:
    file.write(json.dumps(final_vocabulary))
