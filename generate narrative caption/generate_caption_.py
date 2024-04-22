import os
from PIL import Image
from get_signals import find_signals
from get_probabilities import probs

def caption_generator(genders, loc, othersSen, action):
    path_to_data_folder = ""

    captions = []
    for i in range(len(genders)):

        if "girl" in genders[i] or "woman" in genders[i] or "female" in genders[i]:
            caption = "This person is " + genders[i] + " who " + action[i] + " in a " + loc[i].replace("in a"," at a") + "."
            for j in othersSen[i].split("$"):
                if j == "": pass
                else:
                    caption += " She " + j + "."

        else:
            caption = "This person is " + genders[i] + " who " + action[i] + " in a "+ loc[i].replace("in a"," at a") + "."

            for j in othersSen[i].split("$"):
                if j == "": pass
                else:
                    caption += " He " + j + "."

        flag_mul = True

        captions.append(caption)

    return captions
