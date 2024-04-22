import torch
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_id)
tokenizer = CLIPTokenizerFast.from_pretrained(model_id)

model = CLIPModel.from_pretrained(model_id)

def changeString(signalsList,relString):
  signals = []
  for i in range(len(signalsList)):
    signals.append(relString+ " " + signalsList[i])
  return signals

def probs(part, relString, signals, image_list):
  full_im = []
  crop_im = []
  full_im_no_bb = []
  newList = changeString(signals,relString)
  pth = []


  for j in image_list:

    full_im.append(j[0])
    crop_im.append(j[1])
    full_im_no_bb.append(j[2])
    pth.append(j[3])

  if part == "location":
    inputs = processor(text=newList, images=full_im_no_bb, return_tensors="pt", padding=True)
  elif part == "physicals" or part == "gender":
    inputs = processor(text=newList, images=crop_im, return_tensors="pt", padding=True)
  elif part == "action":
    inputs = processor(text=newList, images=full_im_no_bb, return_tensors="pt", padding=True)


  outputs = model(**inputs)
  logits_per_image = outputs.logits_per_image
  probs_ = (logits_per_image.softmax(dim=1).detach().numpy()*100)


  return probs_
