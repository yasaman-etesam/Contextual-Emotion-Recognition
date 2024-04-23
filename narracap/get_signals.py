import numpy as np

def find_signals(relString,signals,image_dir,probs, more_than_one):

  signalHighest = []
  num_of_imgs = probs.shape[0]
  for i in range(num_of_imgs):
    meanVal = np.mean(probs[i]) # equal to 100/len(probs[0])
    std = np.std(probs[i])
    maxVal = 0
    jMax = 0

    sent = ""

    if more_than_one:
      threshold = meanVal+9.0*std
      for j in range(len(probs[i])):
        if probs[i][j] > threshold:
          sent += signals[j] + "$"
      signalHighest.append(sent)
      # else:
      #    signalHighest.append("")

      if signalHighest == []:
        signalHighest.append("")

    else:
      threshold = 0
      # if len(probs[0]) == 2: threshold = 0
      # else: threshold = meanVal + 3 * std

      for j in range(len(probs[i])):
        if probs[i][j] >= maxVal:
          maxVal = probs[i][j]
          jMax = j
      if probs[i][jMax] >= threshold:
        signalHighest.append(signals[jMax])


      else:
        signalHighest.append("")



  return signalHighest
