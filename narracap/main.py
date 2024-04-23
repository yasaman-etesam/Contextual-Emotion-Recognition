from PIL import Image
import os
from sigs import signals
import csv
from get_signals import find_signals
from get_probabilities import probs
from generate_caption_ import caption_generator
import argparse





def get_imgs_path(path_to_emotic, path_to_data_folder, path_to_cropped_folder):
  img_paths = []
  img_paths_crop = []
  count = 0
  for path in os.listdir(path_to_data_folder):
    # check if current path is a file
    for i in os.listdir(os.path.join(path_to_data_folder, path)):
      if os.path.isfile(os.path.join(path_to_data_folder, path, i)):
        count += 1
        emot_jj = i[:-6]
        if emot_jj[-1] == "_":
            emot_jj = emot_jj[:-1]
        if not os.path.exists(os.path.join(path_to_emotic, path, "images", emot_jj + ".jpg")):
            print(os.path.join(path_to_emotic, path, "images", emot_jj + ".jpg"))
            error
        img_paths.append((os.path.join(path_to_data_folder, path, i), os.path.join(path_to_cropped_folder, path, i), os.path.join(path_to_emotic, path, "images", emot_jj+".jpg")))
        # img_paths_crop.append(os.path.join(path_to_cropped_folder, path, i))

  return count, img_paths


def narracap(args):
    new_set_800, gender, locContext, kinetic = signals()

    path_to_data_folder = args.image_BB
    path_to_cropped_folder = args.image_cropped
    path_to_emotic = args.emotic

    count, img_path= get_imgs_path(path_to_emotic, path_to_data_folder, path_to_cropped_folder)

    num_of_input_images = count


    print(num_of_input_images)
    print(len(img_path))

    imgs_vec = []
    for i in img_path[:num_of_input_images]:

        im1 = Image.open(i[0])
        im2 = Image.open(i[1])
        im3 = Image.open(i[2])
        im1c = im1.copy()
        im2c = im2.copy()
        im3c = im3.copy()
        imgs_vec += [(im1c, im2c, im3c, i[1])]
        im1.close()
        im2.close()
        im3.close()


    dict = {}

    phys_sigs = new_set_800

    print(len(new_set_800))

    start = 0
    while (start<len(imgs_vec)):
        if start + 1000 < len(imgs_vec):
            finish = start + 1000
        else:
            finish = len(imgs_vec)

        imgs_vec_ = imgs_vec[start:finish]

        print("getting gender and age info...")

        genderProb = probs("gender", "A photo of", gender, imgs_vec_)
        ####genderProb = probs("gender", "The person in the image", gender, imgs_vec)
        genderSen = find_signals("The person in the image",gender,path_to_data_folder,genderProb, False)

        print("getting location info...")

        ########locProb = probs("location", "This image is happening",locContext, imgs_vec)
        locProb = probs("location", "A photo of a",locContext, imgs_vec_)
        locations = find_signals("This image is happening",locContext,path_to_data_folder,locProb, False)

        print("predictig activity...")

        actionProb = probs("action", "A photo of a person who", kinetic , imgs_vec_)
        #######actionProb = probs("action", "The person in this image", kinetic , imgs_vec)
        actions = find_signals("The person in this image", kinetic ,path_to_data_folder, actionProb, False)

        print("Predicting physical signals...")
        othersProb = probs("physicals", "A photo of a person who" ,phys_sigs,imgs_vec_)
        #othersProb = probs("physicals", "The person in the image" ,phys_sigs,imgs_vec)
        social_signals = find_signals("The person in the image",phys_sigs,path_to_data_folder,othersProb, True)


        print("Generating captions...")
        captions = caption_generator(genderSen, locations, social_signals, actions)


        print(len(captions))
        for i in range(len(captions)):
            dict[img_path[start+i][0]] = captions[i]

        start = start + 1000
    print(num_of_input_images)
    with open("naracaps.csv",'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in dict:
            writer.writerow([i, dict[i]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_BB', type=str, default='/localhome/yetesam/Desktop/apr_13/may15_test', help='Path to the folder containing the images with red bunding box of target')
    parser.add_argument('--image_cropped', type=str, default='/localhome/yetesam/Desktop/apr_13/may15_test_bb', help='Path to the folder containg the cropped bounding boxes of target individuals')
    parser.add_argument('--emotic', type=str, default='/localhome/yetesam/Desktop/emotic-dataset/emotic/', help='Path to emotic dataset')
    args = parser.parse_args()

    narracap(args)
