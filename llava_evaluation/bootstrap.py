import os
import csv
from sklearn.metrics import average_precision_score
import numpy as np
import math
import jsonlines
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, jaccard_score, classification_report
import random
import argparse


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="llava_input.jsonl", help="llava input file which we pass to it to get the output")
    parser.add_argument("--output", default="llava_output_finetuned_using_trainset.jsonl", type=str, help="llava output")
    parser.add_argument("--gt", default="gt_all.csv", type=str, help="ground truth labels from EMOTIC")
    parser.add_argument("--round", default=10000, type=int, help="number of rounds of sampling in bootstrap")
    parser.add_argument("--interval", default=95, type=int, help="interval percentage for bootstrap")
    args = parser.parse_args()
    return args

class bootstrap:
    def __init__(self, args, lbs):
        self.args = args
        self.lbs = lbs
    def evaluation(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        hamming = hamming_loss(y_true, y_pred)
        jaccard = jaccard_score(y_true, y_pred, average='macro')
        exact_match = accuracy_score(y_true, y_pred, normalize=True)
        return precision, recall, f1, hamming, exact_match

    def generate_result_and_gt_dict(self):
        res_dict = {}
        qid_img = {}
        gt_dict = {}
        img_list = []
        with jsonlines.open(self.args.input) as f:
            for line in f.iter():
                qid_img[line["question_id"]] = line['image']
                img_list.append(line['image'])

        with jsonlines.open(self.args.output) as f:
            for line in f.iter():
                res_dict[qid_img[line["question_id"]]] = line["text"]

        with open(self.args.gt) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                gt_dict[row[0]] = row[1].replace("[","").replace("]","").replace(" ","").replace("'", "").split(',')

        self.res_dict, self.gt_dict, self.img_list =  res_dict, gt_dict, img_list


    def bootstraping(self):
        self.generate_result_and_gt_dict()
        all_p = []
        all_r = []
        all_f1 = []
        all_hamming = []
        all_exact_match = []
        for rnd in range(self.args.round):
            new_list = random.choices(self.img_list, k=len(self.img_list))
            gt_np = np.zeros((len(self.res_dict), len(self.lbs)))
            pred = np.zeros((len(self.res_dict), len(self.lbs)))
            im_num = -1
            for i in new_list:
                im_num += 1
                for j in range(len(self.lbs)):
                    if self.lbs[j] in self.res_dict[i] :
                        pred[im_num, j] = 1
                    if self.lbs[j] in self.gt_dict[i]: gt_np[im_num, j] = 1
            precision, recall, f1, hamming, exact_match = self.evaluation(gt_np, pred)
            all_p.append(precision)
            all_r.append(recall)
            all_f1.append(f1)
            all_hamming.append(hamming)
            all_exact_match.append(exact_match)

        all_p_sorted = sorted(all_p)
        all_r_sorted = sorted(all_r)
        all_f1_sorted = sorted(all_f1)
        all_hamming_sorted = sorted(all_hamming)
        all_exact_match_sorted = sorted(all_exact_match)
        lower = 0.5 * (((100 - self.args.interval) * self.args.round)/100)
        upper = self.args.round - lower
        lower, upper = int(lower), int(upper)
        print(lower, upper)
        p_cutted = list(all_p_sorted[lower:upper])
        r_cutted = list(all_r_sorted[lower:upper])
        f1_cutted = list(all_f1_sorted[lower:upper])
        hamming_cutted = list(all_hamming_sorted[lower:upper])
        exact_match_cutted = list(all_exact_match_sorted[lower:upper])
        return p_cutted, r_cutted, f1_cutted, hamming_cutted, exact_match_cutted



if __name__=="__main__":

    lbs = ["suffering", "pain", "sadness", "aversion", "disapproval", "anger", "fear", "annoyance", "fatigue", "disquietment","doubt/confusion", "embarrassment","disconnection","affection", "confidence", "engagement", "happiness", "peace", "pleasure", "esteem","excitement", "anticipation", "yearning", "sensitivity", "surprise", "sympathy"]
    args = config()
    bootstrap_ = bootstrap(args, lbs)
    p_cutted, r_cutted, f1_cutted, hamming_cutted, exact_match_cutted = bootstrap_.bootstraping()


    print("precision confidence_interval= ",p_cutted[-1] ,"minus", p_cutted[0] , "=",p_cutted[-1] - p_cutted[0])
    print("recall confidence_interval= ",r_cutted[-1] ,"minus", r_cutted[0] , "=",r_cutted[-1] - r_cutted[0])
    print("f1 confidence_interval= ",f1_cutted[-1] ,"minus", f1_cutted[0] , "=", f1_cutted[-1] - f1_cutted[0])
    print("hamming confidence_interval= ",hamming_cutted[-1] ,"minus", hamming_cutted[0] , "=",hamming_cutted[-1] - hamming_cutted[0])
    print("exact_match confidence_interval= ",exact_match_cutted[-1] ,"minus", exact_match_cutted[0] , "=",exact_match_cutted[-1] - exact_match_cutted[0])

print("---------------------------------------")

print("precision SE:  ", np.std(p_cutted))
print("recall SE:  ", np.std(r_cutted))
print("f1 SE:  ", np.std(f1_cutted))
print("hamming SE:  ", np.std(hamming_cutted))
print("exact_match SE:  ", np.std(exact_match_cutted))
