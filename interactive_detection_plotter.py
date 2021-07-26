import argparse
from datetime import timedelta
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import torch


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_dict_path', required=True, type=Path)
    
    args = parser.parse_args()
    return args


def update_annot(ind):
    x, y = line.get_data()
    annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
    text = "Threshold {}: {}\n F1 Score: {}\n Precision: {}\n Recall: {}\n Time in alarm: {}%\n Advance/Delay: {} h\n TP: {} \n FP: {} \n FN: {}".format(" ".join([str(n-1) for n in ind["ind"]]), 
                                                                                                                                                        " ".join([str(th_list[n]) for n in ind["ind"]]),
                                                                                                                                                         " ".join([str(round(f1[n], 2)) for n in ind["ind"]]),
                                                                                                                                                         " ".join([str(round(ppv[n], 2)) for n in ind["ind"]]),
                                                                                                                                                         " ".join([str(round(tpr[n], 2)) for n in ind["ind"]]),
                                                                                                                                                         " ".join([str(time_alrm_ar[n]) for n in ind["ind"]]), 
                                                                                                                                                         " ".join([str(advance_delay_ar[n]) for n in ind["ind"]]),
                                                                                                                                                         " ".join([str(TP_list[n]) for n in ind["ind"]]),
                                                                                                                                                         " ".join([str(FP_list[n]) for n in ind["ind"]]),
                                                                                                                                                         " ".join([str(FN_list[n]) for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = line.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()


if __name__ == '__main__':

    matplotlib.use('tkagg')

    args = parse()
    detection_dict = torch.load(args.det_dict_path)

    while input("Do you want an interactive plot for a specific key (channel, consecutie outliers, hysteresis)? y/n ") == "y":

        ch = input("Insert Channel: ")
        co = input("Insert Consecutive Outliers: ")
        hy = input("Insert Hysteresis: ")
        key = "CH"+str(ch)+"_CO"+str(co)+"_HY"+str(hy)

        tpr = np.zeros(len(detection_dict[key]))
        ppv = np.zeros(len(detection_dict[key]))
        f1 = np.zeros(len(detection_dict[key]))
        time_alrm_ar = np.zeros(len(detection_dict[key]))
        advance_delay_ar = list()
        th_list = list()
        TP_list = list()
        FP_list = list()
        FN_list = list()

        for i, th in enumerate(detection_dict[key].keys()):
            TP = detection_dict[key][th]["TP"]
            FP = detection_dict[key][th]["FP"]
            FN = detection_dict[key][th]["FN"]

            tpr[i] = TP/(TP+FN)
            ppv[i] = TP/(TP+FP) if (TP+FP) != 0 else 0
            f1[i] = TP/(TP+(FP+FN)/2)
            time_alrm_ar[i] = round(detection_dict[key][th]["TIME_ALARM"], 3)
            adv_del = round(detection_dict[key][th]["ADVANCE_DELAY"]) if not np.isnan(detection_dict[key][th]["ADVANCE_DELAY"]) else np.nan
            if np.isnan(adv_del):
                adv_del_time = np.nan
            else:
                if adv_del >= 0:
                    adv_del_time = timedelta(minutes=adv_del)
                else:
                    adv_del_time = "-"+str(timedelta(minutes=abs(adv_del)))
            advance_delay_ar.append(adv_del_time)
            th_list.append(th)
            TP_list.append(TP)
            FP_list.append(FP)
            FN_list.append(FN)

        ppv_tpr_auc = metrics.auc(tpr, ppv)

        fig, ax = plt.subplots()
        plt.title('TPR/PPV ' + key)
        line, = plt.plot(tpr, ppv, 'b', label='AUC = %0.2f' % ppv_tpr_auc, marker="o")

        annot = ax.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        plt.legend()
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        fig.canvas.mpl_connect("button_press_event", hover)
        plt.show()
