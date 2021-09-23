import argparse
from datetime import timedelta
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_dict_path', required=True, type=Path)
    
    args = parser.parse_args()
    return args


def interactive_plot(detection_dict, key, ch_name = None):
    matplotlib.use('tkagg')

    def update_annot(ind):
        x, y = line.get_data()
        annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
        text = "Threshold {}: {}\n Precision: {}\n Recall: {}\n Time in alarm: {}%\n Advance/Delay: {} h\n TP: {} \n FP: {} \n FN: {}".format(" ".join([str(n) for n in ind["ind"]]),
                                                                                                                                              " ".join([str(th_list[n]) for n in ind["ind"]]),
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

    tpr = np.zeros(len(detection_dict[key]))
    ppv = np.zeros(len(detection_dict[key]))
    time_alrm_ar = np.zeros(len(detection_dict[key]))
    advance_delay_ar = [None for i in range(len(detection_dict[key]))]
    print(advance_delay_ar)
    print(len(advance_delay_ar))
    th_list = np.zeros(len(detection_dict[key]))
    TP_list = np.zeros(len(detection_dict[key]))
    FP_list = np.zeros(len(detection_dict[key]))
    FN_list = np.zeros(len(detection_dict[key]))

    for i, th in enumerate(detection_dict[key].keys()):
        th_list[i] = th
        TP_list[i] = detection_dict[key][th]["TP"]
        FP_list[i] = detection_dict[key][th]["FP"]
        FN_list[i] = detection_dict[key][th]["FN"]
        tpr[i] = detection_dict[key][th]["TPR"]
        ppv[i] = detection_dict[key][th]["PPV"]
        time_alrm_ar[i] = round(detection_dict[key][th]["TIME_ALARM"], 3)
        adv_del = round(detection_dict[key][th]["ADVANCE_DELAY"]) if not np.isnan(detection_dict[key][th]["ADVANCE_DELAY"]) else np.nan
        if np.isnan(adv_del):
            adv_del_time = np.nan
        else:
            if adv_del >= 0:
                adv_del_time = str(timedelta(minutes=adv_del))
            else:
                adv_del_time = "-"+str(timedelta(minutes=abs(adv_del)))
        advance_delay_ar[i] = adv_del_time

    fig, ax = plt.subplots()
    if ch_name is not None:
        ch_str, co_str, hy_str = key.split("_")
        ch = ch_str[2:]
        ch_name_idx = [i for i, s in enumerate(ch_name) if str(ch) in s][0]
        plt.title('TPR/PPV ' + 'CH ' + ch_name[ch_name_idx] + ' ' + co_str + ' ' + hy_str)

    line, = plt.plot(tpr, ppv, 'b', marker="o")

    annot = ax.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    fig.canvas.mpl_connect("button_press_event", hover)
    plt.show()


if __name__ == '__main__':
    args = parse()
    detection_dict = torch.load(args.det_dict_path)

    while input("Do you want an interactive plot for a specific key (channel, consecutive outliers, hysteresis)? y/n ") == "y":
        ch = input("Insert Channel: ")
        co = input("Insert Consecutive Outliers: ")
        hy = input("Insert Hysteresis: ")
        key = "CH" + str(ch) + "_CO" + str(co) + "_HY" + str(hy)
        if key in detection_dict:
            interactive_plot(detection_dict, key)
        else:
            print("Key: " + key + " not in detection_dict")
