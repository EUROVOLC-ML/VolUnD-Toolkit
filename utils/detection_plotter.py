import argparse
import os
from datetime import timedelta
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

matplotlib.use('tkagg')


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_dict_path', required=True, type=Path,
                        help='The path for detection dict')
    parser.add_argument('--save_plots', action="store_true",
                        help='Whether to save plots')
    
    args = parser.parse_args()
    return args


def update_annot(ind, line, annot, th_list, ppv, tpr, time_alrm_ar, advance_delay_ar, TP_list, FP_list, FN_list):
    x, y = line.get_data()
    annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
    text = "Threshold {}: {}\n Precision: {}\n Recall: {}\n Time in alarm: {}%\n Advance/Delay: {} h\n TP: {} \n FP: {} \n FN: {}".format(
        " ".join([str(n) for n in ind["ind"]]),
        " ".join([str(round(th_list[n], 3)) for n in ind["ind"]]),
        " ".join([str(round(ppv[n], 3)) for n in ind["ind"]]),
        " ".join([str(round(tpr[n], 3)) for n in ind["ind"]]),
        " ".join([str(time_alrm_ar[n]) for n in ind["ind"]]),
        " ".join([str(advance_delay_ar[n]) for n in ind["ind"]]),
        " ".join([str(TP_list[n]) for n in ind["ind"]]),
        " ".join([str(FP_list[n]) for n in ind["ind"]]),
        " ".join([str(FN_list[n]) for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event, annot, ax, line, fig, th_list, ppv, tpr, time_alrm_ar, advance_delay_ar, TP_list, FP_list, FN_list):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = line.contains(event)
        if cont:
            update_annot(ind, line, annot, th_list, ppv, tpr, time_alrm_ar, advance_delay_ar, TP_list, FP_list, FN_list)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()


def interactive_plot(detection_dict, ch_name=None):
    while input("Do you want an interactive plot for a specific key (channel, consecutive outliers, hysteresis)? y/n ") == "y":
        ch = input("Insert Channel: ")
        co = input("Insert Consecutive Outliers: ")
        hy = input("Insert Hysteresis: ")
        key = "CH" + str(ch) + "_CO" + str(co) + "_HY" + str(hy)
        if key in detection_dict:
            tpr = np.zeros(len(detection_dict[key]))
            ppv = np.zeros(len(detection_dict[key]))
            time_alrm_ar = np.zeros(len(detection_dict[key]))
            advance_delay_ar = [None] * len(detection_dict[key])
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
            else:
                plt.title('TPR/PPV ' + key)
            line, = plt.plot(tpr, ppv, 'b', marker="o")
            annot = ax.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w"),
                                arrowprops=dict(arrowstyle="->"))
            annot.set_visible(False)
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            fig.canvas.mpl_connect("button_press_event", lambda event: hover(event, annot, ax, line, fig, th_list, ppv, tpr, time_alrm_ar, advance_delay_ar, TP_list, FP_list, FN_list))
            plt.show()
        else:
            print("Key: " + key + " not in detection_dict")


def plot(det_dict_path, ch_name=None):
    detection_dict = torch.load(det_dict_path)
    if "VOT" in list(detection_dict.keys())[0]:
        plot_dir = Path(det_dict_path).parent.absolute() / Path("detectionPlot_voting")
        vote = True
    else:
        plot_dir = Path(det_dict_path).parent.absolute() / Path("detectionPlot_no_voting")
        vote = False
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    if vote is False:
        for key in tqdm(detection_dict.keys(), desc="Plotting"):
            tpr = np.zeros(len(detection_dict[key]))
            ppv = np.zeros(len(detection_dict[key]))
            for i, th in enumerate(detection_dict[key].keys()):
                tpr[i] = detection_dict[key][th]["TPR"]
                ppv[i] = detection_dict[key][th]["PPV"]

            plt.figure()
            if ch_name is not None:
                ch_str, co_str, hy_str = key.split("_")
                ch = ch_str[2:]
                ch_name_idx = [i for i, s in enumerate(ch_name) if str(ch) in s][0]
                plt.title('TPR/PPV ' + 'CH ' + ch_name[ch_name_idx] + ' ' + co_str + ' ' + hy_str)
            else:
                plt.title('TPR/PPV ' + key)
            plt.plot(tpr, ppv, 'b', marker="o")
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.savefig(os.path.join(plot_dir, key + ".png"), dpi=300)
            plt.close()
    elif vote is True:
        print("Plotting...")
        tpr = np.zeros(len(detection_dict))
        ppv = np.zeros(len(detection_dict))
        f1 = np.zeros(len(detection_dict))
        for i, key in enumerate(detection_dict.keys()):
            tpr[i] = detection_dict[key]["TPR"]
            ppv[i] = detection_dict[key]["PPV"]
            f1[i] = detection_dict[key]["F1"]

        ch_str, co_str, hy_str, th = key.split("_")
        plt.figure()
        plt.title('TPR/PPV ' + ' ' + co_str + ' ' + hy_str)
        plt.plot(tpr, ppv, 'b', marker="o")
        for i in range(len(detection_dict)):
            plt.annotate("VOT" + str(i+1), (tpr[i], ppv[i]))
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(os.path.join(plot_dir, "TPR_PPV_voting.png"), dpi=300)
        plt.close()

        plt.figure()
        plt.title('F1/Vote ' + ' ' + co_str + ' ' + hy_str)
        plt.plot(range(1, len(f1)+1), f1, 'b', marker="o")

        for i in range(len(detection_dict)):
            plt.annotate("VOT" + str(i+1), (i+1, f1[i]))

        plt.ylim([-0.01, 1.01])
        plt.xlabel('Vote')
        plt.ylabel('F1')
        plt.savefig(os.path.join(plot_dir, "VOT_F1_voting.png"), dpi=300)
        plt.close()

    plt.close('all')


if __name__ == '__main__':
    args = parse()
    detection_dict = torch.load(args.det_dict_path)

    if args.save_plots is True:
        plot(args.det_dict_path)

    if "VOT" in list(detection_dict.keys())[0]:
        print("Interactive plotter works only without voting mechanism")
    else:
        interactive_plot(detection_dict)
