from pathlib import Path
import pandas as pd
import torch
from datetime import timedelta
import numpy as np
import argparse


def get_parser_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--detection_path", type=str, default="./logs", help="Folder where to find the detection dict")
    parser.add_argument("--voting", action="store_true", help="Choose whether to analyze voting detection (True) or non voting (False)")

    args = parser.parse_args()
    return args


def check_predictions(detection_dict, detection_path, excel_name):
    data = []
    for key in detection_dict.keys():
        for th in detection_dict[key].keys():
            TP = detection_dict[key][th]["TP"]
            FP = detection_dict[key][th]["FP"]
            FN = detection_dict[key][th]["FN"]
            TPR = detection_dict[key][th]["TPR"]
            FDR = detection_dict[key][th]["FDR"]
            PPV = detection_dict[key][th]["PPV"]
            F1 = detection_dict[key][th]["F1"]
            F05 = detection_dict[key][th]["F05"]
            time_alrm_ar = round(detection_dict[key][th]["TIME_ALARM"], 3)
            adv_del = round(detection_dict[key][th]["ADVANCE_DELAY"]) if not np.isnan(detection_dict[key][th]["ADVANCE_DELAY"]) else np.nan
            if np.isnan(adv_del):
                adv_del_time = np.nan
            else:
                if adv_del >= 0:
                    adv_del_time = str(timedelta(minutes=adv_del))
                else:
                    adv_del_time = "-" + str(timedelta(minutes=abs(adv_del)))
            data.append([key, th, F1, F05, TPR, PPV, FDR, TP, FP, FN, time_alrm_ar, adv_del_time])

    df = pd.DataFrame(data, columns=["key", "TH", "F1", "F05", "TPR", "PPV", "FDR", "TP", "FP", "FN", "Time in alarm %", "Advance/Delay hh:mm:ss"])

    print("Saving prediction results in: " + str(Path(detection_path) / Path(excel_name)))
    df.to_excel(Path(detection_path) / Path(excel_name))


if __name__ == "__main__":
    args = get_parser_args()

    vot = "voting" if args.voting else "no_voting"

    det_dict = "detection_dict_" + vot + ".pt"
    excel_name = "detection_" + vot + ".xlsx"

    detection_dict = torch.load(Path(args.detection_path) / Path(det_dict))

    check_predictions(detection_dict, args.detection_path, excel_name)
