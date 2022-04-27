import itertools
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm
from utils.check_predictions import check_predictions
from utils.dataset import Dataset
from utils.detection_plotter import interactive_plot, plot
from utils.model import Model
from utils.parser import check_detection_args, detection_parse
from utils.saver import Saver


def overlap(event, alarm):
    if alarm[1] < event[0] or alarm[0] > event[1]:
        return False
    else:
        return event[0]-alarm[0]


def remove_event_nan(ev_list, alm_list):
    ev_list_mod = list()
    for ev in ev_list:
        for al in alm_list:
            if np.isnan(al[2]):
                if ev[0] >= al[0] and ev[1] <= al[1]:
                    ev_list_mod.append(ev)
    return sorted(list(set(ev_list) - set(ev_list_mod)), key=lambda tup: tup[0])


if __name__ == '__main__':
    # Get params
    args = detection_parse()

    # Retrieve absolute path of checkpoint
    checkpoint = os.path.abspath(args['checkpoint'])

    # Load arguments
    hyperparams = Saver.load_hyperparams(checkpoint)
    checkpoint_dict = Saver.load_checkpoint(checkpoint)

    # Normalization
    normalize_params = {"mean": args['mean'], "std": args['std']}

    # Instantiate dataset
    detection_dataset = Dataset(args['detection_dir'],
                                data_location=args['data_location'],
                                chunk_len=args['chunk_len'],
                                chunk_only_one=args['chunk_only_one'],
                                chunk_rate=args['chunk_rate'],
                                chunk_random_crop=args['chunk_random_crop'],
                                data_sampling_frequency=args['data_sampling_frequency'],
                                chunk_linear_subsample=args['chunk_linear_subsample'],
                                chunk_butterworth_lowpass=args['chunk_butterworth_lowpass'],
                                chunk_butterworth_highpass=args['chunk_butterworth_highpass'],
                                chunk_butterworth_order=args['chunk_butterworth_order'],
                                normalize_params=normalize_params,
                                channels_list=args['channels_list'],
                                channels_name=args['channels_name'],
                                provider=args['data_provider'])

    # Instantiate loader
    detection_loader = data.DataLoader(detection_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0, drop_last=True)

    # Calculate sample len
    _, _, _, times = list(map(list, zip(*detection_dataset)))
    sample_len = int(min(i for i in [j-i for i, j in zip(times[:-1], times[1:])] if i > 0)/60)  # minutes

    args, labels_list, date_time_list = check_detection_args(args, detection_dataset.get_channels_name())

    # Setup model
    model = Model(data_len=int(hyperparams['chunk_len'] / hyperparams['chunk_linear_subsample']),
                  data_channels=hyperparams['data_channels'],
                  layers_base=hyperparams['layers_base'],
                  channels_base=hyperparams['channels_base'],
                  min_spatial_size=hyperparams['min_spatial_size'],
                  start_dilation=hyperparams['start_dilation'],
                  min_sig_dil_ratio=hyperparams['min_sig_dil_ratio'],
                  max_channels=hyperparams['max_channels'],
                  h_size=hyperparams['h_size'],
                  enable_variational=hyperparams['enable_variational'])
    model.load_state_dict(checkpoint_dict['model_state_dict'])
    model.eval()
    model.to(args['device'])

    # Model evaluation
    out = []
    with torch.no_grad():
        for sig, _, _, _ in tqdm(detection_loader, desc='Inferring'):
            rec, _, _ = model(sig.to(args['device']))
            out.append(rec.detach().cpu())

    # Group reconstructions
    outLIN = []
    outLABEL = []
    outTIMESTAMP = []
    for i, sig_batch in enumerate(tqdm(out, desc='Elaborating')):
        for j in range(sig_batch.shape[0]):  # batch
            tmp_sig = torch.zeros(sig_batch.shape[1:])
            for k in range(sig_batch.shape[1]):  # channel
                # Insert nan on reconstruction distance if signal is all 0 (station off)
                if detection_dataset[i*args['batch_size']+j][0][k].abs().max() != 0:
                    tmp_sig[k] = detection_dataset[i * args['batch_size']+j][0][k] - sig_batch[j, k]
                else:
                    tmp_sig[k] = np.nan
            outLIN.append(tmp_sig)
            outLABEL.append(detection_dataset[i*args['batch_size']+j][1])
            outTIMESTAMP.append(detection_dataset[i*args['batch_size']+j][2])
    outUNIONdiff = torch.stack(outLIN)
    outDATETIME = [datetime.fromtimestamp(t, timezone.utc) for t in outTIMESTAMP]

    # Compute reconstruction distances
    print("Compute reconstruction distances per channel...")
    dist = outUNIONdiff.pow(2).sum(2).sqrt()
    # Concat mean of dists
    dist = torch.cat([dist, dist.mean(dim=1).unsqueeze(1)], 1)

    rounded_outDATETIME = []
    for dt in outDATETIME:
        # round to nearest sample_len minutes
        round_to = 60 * sample_len  # seconds = 60 * sample_len minutes
        seconds = (dt - dt.min.replace(tzinfo=timezone.utc)).seconds
        rounding = (seconds+round_to/2) // round_to * round_to
        rounded_dt = dt + timedelta(0, rounding-seconds, -dt.microsecond)
        rounded_outDATETIME.append(rounded_dt)

    labels = labels_list[date_time_list.index(min(rounded_outDATETIME)):date_time_list.index(max(rounded_outDATETIME))+1]

    # Complete time series
    df = pd.DataFrame(list(zip(rounded_outDATETIME, outDATETIME)), columns=['roundedDatetime', 'Datetime'])

    threshold_dict = dict()
    for i in range(dist.shape[1]):
        df["dist_" + str(i)] = dist[:, i]
        if args['voting'] is False:
            threshold_dict[i] = [0] + np.nanpercentile(dist[:, i], args['threshold_percentiles']).tolist()

    r = pd.date_range(start=df.roundedDatetime.min(), end=df.roundedDatetime.max(), freq=str(sample_len) + "T")
    df = df.set_index('roundedDatetime').reindex(r).fillna(np.nan).rename_axis('roundedDatetime').reset_index()
    df['labels'] = labels

    # Extract complete reconstruction distance from df
    complete_dist = torch.Tensor()
    for i in range(dist.shape[1]):
        complete_dist = torch.cat([complete_dist, torch.tensor(df['dist_' + str(i)]).unsqueeze(1)], 1)

    # Create event_list, list of (start, finish, label) of each event
    events = list(zip(df.index, df.labels))
    ev_groups = [list(group) for key, group in itertools.groupby(events, lambda i: i[1] == args['detection_labels'][0])]
    event_list = [(it[0][0], it[-1][0], it[0][1]) for it in ev_groups if it[0][1] == args['detection_labels'][0]]

    # Create output folder
    timestamp_str = datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H-%M-%S')
    vot = "voting" if args['voting'] else "no_voting"
    if os.path.isfile(checkpoint) or os.path.basename(os.path.normpath(checkpoint)) == "ckpt":
        detection_output = os.path.abspath(os.path.join(os.path.join(os.path.dirname(checkpoint), os.pardir), "output/detection/" + f'{timestamp_str}_{vot}'))
    else:
        detection_output = os.path.join(checkpoint, "output/detection/" + f'{timestamp_str}_{vot}')
    Path(detection_output).mkdir(parents=True, exist_ok=True)

    # Dump experiment hyper-params
    with open(os.path.join(detection_output, 'hyperparams.txt'), mode='w') as f:
        args_str = [f'{a}: {v}\n' for a, v in args.items()]
        args_str.append(f'exp_name: {timestamp_str}\n')
        f.writelines(sorted(args_str))

    # Creation of alarms and detection
    alrm_dict = dict()
    if args['voting'] is False:
        print("Channel voting mechanism disabled")

        # Create list of combinations of channels_list, consecutive_outliers, hysteresis
        ch_co_hy_list = list(itertools.product(range(len(args['channels_list'])), args['consecutive_outliers'], args['hysteresis']))

        for ch, co, hy_hours in tqdm(ch_co_hy_list, desc="Finding Alarms"):
            hy = int(hy_hours * (60/sample_len))
            th_dict = dict()
            for t, th in enumerate(threshold_dict[ch]):
                ch_th_dist = [it >= float(th) if it == it else np.nan for it in complete_dist[:, ch].tolist()]

                shifted_lists = [ch_th_dist]
                for c in range(1, co):
                    shifted_lists.append([0]*c + ch_th_dist[:-c])
                nan_shifted_lists = [np.isnan(s).tolist() for s in shifted_lists]
                shifted_lists_t = torch.tensor(shifted_lists)
                nan_shifted_lists_t = torch.tensor(nan_shifted_lists)
                nan_sum = nan_shifted_lists_t.sum(dim=0)
                nan_vot = (nan_sum >= co/2).tolist()
                all_ok = shifted_lists_t.all(dim=0).to(torch.float)
                all_ok[nan_vot] = np.nan

                alm_list = [True if it == 1 else (False if it == 0 else np.nan) for it in all_ok.tolist()]
                groups = [list(group) for key, group in itertools.groupby(alm_list, lambda i: (i == True, i == False))]
                hysteresis_groups = [groups[0]] + [[True if j < hy else groups[i][j] for j in range(len(groups[i]))] if ((groups[i][0] == False or np.isnan(groups[i][0])) and groups[i-1][-1] == True) else groups[i] for i in range(1, len(groups))]
                ch_th_dist_hysteresis = [item for sublist in hysteresis_groups for item in sublist]
                alm_groups = [list(group) for key, group in itertools.groupby(enumerate(ch_th_dist_hysteresis), lambda i: (i[1] == True, i[1] == False))]
                start_finish_alarms = [(gt[0][0], gt[-1][0], gt[0][1]) for gt in alm_groups if (gt[0][1] == True or np.isnan(gt[0][1]))]
                th_dict[th] = start_finish_alarms
                # th_dict[str(th) + "alrm_list"] = ch_th_dist_hysteresis
            alrm_dict["CH"+str(args['channels_list'][ch])+"_CO" + str(co)+"_HY"+str(hy_hours)] = th_dict

        detection_dict = dict()
        for key in tqdm(alrm_dict.keys(), desc="Detecting"):
            th_dict = dict()
            for t, th in enumerate(alrm_dict[key].keys()):
                if not isinstance(th, str):
                    dt_dict = dict()
                    detections = list()
                    already_detected = list()
                    count_FP = 0
                    count_FN = 0
                    time_alarm = 0
                    advance_delay = 0

                    event_list_clean = remove_event_nan(event_list, alrm_dict[key][th])

                    for ev in event_list_clean:
                        for al in alrm_dict[key][th]:
                            if al[2]:
                                ov = overlap(ev, al)
                                if ov is not False:
                                    if ev not in (item[0] for item in detections):
                                        detections.append((ev, al, ov))
                                        advance_delay += ov
                                    else:
                                        already_detected.append(al)

                    for al in alrm_dict[key][th]:
                        if al[2]:
                            time_alarm += al[1]-al[0]+1
                            if al not in (item[1] for item in detections) and al not in already_detected:
                                count_FP += 1

                    for ev in event_list_clean:
                        if ev not in (item[0] for item in detections):
                            count_FN += 1

                    TP = len(detections)
                    FP = count_FP
                    FN = count_FN

                    TPR = TP/(TP+FN) if (TP+FN) != 0 else 0
                    PPV = TP/(TP+FP) if (TP+FP) != 0 else 0
                    FDR = FP/(TP+FP) if (TP+FP) != 0 else 0
                    F1 = 2*((PPV*TPR)/(PPV+TPR)) if (PPV+TPR) != 0 else 0
                    F05 = ((1+0.5**2)*PPV*TPR) / ((0.5**2) * (PPV+TPR)) if (PPV+TPR) != 0 else 0

                    dt_dict["TP"] = TP
                    dt_dict["FP"] = FP
                    dt_dict["FN"] = FN
                    dt_dict["TPR"] = TPR
                    dt_dict["PPV"] = PPV
                    dt_dict["FDR"] = FDR
                    dt_dict["F1"] = F1
                    dt_dict["F05"] = F05
                    dt_dict["TIME_ALARM"] = (time_alarm/len(events))*100
                    dt_dict["ADVANCE_DELAY"] = (advance_delay*sample_len)/len(detections) if len(detections) != 0 else np.nan
                    dt_dict["DETECTIONS"] = detections
                    th_dict[str(([0] + args['threshold_percentiles'])[t])] = dt_dict

            detection_dict[key] = th_dict
    elif args['voting'] is True:
        print("Channel voting mechanism enabled")
        ch_co_hy_list = list(itertools.product(range(1, len(args['detection_channels_voting'])+1), args['consecutive_outliers_voting'], args['hysteresis_voting']))

        # If all percentiles are equal
        th_str = str(args['consecutive_outliers_voting'][0]) if args['consecutive_outliers_voting'].count(args['consecutive_outliers_voting'][0]) == len(args['consecutive_outliers_voting']) else "best"

        vot_index = [i for i in range(len(args['channels_list'])) if args['channels_list'][i] in args['detection_channels_voting']]
        complete_dist_vot = complete_dist[:, vot_index]
        th_perc_list = [args['threshold_percentile_voting'][i] for i in vot_index]
        ch_th_dist_voting = list()
        for ch in tqdm(range(len(args['detection_channels_voting'])), desc="Voting"):
            th = np.nanpercentile(complete_dist_vot[:, ch], th_perc_list[ch])
            ch_th_dist = [it >= float(th) if it == it else np.nan for it in complete_dist_vot[:, ch].tolist()]
            ch_th_dist_voting.append(ch_th_dist)
        ch_th_dist_voting = np.array(ch_th_dist_voting).sum(0)

        for ch, co, hy_hour in tqdm(ch_co_hy_list, desc="Finding alarms"):
            hy = int(hy_hour * (60/sample_len))
            ch_th_dist_vot = [it >= ch if it == it else np.nan for it in ch_th_dist_voting]
            shifted_lists = [ch_th_dist_vot]
            for i in range(1, co):
                shifted_lists.append([0]*i + ch_th_dist_vot[:-i])
            nan_shifted_lists = [np.isnan(s).tolist() for s in shifted_lists]
            shifted_lists_t = torch.tensor(shifted_lists)
            nan_shifted_lists_t = torch.tensor(nan_shifted_lists)
            nan_sum = nan_shifted_lists_t.sum(dim=0)
            nan_vot = (nan_sum >= co/2).tolist()
            all_ok = shifted_lists_t.all(dim=0).to(torch.float)
            all_ok[nan_vot] = np.nan

            alm_list = [True if it == 1 else (False if it == 0 else np.nan) for it in all_ok.tolist()]
            groups = [list(group) for key, group in itertools.groupby(alm_list, lambda i: (i == True, i == False))]
            hysteresis_groups = [groups[0]] + [[True if j < hy else groups[i][j] for j in range(len(groups[i]))] if ((groups[i][0] == False or np.isnan(groups[i][0])) and groups[i-1][-1] == True) else groups[i] for i in range(1, len(groups))]
            ch_th_dist_hysteresis = [item for sublist in hysteresis_groups for item in sublist]
            alm_groups = [list(group) for key, group in itertools.groupby(enumerate(ch_th_dist_hysteresis), lambda i: (i[1] == True, i[1] == False))]
            start_finish_alarms = [(gt[0][0], gt[-1][0], gt[0][1]) for gt in alm_groups if (gt[0][1] == True or np.isnan(gt[0][1]))]
            alrm_dict["VOT"+str(ch)+"_CO" + str(co)+"_HY"+str(hy)+"_THperc"+th_str] = start_finish_alarms

        detection_dict = dict()
        for key in tqdm(alrm_dict.keys(), desc="Detecting"):
            dt_dict = dict()
            detections = list()
            already_detected = list()
            count_FP = 0
            count_FN = 0
            time_alarm = 0
            advance_delay = 0

            event_list_clean = remove_event_nan(event_list, alrm_dict[key])

            for ev in event_list_clean:
                for al in alrm_dict[key]:
                    if al[2]:
                        ov = overlap(ev, al)
                        if ov is not False:
                            if ev not in (item[0] for item in detections):
                                detections.append((ev, al, ov))
                                advance_delay += ov
                            else:
                                already_detected.append(al)

            for al in alrm_dict[key]:
                if al[2]:
                    time_alarm += al[1]-al[0]+1
                    if al not in (item[1] for item in detections) and al not in already_detected:
                        count_FP += 1

            for ev in event_list_clean:
                if ev not in (item[0] for item in detections):
                    count_FN += 1

            TP = len(detections)
            FP = count_FP
            FN = count_FN

            TPR = TP/(TP+FN) if (TP+FN) != 0 else 0
            PPV = TP/(TP+FP) if (TP+FP) != 0 else 0
            FDR = FP/(TP+FP) if (TP+FP) != 0 else 0
            F1 = 2*((PPV*TPR)/(PPV+TPR)) if (PPV+TPR) != 0 else 0
            F05 = ((1+0.5**2)*PPV*TPR) / ((0.5**2)*(PPV+TPR)) if (PPV+TPR) != 0 else 0

            dt_dict["TP"] = TP
            dt_dict["FP"] = FP
            dt_dict["FN"] = FN
            dt_dict["TPR"] = TPR
            dt_dict["PPV"] = PPV
            dt_dict["FDR"] = FDR
            dt_dict["F1"] = F1
            dt_dict["F05"] = F05
            dt_dict["TIME_ALARM"] = (time_alarm/len(events))*100
            dt_dict["ADVANCE_DELAY"] = (advance_delay*sample_len)/len(detections) if len(detections) != 0 else np.nan
            dt_dict["DETECTIONS"] = detections

            detection_dict[key] = dt_dict

    # Saving output
    torch.save(alrm_dict, os.path.join(detection_output, "alrm_dict_" + vot + ".pt"))
    det_dict_path = os.path.join(detection_output, "detection_dict_" + vot + ".pt")
    torch.save(detection_dict, det_dict_path)

    # Check Prediction
    check_predictions(detection_dict, detection_output, "detection_" + vot + ".xlsx")

    # Plot
    plot(det_dict_path, args['channels_name'])

    if args['voting'] is False:
        # Start interactive plot
        interactive_plot(detection_dict, args['channels_name'])
