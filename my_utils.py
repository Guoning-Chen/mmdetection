from tools.analyze_logs import load_json_logs


def merge_log(train_log, retrain_log):
    new_dict, retrain_dict = load_json_logs([train_log, retrain_log])
    start_epoch = len(new_dict)
    for epoch, value in retrain_dict.items():
        new_dict[epoch+start_epoch] = value
    return new_dict


if __name__ == "__main__":
    merge_log(train_log='work_dirs/r50pf_fpn_1x_sk/20210630_162250.log.json',
              retrain_log='work_dirs/retrain_r50_pf-B/20210701_105422.log.json')