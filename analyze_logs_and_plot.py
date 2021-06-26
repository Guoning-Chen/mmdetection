from tools import analyze_logs


if __name__ == '__main__':
    json_logs = ['work_dirs/mask_rcnn_r101_fpn_1x_sk/20210626_155358.log.json']
    log_dicts = analyze_logs.load_json_logs(json_logs)
    pass