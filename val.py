import os
from config import cfg
import argparse
from datasets import make_dataloader, make_dataloader_all, make_dataloader_val
from model import make_model
from processor import do_inference, do_inference_all
from utils.logger import setup_logger
import json

def calculate(train_txt='/raid/chenby/person_rID/2019/REID2019_A/train_list.txt',
              json_result='/raid/chenby/person_rID/2019/REID2019_A/result.json'):
    with open(train_txt, "r") as f:
        lines = f.readlines()
    img_dict = {}
    for line in lines:
        img_name, img_label = [i for i in line.replace('\n', '').split()]
        img_dict[img_name.replace('train/', '')] = img_label

    with open(json_result, 'r') as load_f:
        result = json.load(load_f)

    count = 0
    for q_img, g_imgs in result.items():
        if img_dict[q_img] == img_dict[g_imgs[0]]:
            count += 1
    rank1 = count / len(result)

    res_all = 0
    for q_img, g_imgs in result.items():
        correct = 0
        res = 0
        for i in range(len(g_imgs)):
            pred = 0
            if img_dict[q_img] == img_dict[g_imgs[i]]:
                correct += 1
                pred = 1
            res += pred * correct / (i+1)
        if correct > 0:
            res_all += res/correct

    mAP200 = res_all / len(result)

    print('rank1:{:.4f}, mAP@200:{:.4f}, result:{:.4f}'.format(rank1, mAP200, (rank1 + mAP200) / 2))

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    val_loader_query, val_loader_gallery, num_query, num_classes = make_dataloader_val(cfg)
    model = make_model(cfg, num_class=num_classes)
    model.load_param(cfg.TEST.WEIGHT)
    do_inference_all(cfg,
                     model,
                     val_loader_query,
                     val_loader_gallery,
                     num_query,
                     is_save=False)

if __name__ == "__main__":
    main()
    calculate()
