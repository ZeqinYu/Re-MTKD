import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import warnings

warnings.filterwarnings("ignore")
import gc
import argparse
import torch
import numpy as np
from tqdm import tqdm
from dataset import ImageDataset
from evaluation import cal_f1, cal_auc
import random
from Cue_Net.model.CueNet import CueNet
from sklearn.metrics import f1_score, roc_auc_score


def set_seed(seed=98):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def test_visual(args, model, test_dataloader):
    gc.collect()
    model.eval()
    with torch.no_grad():
        f1, auc = [], []
        class_label_list, score_list, score_threshold_list = [], [], []
        for x, gt, class_label in tqdm(test_dataloader):
            x = x.to(device)
            gt = gt.numpy()

            prob, score, _, _ = model(x)
            prob = prob.detach().cpu().numpy()
            score = score.detach().cpu().numpy()
            prob_threshold = (prob > 0.5).astype(np.float32)
            score_threshold = (score > 0.5).astype(np.float32)
            class_label = class_label.cpu().numpy().astype(np.float32)

            score = np.array(score, dtype=np.float32).reshape(-1)
            score_threshold = np.array(score_threshold, dtype=np.float32).reshape(-1)
            class_label = np.array(class_label, dtype=np.float32).reshape(-1)

            class_label_list.append(class_label)
            score_list.append(score)
            score_threshold_list.append(score_threshold)

            if class_label.item() == 1:
                f1.append(cal_f1(gt, prob_threshold))
                auc.append(cal_auc(gt, prob))

        print(
            f'\n\n'
            f'==> model: {args.model_path}\n'
            f'==> dataset: {args.txt_list}\n'
            f'== test result: '
            f'localization f1: {np.mean(f1):.4f} '
            f'localization auc: {np.mean(auc):.4f} '
            f'detection f1: {f1_score(class_label_list, score_threshold_list):.4f}  '
            f'detection auc: {roc_auc_score(np.concatenate(class_label_list), np.concatenate(score_list)):.4f}  '
        )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='./Cue_Net/checkpoint/ICCV2025_DeepID_Detection_1st_Sunlight.pt')
    parser.add_argument('--img_size', type=list, default=[512, 512])
    parser.add_argument('--txt_list', type=str,
                        default='./test_datasets_txt/copymove_test+.txt')
    parser.add_argument('--resize', type=bool, default=True)
    # AAAI 2025 Re-MTKD uses "resize = True" for testing,
    # ICCV DeepID 2025 competition uses "resize = False" for testing.
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--backbone_model', type=str, default='backbone')

    args = parser.parse_args()

    test_dataset = ImageDataset(
        txt_file=args.txt_list,
        img_size=args.img_size,
        resize=args.resize,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=8,
        drop_last=False)

    model = CueNet(backbone=args.backbone_model, num_classes=1, pretrained=True).to(device)
    model_param = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(model_param['model_state_dict'])
    del model_param

    test_visual(args, model, test_dataloader)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed=98)
    print(f'=====Use {device}=========')
    main()

