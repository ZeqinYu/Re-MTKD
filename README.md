# Re-MTKD

---

Official implementation of Re-MTKD from our AAAI 2025 (Oral) paper “Reinforced Multi-teacher Knowledge Distillation for Efficient General Image Forgery Detection and Localization”. <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/ZeqinYu/Re-MTKD) [![arXiv](https://img.shields.io/badge/Arxiv-2504.05224-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2504.05224) <br>

---

## 📰 News
* **[2025.10.22]** We have released the testing code. If you have any questions, please feel free to contact us.
* **[2025.10.18]** 🔥🔥 Our team won 1st place ×2 in both the Detection and Localization tracks of the ICCV 2025 DeepID Challenge using our Re-MTKD framework.
We have released the competition Docker implementation at [ICCV-DeepID2025-Sunlight](https://github.com/ZeqinYu/ICCV-DeepID2025-Sunlight), which follows the same format used in the official challenge.
A testing-friendly code version will be released soon on this main repository.

> **TODO**
> - [x] 📦 **[2025.10.18]** ~~Release Re-MTKD code & checkpoints~~ 
> - [x] 🔗 **[2025.10.18]** ~~Release train/val split configuration files~~ 
> - [x] 🔗 **[2025.06.09]** ~~Release test split configuration files~~ 

## 🎯 Test
1. Clone the repository:
    ```bash
    git clone https://github.com/ZeqinYu/Re-MTKD.git
    cd Re-MTKD
    ```
2. Prepare Model:

    The pretrained weights can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1WOZdJY3VJ5SWpqPYTttGA36dRcU1OGSf) or [Baidu Pan]( https://pan.baidu.com/s/1GhCtZWbWld9Zx7GGpJxcVQ) (42rp), and place the weights into `/yourpath/Re-MTKD/Cue_Net/checkpoint`. Then modify the `model_path` variable in `/yourpath/Re-MTKD/test.py` to test different models.

4. Test Model:

   You can test Re-MTKD by:
   ```bash
   python test.py
   ```
   **Note: AAAI 2025 Re-MTKD uses "resize = True" for testing; ICCV DeepID 2025 competition uses "resize = False" for testing.**

## ✍️ Citation
```bibtex
@inproceedings{yu2025reinforced,
  title={Reinforced Multi-teacher Knowledge Distillation for Efficient General Image Forgery Detection and Localization},
  author={Yu, Zeqin and Ni, Jiangqun and Zhang, Jian and Deng, Haoyi and Lin, Yuzhen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={1},
  pages={995--1003},
  year={2025}
}
