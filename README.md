# SPCT

SPCT: Soft Pattern Coding Transformer for Lightweight Image Super-Resolution

Qian Wang, Ruilong Guo, Yanyu Mao, Yao Tang and Jiulun Fan

## 💻Environment

- [PyTorch >= 1.9](https://pytorch.org/)
- [Python 3.7](https://www.python.org/downloads/)
- [Numpy](https://numpy.org/)
- [BasicSR >= 1.3.4.9](https://github.com/XPixelGroup/BasicSR)

## 🔧Installation

```python
pip install -r requirements.txt
```

## 📜Data Preparation

The trainset uses the DIV2K (800). In order to effectively improve the training speed, images are cropped to 480 * 480 images by running script extract_subimages.py, and the dataloader will further randomly crop the images to the GT_size required for training. GT_size defaults to 128/192/256 (×2/×3/×4). 

```python
python extract_subimages.py
```

The input and output paths of cropped pictures can be modify in this script. Default location: ./datasets/DIV2K.

## 🚀Train

▶️ You can change the training strategy by modifying the configuration file. The default configuration files are included in ./options/train/SPCT. Take one GPU as the example.

```python
### Train ###
### SPCT ###
python train.py -opt ./options/train/SPCT/train_SPCT_x2.yml --auto_resume  # ×2
python train.py -opt ./options/train/SPCT/train_SPCT_x3.yml --auto_resume  # ×3
python train.py -opt ./options/train/SPCT/train_SPCT_x4.yml --auto_resume  # ×4
python train.py -opt ./options/train/SPCT/train_SPCT_Tiny_x2.yml --auto_resume  # ×2
python train.py -opt ./options/train/SPCT/train_SPCT_Tiny_x3.yml --auto_resume  # ×3
python train.py -opt ./options/train/SPCT/train_SPCT_Tiny_x4.yml --auto_resume  # ×4
```

For more training commands, please check the docs in [BasicSR](https://github.com/XPixelGroup/BasicSR).

## 🚀Test

▶️ You can modify the configuration file about the test, which is located in ./options/test/SPCT. At the same time, you can change the benchmark datasets and modify the path of the pre-train model. 

▶️ We will publish all SPCT pre-trained models.

```python
### Test ###
### SPCT for Lightweight Image Super-Resolution ###
python basicsr/test.py -opt ./options/test/SPCT/test_SPCT_x2.yml  # ×2
python basicsr/test.py -opt ./options/test/SPCT/test_SPCT_x3.yml  # ×3
python basicsr/test.py -opt ./options/test/SPCT/test_SPCT_x4.yml  # ×4
python basicsr/test.py -opt ./options/test/SPCT/test_SPCT_Tiny_x2.yml  # ×2
python basicsr/test.py -opt ./options/test/SPCT/test_SPCT_Tiny_x3.yml  # ×3
python basicsr/test.py -opt ./options/test/SPCT/test_SPCT_Tiny_x4.yml  # ×4
```

## 🚩Results

The inference results on benchmark datasets will be available at [Google Drive](https://drive.google.com/file/d).

## :mailbox:Contact

If you have any questions, please feel free to contact us wqabby@xupt.edu.cn, RuilongGuo@stu.xupt.edu.cn and bolt35982@gmail.com.
