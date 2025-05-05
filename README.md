# 1.Dataset
```text
data
├── coco
│   ├── precomp 
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── ......
│   │
│   ├── images   
│        ├── train2014
│        └── val2014
│  
├── f30k
│   ├── precomp  
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── ......
│   │
│   ├── flickr30k-images   # raw coco images
│          ├── xxx.jpg
│          └── ...
│   
└── vocab  
```
The download links for original COCO/F30K images, precomputed BUTD features, and corresponding vocabularies are from the official repo of SCAN(https://github.com/kuanghuei/SCAN#download-data）. The precomp folders contain pre-computed BUTD region features, data/coco/images contains raw MS-COCO images, and data/f30k/flickr30k-images contains raw Flickr30K images. Because the download link for the pre-computed features in SCAN is seemingly taken down. The [link](https://www.dropbox.com/scl/fo/vmd0dvz20t7aae9jal0nc/ALoI0grReuah2PB5NgHGmac?rlkey=rei5ljf7hro7chkxltkcs0odr&e=1&dl=0） provided by the author of [vse_infty](https://github.com/woodfrog/vse_infty) contains a copy of these files.

# 2.Training and Evaluation
1.run ./train_xxx_f30k.sh or ./train_xxx_coco.sh. For example:
```text
sh train_GRU_f30k.sh
```
## 2.Evaluation: Run the following commands after modifying the default data and model path to yourself path.
```text
cd ../
python eval_ensemble.py
```
