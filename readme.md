# cityscapestococo

This repo contains all code which convert cityscapes to coco. The scripts actually provided inside Detectron and maskrcnn-benchmark were all down. To help others quickly setup their training pipeline on cityscapes for instance segmentation, this repo is really helpful.

The converted result cityscapes to coco visualized using `vis_coco.py` inside this repo:

![](https://s2.ax1x.com/2019/10/23/KYJqzj.png)

![](https://s2.ax1x.com/2019/10/23/KYJxe0.png)

![](https://s2.ax1x.com/2019/10/23/KYYCYF.png)

As you can see, all instance classes were token out from cityscapes, which can be conclude as:

```
[
        'person',
        'rider',
        'car',
        'truck',
        'bus',
        'train',
        'motorcycle',
        'bicycle',
    ]

```



## Usage

```
python3 convert_cityscapes_to_coco.py
```

For visual:

```
python3 vis_coco.py path/to/annn.json path/to/images/
```

**note**: You have to move all cityscapes images to a single folder without any subfolders just follow the coco structures.



## Copyright

All rights belongs to Fagang Jin, codes released under Apache License
