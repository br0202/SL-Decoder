# SL-Decoder

## [Self-Supervised Depth Estimation in Laparoscopic Image using 3D Geometric Consistency](https://arxiv.org/abs/2208.08407) (MICCAI 2022)(Data generation code)
By [Baoru Huang](https://baoru.netlify.app/), Jian-Qing Zheng, [Anh Nguyen](https://www.csc.liv.ac.uk/~anguyen), Chi Xu, Ioannis Gkouzionis, Kunal Vyas, David Tuch, Stamatia Giannarou, Daniel S. Elson

![image](https://github.com/br0202/SL-Decoder/blob/master/figure/10_1-l.png "example") ![image](https://github.com/br0202/SL-Decoder/blob/master/figure/10_1-r.png "example")

### Contents
1. [How to use](#Howtouse)
1. [Data Structure](#DataStructure)
2. [Depth Map Generation](#DepthMapGeneration)
3. [Notes](#notes)


### How to use

1. Download 'SL-Decoder'
2. Capture original RGB image with no pattern projection and name it '00000.jpg'
3. Project patterns in patterns.odp on the object that you want to calculate the depth and for each image captured with patten, name it as '00001.jpg, 00002.jpg..., 00022.jpg'
4. For every set, save it in folder 'round_n' where n indicates the round index
5. Save the round folder to folder 'L' and folder 'R' for left and right side images. 


### Data Structure

Dataset/
├─ L/
│  ├─ round_n/
│  │  ├─ 00000.jpg
│  │  ├─ 00001.jpg
│  │  ├─ ......
│  │  ├─ 00022.jpg
├─ R/
│  ├─ round_n/
│  │  ├─ 00000.jpg
│  │  ├─ 00001.jpg
│  │  ├─ ......
│  │  ├─ 00022.jpg
├─ recimgdepth/



### Depth Map Generation
1. Modify camera parameters (Intrinsic and extrinsic parameters)
2. Change the 'folder_list' in GT_depth.py to the path of the 'Dataset' folder
3. Create a folder called 'recimgdepth' under folder 'Dataset'
4. `cd $M3Depth`
5. `python GT_depth.py`



###  Notes:
1. Image size
	- Self.height and self.width under the 'def __init__' show dimensions of the original images.
	- Self.images_l,  self.images_r,  self.three_phase,  self,patterns_l,  and  self.patterns_r  represent dimensions after processing.
  
2. Orientation:
	- Patterns need to be projected vertically 



### Citing 

If you find our paper useful in your research, please consider citing:

        @inproceedings{huang2022self,
          title={Self-supervised Depth Estimation in Laparoscopic Image Using 3D Geometric Consistency},
          author={Huang, Baoru and Zheng, Jian-Qing and Nguyen, Anh and Xu, Chi and Gkouzionis, Ioannis and Vyas, Kunal and Tuch, David and Giannarou, Stamatia and Elson, Daniel S},
          booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
          pages={13--22},
          year={2022},
          organization={Springer}
        }


### License
MIT License

### Acknowledgement
This work was supported by the UK National Institute for Health Research (NIHR) Invention for Innovation Award NIHR200035, the Cancer Research UK Imperial Centre, the Royal Society (UF140290) and the NIHR Imperial Biomedical Research Centre.
