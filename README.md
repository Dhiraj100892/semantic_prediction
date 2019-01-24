# Requirements
Pytorch 1.0, Tensorflow, TensorboardX, TorchVision, PIL, OpenCV, termcolor 

## Steps
##### Place the data in proper location
```buildoutcfg
Keep the color/ depth/ label/ under data/ folder
│----data
│     │---- color
│     │---- depth
│     │---- label
```
##### Change directory to code 
```buildoutcfg
cd code
```
##### Generate train, val data file
```buildoutcfg
python data_txt.py
```
##### Calculate the surface normal corresponding to the data
```buildoutcfg
python surface_norm_from_depth.py
```
##### Train the model
```buildoutcfg
python main.py --pos-weight 10 --use-normal --epoch 400 --exp-id 1 --lr 1e-3 --rot-angle 60.0
```
#####Test the model
```buildoutcfg
python test.py --use-normal --exp-id 2
```