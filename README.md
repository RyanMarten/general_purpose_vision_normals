# General Purpose Vision Normal Estimation Benchmark

- We provide four surface normal datasets for training, validating, and testing:

    - Indoor Scenes: ScanNet (training / validating), NYUv2 (testing)
    - Object Scenes: BlendedMVS (training / validating), DTU (testing)

- Inside ``datasets``, we provide instances of the PyTorch Dataset class for each dataset. 

## Evaluation:
- Manually change the network inside the ```bin/evaluate.py```
  - We use the pretrained network from [Surface Normal Uncertainty](https://github.com/baegwangbin/surface_normal_uncertainty)
  as our standard network. 
    
  - Please download the checkpoint by running ``python download.py`` before running this standard networt:
- Run ```bin/evaluate.py``` and specify visualization and results save path:
  
  ```
  python bin/evaluate.py --results_dir results/saved_images \
                         --log_path results/saved_evaluation \
                         --rotation
  ```
  
- Evluation results and  would be saved under ``results``

## Evaluation Results:
 Datasets | Type | Mean Error | Median Error | 11.25 | 22.5 | 30 |
--- | --- | --- | --- |--- |--- | --- |
 ScanNet (without solving rotation) | Validation | 11.8 | 5.7 | 71.1 | 85.4 | 89.8 |
 NYUv2 | Testing | 16.8 | 9.7 | 56.9 | 75.3 | 82.3 |
 BlendedMVS | Validation | 27.0 | 20.3 | 36.9 | 55.3 | 63.7 |
 DTU | Testing  | 46.6 | 37.9 | 34.8 | 48.0 | 54.3 |

## Data Sources:
- For ScanNet, we use the preprocessed data from [FrameNets](https://github.com/hjwdzh/FrameNet)
- For the other three, we download the data from original websites, and render surface normal using provided depth maps.
