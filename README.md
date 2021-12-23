# General Purpose Vision Normal Estimation Benchmark

- We provide four surface normal datasets for training, validating, and testing:

    - Indoor Scenes: ScanNet (training / validating), NYUv2 (testing)
    - Object Scenes: BlendedMVS (training / validating), DTU (testing)

- Inside ``datasets``, we provide instances of the PyTorch Dataset class for each dataset. 

# Evaluation:
- Manually change the network inside the ```bin/evaluate.py```
- Run ```bin/evaluate.py``` and specify visualization and results save path:
  
  ```
  python bin/evaluate.py --results_dir results/saved_images \
                         --log_path results/saved_evaluation \
                         --rotation
  ```
  
- Evluation results and  would be saved under ``results``
    
