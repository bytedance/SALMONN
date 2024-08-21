## Inference

### Preparation
Install the environment with the following specified config:
```
conda env create -f videosalmonn.yml
```
Then download various checkpoints


### Run inference
```
python inference.py --cfg-path config/test.yaml 
```

### Check the result
The result is saved in the following path:
```
./ckpt/MultiResQFormer/<DateTime>/eval_result.json
```
