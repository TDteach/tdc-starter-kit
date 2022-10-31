# Evasive Trojans

## Description

This track has a different format than the other tracks. Instead of designing better Trojan detectors, your task is to design more evasive Trojan attacks that fool a range of baseline detectors while remaining effective. Crucially, these detectors are trained on the networks you submit (a white-box setting), so the top submissions will help elucidate how hard Trojan detection truly is.

We ask you to train 100 Trojaned MNIST networks and submit the parameters of these networks to the evaluation server. Then, the evaluation server will train and evaluate baseline detectors using your submitted networks and a held-out set of clean networks.

**Data:** We provide a reference set of 200 clean networks trained on MNIST. These networks are drawn from the same distribution of clean networks that are used to train baseline detectors in the evaluation server. More clean networks from this distribution can be trained by using `train_batch_of_models.py` with the trojan_type argument set to "clean". We also provide a set of 200 attack specifications. The attack specifications give the trigger and target label for each Trojaned network that should be submitted to the evaluation server. The evaluation server will reject submissions where the average attack success rate (ASR) of the submitted networks is below 97%. The ASR is determined by the attack specifications.

**Metrics:** Submissions will be evaluated using the maximum AUC across a fixed set of baseline Trojan detectors. Lower is better. If the attack specifications are not met, then the submission will not be evaluated. Participants can check whether the attack specifications are met before submitting their networks to the evaluation server by using an automated script that we will provide.

**Additional Details:** For the MNTD baseline detector, we compute AUC using k-fold cross-validation on the submitted Trojaned networks and held-out clean networks. This is because the MNTD detector requires training on a dataset of networks, while the other detectors do not.

## Folder Content

The file `train_batch_of_models.py` can be used for training a dataset of Trojaned neural networks. It supports batching the training across multiple machines to accelerate experiments. It currently supports training normal Trojans and the baseline evasive Trojans (see `tdc_starter_kit/utils.py` for the training functions). Once your dataset of Trojaned neural networks is done training, we provide code for testing whether it meets the attack specifications and code for generating a submission in `example_submission.ipynb`.


## Usage

First, make sure `./data/test/attack_specifications.pkl` exists and those reference models are stored in `./data/reference_models`.

If you want to change the location accordingly, please adjust the `load_attack_specifications` function in `tools.py` to change the location of `attack_specifications.pkl`, and adjust the `get_data` function in `tools.py` to change the location of reference models that will be used to select the best models for submission.


Run the following command to get the 5 zipfils for online testing: `./submission_0.zip`, `./submission_1.zip`, `./submission_2.zip`, `./submission_3.zip` and `./submission_4.zip`.
```
sh train_batch_of_models.sh
```

On a server with single 3090Ti GPU, running the above command would need about 1 week. 

To accelerate this running on multiple GPUs, please read `train_batch_of_models.sh` and split the job acoordingly on your GPUs, e.g., training 100 models on GPU0 and other 100 models on GPU1 on a dual-GPU server.

To fastly validate the results, you could reduce the total iterations for final adjustment in the 617th-line of `tools.py`.





