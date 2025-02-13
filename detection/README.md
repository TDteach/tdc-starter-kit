# Trojan Detection

This is the starter kit for the Detection Track and the Final Round. To learn more, please see the [Tracks page](https://www.trojandetection.ai/tracks) of the competition website.

## Contents

Using the MNTD baseline as an example, we provide starter code for submission in `example_submission.ipynb`.

## Usage

First, adjust the data path according to your settings:
1) `FINAL_ROUND_FOLDER`, at the 15-th line of `detection.py`;
2) `BENIGN_MODELS_FOLDER`, at the 16-th line of `reversion.py`. 

Run `detection.py` will get all the scores stored in `my_submission/predictions.npy`.

Use the following command to generate the `submission.zip`
```bash
cd my_submission && zip ../submission.zip ./* && cd ..
```
