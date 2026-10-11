Robust CIFAR-10 classification from 500 images (Section B, Question 1)

A small Inception-style CNN (104,842 trainable parameters, trained from scratch) that is trained on only 400 clean CIFAR-10 images (plus 100 validation images) and evaluated on a shifted version of the CIFAR-10 test set. The notebook contains the baseline, the augmentation design, the regularisation choices, the ablation study, the error analysis and the test-time-augmentation bonus.
Data

CIFAR-10 is loaded through torchvision.datasets.CIFAR10 and stored in ./data. The notebook builds the three datasets itself from fixed seeds, so nothing else has to be downloaded:

train_small: 500 training images, 50 per class (seed 1234), split into 400 training and 100 validation images (stratified, 40 and 10 per class, seed 1235).
clean_test: the original 10,000 CIFAR-10 test images (used for reporting only).
shifted_test: the 10,000 test images passed through a fixed, seeded corruption pipeline (seed 777): random rotation up to ±25°, brightness and contrast changes, a black rectangular occlusion covering about 25% of the image, blur and noise on about half of the images, and a hidden corruption (random grayscale or random RGB channel swap on 50% of the images).

Important: the official hidden corruption was not available when this was run. The hidden corruption above is my own assumption, so all shifted-test numbers refer to this re-created set.

Offline use: put cifar-10-python.tar.gz into the ./data folder (the exact file name matters) before running, and torchvision should use it instead of downloading.
Expected results (3 seeds, mean ± std, accuracy in %)

The plain baseline reaches 100.00 on train, 46.67 ± 4.78 on validation, 41.59 ± 0.66 on clean test and 14.21 ± 0.33 on shifted test. The final model reaches 45.67 ± 3.09 on validation, 43.35 ± 3.46 on clean test and 24.75 ± 1.43 on shifted test (25.16 ± 1.42 with test-time augmentation over 8 views). Selected hyper-parameters: AdamW, learning rate 0.003, weight decay 0.01, Mixup with p = 0.7 and label smoothing 0.2. The largest single improvement comes from adding Cutout / Random Erasing (+8.18 points over geometric + photometric augmentation).

Using the official data (optional)

If the official files are provided, save them next to the notebook and run all cells again. The notebook uses them automatically.

train_small.npz with x (uint8, N×32×32×3) and y (labels). It is split 400/100 with seed 1235.
shifted_test.npz with x (uint8, 10000×32×32×3), in the same order as the CIFAR-10 test set, because the labels come from torchvision's test labels.

If the hidden corruption is given as a function instead, set OFFICIAL_HIDDEN_FN in the cell after E.4. After using official data, the numbers in the text cells and the summary must be updated to match the new output.

Notes and limitations
Only three seeds were run, so differences of about 1 point or less should not be over-interpreted.
The Part B answers (vertical flip, ±90° rotation, over-augmentation) are arguments; no separate experiment was run for them.
The final model is not better than the Cutout-only row of the ablation (24.75 vs 24.88 shifted test).
The final model is not more robust than the plain baseline to the hidden grayscale or channel-swap corruption.
