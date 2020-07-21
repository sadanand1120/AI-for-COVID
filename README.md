# AI-for-COVID

There are some changes in both versions due to shift from windows to linux. The linux version uses multiprocessing for creating independent processes one each for each camera input, and for each camera source, it agin splits up the processes for different behaviours. However, because in windows "forking" of processes is not available, so the windows version doesn't use multiprocessing and rather uses multithreading.

Both versions, can be used for any number of behaviours and any number of input sources. (However I had tested for only two videos simultaneously).

The `scripts/all_behaviours.py` takes care of all behaviour functions we want to include. Currently, it only has one behaviour method `faceMaskDetector` which is used in the `integrating_behaviours.py` script. Other behaviour methods can also be included in the `AllBehaviours` class defined in `all_behaviours.py` script.

## For Using/Testing:

First, get the checkpoint file from [here](https://drive.google.com/drive/folders/1UlF6PmTwwd4cm-wD9v6Qy7gbC_tzif_j). Then place the two input videos in the `videos/input` folder.

The main script is `integrating_behaviours.py`. It takes two command line args, one each for two input videos.

Execute the below command in command prompt:
```
python integrating_behaviours.py --input1 path/to/video1 --input2 path/to/video2
```
