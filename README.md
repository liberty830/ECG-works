Implemetation of necessary steps to analyze ECG signals especially for wearable ECG devices which includes a variety of noise sources.

1. Noise Removal: There are many sources of noise for the wearable ECG patch compared to the Holter which is normally used in the hospitals.
                  Those noises are not effectively removed with the traditional methods, such as DWT which uses frequency bands to filter.
                  So, here I introduce semi-supervised learning technique from the clean set to remove noise patterns in ECG with Deep Learning
                  
2. R-peak detection: There are many state-of-art methods to detect QRS-complex. However, since there are much more variance in wearable ECG devices,
                     those methods do not work well for this case. To alleviate this, I used hierachycal approach by combining existing methods.
                     And I improved the computation speed more than 10 times with vectorization or "NUMBA" package in Python.
                     
                     
References: 

1. QRS detection method 1: https://www.robots.ox.ac.uk/~gari/teaching/cdt/A3/readings/ECG/Pan+Tompkins.pdf
2. QRS detection method 2: https://www.scitepress.org/papers/2010/27427/27427.pdf
3. Reference for the implementation: https://github.com/berndporr/py-ecg-detectors
4. Open source wearable ECG patch data: https://physionet.org/content/butqdb/1.0.0/
5. Open source holter ECG data: https://physionet.org/content/mitdb/1.0.0/
                   
