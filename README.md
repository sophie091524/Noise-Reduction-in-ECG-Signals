# Noise-Reduction-in-ECG-Signals

In this project, a denoising autoencoder (DAE) using fully convolutional network(FCN) is proposed for ECG signal denoising. Meanwhile, the proposed FCN-based DAE can perform compression with regard to the DAE architecture. 

The motivation of the project is because in real-world scenario, ECG signals are prone to be contaminated by different kinds of noise, such as baseline wander (BW), muscle artifact (MA), and electrode motion (EM) [1]. All these noises may cause deformations on ECG waveforms and mask tiny features that are important for diagnosis. Accordingly, the removal of noises from ECG signals becomes necessary.

## Proposed FCN-based DAE

The proposed FCN-based DAE consists of an encoder and a decoder with 13 layers as shown in below. 

## EXPERIMENTS
### Experimental Data
We employ MIT-BIH Arrhythmia database [2] for denoising ECG signals. The proposed approach was applied to all patients and the ECG signals are segmented including 1024 samples. 

Real noises including BW, MA and EM are obtained from MIT- BIH Normal Sinus Rhythm Database (NSTDB). All signals are normalized so that the amplitudes of the sampling points lay between 0 and 1 as a preliminary operation [29].

The dataset is split into 80% training, 10% validation and the remaining 10% were tested for the performance evaluation of the training model. In total, the dataset contains 7680 training, 960 validation and 960 testing fragments. All training data are corrupted with input SNR of -2.5, 0, 2.5, 5 and 7.5 dB. Both training and validation sets share the same SNR range
### Experimental Results

## Discussion

## Publication
