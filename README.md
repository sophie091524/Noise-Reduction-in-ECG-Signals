# Noise-Reduction-in-ECG-Signals

In this project, a denoising autoencoder (DAE) using fully convolutional network(FCN) is proposed for ECG signal denoising. Meanwhile, the proposed FCN-based DAE can perform compression with regard to the DAE architecture. 

The motivation of this project is because in real-world scenario, ECG signals are prone to be contaminated by different kinds of noise, such as baseline wander (BW), muscle artifact (MA), and electrode motion (EM) [1]. All these noises may cause deformations on ECG waveforms and mask tiny features that are important for diagnosis. Accordingly, the removal of noises from ECG signals becomes necessary.

## Proposed FCN-based DAE
The proposed FCN-based DAE consists of an encoder and a decoder with 13 layers as shown in below. 
<div align=center><img width="300" height="200" src="https://github.com/sophie091524/Noise-Reduction-in-ECG-Signals/blob/master/pic/fcn.jpg"/></div>

## Experiments
### Experimental Data
We employ MIT-BIH Arrhythmia database [2] for denoising ECG signals. Real noises including BW, MA and EM were collected from the MIT-BIH Noise Stress Test Database (NSTDB). 

The dataset was split into 80% training, 10% validation and the remaining 10% were used to evaluate the denoising models. All training data are corrupted with input SNR of -2.5, 0, 2.5, 5 and 7.5 dB. Both training and validation sets share the same SNR levels, and the testing set consisted of 3 different levels of input SNR of -1,3 and 7dB. All signals were normalized as a preliminary operation so that the amplitudes of the sampling points laid between 0 and 1 [3].

### Experimental Results
The denoising performance is compared to DNN based and CNN based DAE. As it can be seen, FCN performs better denoising of the ECG signal as compared to the two other approaches, as higher SNR_imp denotes more resemblance to clean signals. We can also observed that the RMSE and PRD values are lower for the proposed work as desired. These results demonstrate that FCN can achieve promising performance in reconstructing a denoised output signal form the original ECG in all noise levels compared to DNN and CNN.

For visual assessment, we can clearly observe that DNN appears a severe loss of amplitudes of R peaks. The waveforms generated by DNN have smaller R peaks that fail to reconstruct accurate ECG signals. This phenomenon can also be observed in CNN but is not as serious as that in DNN. In contrast, FCN can generally maintain the shape of QRS complexes than the compared methods. With high-fidelity QRS complexes, FCN preserve more clinically relevant information. 
<div align=center><img width="625" height="450" src="https://github.com/sophie091524/Noise-Reduction-in-ECG-Signals/blob/master/pic/result.jpg"/></div>

Overall, both quantitative and visual comparisons demonstrate that FCN gains an advantage over compared methods in terms of noise reduction and clinical detail preservation.

## Discussion
The experimental results have found that FCN outperform DNN and CNN, especially with DNN having the worst performance. This failing may be attributed to the limitation of fully connected layers. When generating waveforms, fully connected layers have high correlation with each other which result in the loss of spatial information. In contrast, the convolutional layers have the property of local connectivity. Each neuron only depends on a small region of the previous layer which is called the receptive field. The input features share the same weights within receptive field confining local patterns and enable FCN to have the ability to extract and preserve local information effectively.
<div align=center><img width="500" height="180" src="https://github.com/sophie091524/Noise-Reduction-in-ECG-Signals/blob/master/pic/fcl_cl.jpg"/></div>

## Conclusion
To the best of our knowledge, this is the first study on 1-D ECG signal using FCN-based DAE for the process of noise removal. Performances of our algorithm shows higher SNRimp, lower RMSE and  lower PRD compared to DNN- and CNN- based DAEs with the same compression ratio. Additionally, the proposed method obtains high compression performance, where each ECG signals with 1024 samples can be successfully reconstructed by representing only 32 dimensions. With high noise reduction and low signal distortion, the practicality and superiority of our method is suitable for clinical diagnosis.

## References
[1] P. Xiong, H. Wang, M. Liu, S. Zhou, Z. Hou, and X. Liu, “Ecg signal enhancement based on improved denoising auto-encoder,” Engineering Applications of Artificial Intelligence, vol. 52, pp. 194–202, 2016.

[2] R. Mark and G. Moody. (1997, May) MIT-BIH Arrhythmia Database. [Online]. Available: http://ecg.mit.edu/dbinfo.html

[3] O. Yildirim, R. San Tan,  and U. R. Acharya, “An  efficient compres-   sion of ecg signals using deep convolutional autoencoders,” Cognitive Systems Research, vol. 52, pp. 198–211, 2018.

## Publication
H.-T. Chiang, Y.-Y. Hsieh, S.-W. Fu, K.-H. Hung, Y. Tsao, and S.-Y. Chien. Noisereduction in ecg signals using fully convo-lutional denoising autoencoders. IEEE Ac-cess, 7:60806–60813, 2019.
