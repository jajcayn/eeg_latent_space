# Towards a dynamical understanding of microstate analysis of M/EEG data

*Accompanying code for Jajcay & Hlinka: Towards a dynamical understanding of microstate analysis of M/EEG data (2023), preprint almost submitted to bioRxiv*

**Tidying up in progress**

## Abstract

One of the interesting aspects of EEG data is the presence of temporally stable and spatially coherent patterns of activity, known as microstates, which have been linked to various cognitive and clinical phenomena. However, there is still no general agreement on the validity of microstate analysis, and it is unclear whether the base assumption of microstate analysis is indeed true. Various clustering algorithms have been used for microstate derivation, and some studies suggest that the microstate time series may provide insight into the neural activity of the brain in the resting state. This study aims to test the hypothesis that dynamical microstate properties might be obtainable from linear characteristics of the underlying EEG signal, in particular, by estimating dynamical microstate properties from the cross-covariance and autocorrelation structure of the EEG data. In the first step, we generated a Fourier transform surrogate EEG signal and compared microstate properties in real and synthetic datasets and found that these are largely similar, thus hinting at the fact that microstate properties depend to a very high degree on the linear covariance and autocorrelation structure of the underlying EEG data. Finally, we treated the EEG data as a vector autoregression process, estimated its parameters, and generated surrogate stationary and linear data from fitted VAR. We conclude that most microstate measures are comparable to those estimated from real EEG data, which strengthens our conclusion.
