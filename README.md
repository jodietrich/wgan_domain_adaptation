Bridging the Gap Between 1.5 T and 3 T MRI Data Using Generative Adversarial Networks

Semester Project of the autumn semester 2017 by Jonathan Dietrich advised by Dr. Christian F. Baumgartner.

Authors:
- Jonathan Dietrich ([email](mailto:jonathan.dietrich1@gmail.com))
- Christian F. Baumgartner ([email](mailto:baumgartner@vision.ee.ethz.ch))

Abstract:
Medical image analysis using machine learning techniques suffers from a lack of labeled data sets. Specifically,
while some large annotated datasets exist, there are many datasets that have been acquired with different
protocols for which no annotations are available. Training machine learning algorithms on a source
domain where labels are available and applying them to unlabeled target data is thus of great interest. The
majority of existing approaches require at least some supervision in the target domain. Furthermore, only
one related work takes advantage of the predictive power of deep neural networks. In this thesis we investigate
a method which uses generative adversarial networks to perform unsupervised domain adaptation,
meaning no pairs of corresponding images are required. Images are translated from the source domain to
the target domain. This allows translating a labeled data set to another domain, where no labeled images
exist and training a machine learning algorithm on the translated data set. The effectiveness of this domain
adaptation technique is tested by training on 1.5 T MRI data in the ADNI data set and testing on 3 T images
or vice versa. We find that, after our preprocessing, there is only a small domain gap between 1.5 T images
and 3 T. Nevertheless, the studied methods manage to largely bridge this domain gap. Furthermore we test
the influence of conditioning image translation on noise. Whether noise helps to bridge the domain gap is
inconclusive, but we observe that the noise changes the anatomical structure of the output images.

## Requirements 

- Python 3.4 (only tested with 3.4.3)
- Tensorflow >= 1.0 (only tested with 1.1.0)
- The remainder of the requirements are given in `requirements.txt`

