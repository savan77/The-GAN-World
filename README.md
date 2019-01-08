# The GAN World
Everything about Generative Adversarial Networks

## Table of Contents
- [Introduction](#Introduction)
- [Papers and Code](#Papers)
- [Projects](#Projects)
- [Tutorials, Blogs and Talks](#Tutorials-Blogs-Talks)
- [Datasets](#Datasets)
- [Other Resources](#other)
- [Contributing](#contribute)

## Introduction
Generative Adversarial Networks are very popular generative models which can be trained to generate synthetic data that is similar to the training data. Basic idea behind GANs is, we have two models, one called **Generator** and another called **Discriminator**. Generator takes noise as an input and produces synthetic data. Then, this generated data(fake data) along with original data from training dataset is fed into disciminator. Here, discriminator tries to distinguish between original data and fake data. As learning proceeds generator learns to generate more and more realistic data and discriminator learns to get better at distinguishing generated and fake data. In other words, GANs learn a probability distribution of the training data which we can use later to sample the data from it. Here, we have two networks(generator and discriminator) which we need to train simultaneously. GANs are also famous for their unstable training, they are hard to train. But we have made great progress in this field especially in image generation. As of now, we have GAN models which can generate high-resolution realistic images.  GANs are so popular that every week new paper on GAN is coming out. This repository contains various resources which can be used to learn or implement GANs.

## Papers and Code

### Generative Adversarial Networks [[Paper](https://arxiv.org/abs/1406.2661)]
* [Vanilla GAN implementation in PyTorch and TensorFlow](https://github.com/wiseodd/generative-models/tree/master/GAN/vanilla_gan)

### Deep Convolutional Generative Adversarial Networks [[Paper](https://arxiv.org/abs/1511.06434)]
* [A tensorflow implementation of "Deep Convolutional Generative Adversarial Networks"](https://github.com/carpedm20/DCGAN-tensorflow)
* [A torch implementation of DCGAN](https://github.com/soumith/dcgan.torch)

### Wasserstein GAN [[Paper](https://arxiv.org/abs/1701.07875)]
* [PyTorch implementation of Wasserstein GAN](https://github.com/martinarjovsky/WassersteinGAN)
* [TensorFlow implementation of Wasserstein GAN](https://github.com/shekkizh/WassersteinGAN.tensorflow)

### Bayesian GAN [[Paper](https://arxiv.org/abs/1705.09558)]
* [TensorFlow implementation of Bayesian GAN](https://github.com/andrewgordonwilson/bayesgan/)

### DiscoGAN [[Paper](https://arxiv.org/abs/1703.05192)]
* [PyTorch implementation of Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://github.com/carpedm20/DiscoGAN-pytorch)
* [TensorFlow and PyTorch implementation of DiscoGAN](https://github.com/wiseodd/generative-models/tree/master/GAN/disco_gan)

### Bayesian GAN [[Paper](https://arxiv.org/abs/1705.09558)]
* [TensorFlow implementation of Bayesian GAN](https://github.com/andrewgordonwilson/bayesgan/)

### Energy-based Generative Adversarial Network [[Paper](https://arxiv.org/abs/1609.03126)]
* [TensorFlow and PyTorch implementaion of EBGAN](https://github.com/wiseodd/generative-models/tree/master/GAN/ebgan)

### Boundary Equilibrium GAN [[Paper](https://arxiv.org/abs/1703.10717)]
* [TensorFlow implementation of Boundary Equilibrium Generative Adversarial Networks](https://github.com/carpedm20/BEGAN-tensorflow)

### Coupled Generative Adversarial Networks [[Paper](https://arxiv.org/abs/1606.07536)]
* [TensorFlow and PyTorch implementation of COGAN](https://github.com/wiseodd/generative-models/tree/master/GAN/coupled_gan)

### MAGAN: Margin Adaptation for Generative Adversarial Networks [[Paper](https://arxiv.org/abs/1704.03817)]
* [TensorFlow and PyTorch implementation of MAGAN](https://github.com/wiseodd/generative-models/tree/master/GAN/magan)

### InfoGAN : Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets [[Paper](https://arxiv.org/abs/1606.03657)]
* [TensorFlow and PyTorch implementation of InforGAN](https://github.com/wiseodd/generative-models/tree/master/GAN/infogan)

### SEGAN : Speech Enhancement Generative Adversarial Networks [[Paper](https://arxiv.org/pdf/1703.09452.pdf)]
* [TensorFlow implementation of SEGAN](https://github.com/santi-pdp/segan)

### Conditional Generative Adversarial Nets [[Paper](https://arxiv.org/abs/1411.1784)]
* [TensorFlow and PyTorch implementation of CGAN](https://github.com/wiseodd/generative-models/tree/master/GAN/conditional_gan)

### Boundary-Seeking Generative Adversarial Networks [[Paper](https://arxiv.org/abs/1702.08431)]
* [TensorFlow and PyTorch implementation of Boundary-Seeking GAN](https://github.com/wiseodd/generative-models/tree/master/GAN/boundary_seeking_gan)

### Softmax GAN [[Paper](https://arxiv.org/abs/1704.06191)]
* [TensorFlow and PyTorch implementation of Softmax GAN](https://github.com/wiseodd/generative-models/tree/master/GAN/softmax_gan)

### Cycle GAN [[Paper](https://arxiv.org/pdf/1703.10593.pdf)]
* [PyTorch implementation of Cycle GAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* [TensorFlow implementaion of Cycle GAN](https://github.com/XHUJOY/CycleGAN-tensorflow)
* [Torch implementation of Cycle GAN](https://github.com/junyanz/CycleGAN)

### GAWWN : Generative Adversarial What-Where Network [[Paper](http://www.scottreed.info/files/nips2016.pdf)]
* [Torch implementation of GAWWN](https://github.com/reedscot/nips2016)

###  StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks [[Paper](https://arxiv.org/pdf/1612.03242v1.pdf)]
* [TensorFlow implementation of StackGAN](https://github.com/hanzhanggit/StackGAN)

### End-to-end Adversarial Learning for Generative Conversational Agents [[Paper](https://arxiv.org/abs/1711.10122)]
* [Keras implementation](https://github.com/oswaldoludwig/Adversarial-Learning-for-Generative-Conversational-Agents)

### StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation [[Paper](https://arxiv.org/abs/1711.09020)]
* [PyTorch implementation of StarGAN](https://github.com/yunjey/StarGAN) 

### Unsupervised Cross-Domain Image Generation [[Paper](https://arxiv.org/abs/1611.02200)]
### Generative Adversarial Nets from a Density Ratio Estimation Perspective [[Paper](https://arxiv.org/abs/1610.02920)]
### BCGAN : Bayesian Conditional Generative Adverserial Networks [[Paper](https://arxiv.org/abs/1706.05477)]
### SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient [[Paper](https://arxiv.org/abs/1609.05473v5)]
### Gang of GANs : Generative Adversarial Networks with Maximum Margin Ranking [[Paper](https://arxiv.org/abs/1704.04865)]
### SketchGAN : Adversarial Training For Sketch Retrieval [[Paper](https://arxiv.org/abs/1607.02748)]
### Unrolled Generative Adversarial Networks [[Paper](https://arxiv.org/abs/1611.02163)]
### TextureGAN : Controlling Deep Image Synthesis with Texture Patches [[Paper](https://arxiv.org/abs/1706.02823)]
* [PyTorch implementation of TextureGAN](https://github.com/janesjanes/Pytorch-TextureGAN)
### Temporal Generative Adversarial Nets [[Paper](https://arxiv.org/abs/1611.06624v1)]
### Recurrent Topic-Transition GAN for Visual Paragraph Generation [[Paper](https://arxiv.org/abs/1703.07022)]
### Triangle Generative Adversarial Networks [[Paper](https://arxiv.org/abs/1709.06548)]
### AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks [[Paper](https://arxiv.org/abs/1711.10485)]
### Structured Generative Adversarial Networks [[Paper](https://arxiv.org/abs/1711.00889)]
### BigGan: Large Scale GAN Training for High Fidelity Natural Image Synthesis [[Paper](https://arxiv.org/abs/1809.11096)]
* [PyTorch implementation BigGan](https://github.com/AaronLeong/BigGAN-pytorch)


## Projects
* Image Completion with Deep Learning in TensorFlow[[Blog](http://bamos.github.io/2016/08/09/deep-completion/)][[Github](https://github.com/bamos/dcgan-completion.tensorflow)]
* Image Super Resolution with Deep Learning[[Github](https://github.com/david-gpu/srez)]
* Neural Photo Editor : A simple interface for editing natural photos with generative neural networks[[Github](https://github.com/ajbrock/Neural-Photo-Editor)]
* iGAN : Interactive Image Generation via Generative Adversarial Networks[[Github](https://github.com/junyanz/iGAN)]
* CleverHans : A library for benchmarking vulnerability to adversarial examples[[Github](https://github.com/tensorflow/cleverhans)]
* VideoGAN : Generating Videos with Scene Dynamics[[Blog](http://carlvondrick.com/tinyvideo/)][[Github](https://github.com/cvondrick/videogan)]


## Tutorials, Blogs and Talks
* [NIPS 2016 Tutorial on Generative Adversarial Networks by Ian Goodfellow](https://arxiv.org/abs/1701.00160) - This tutorial by Ian Goodfellow (Inventor of GAN) covers almost everything you need to get started with Generative Adversarial Networks. You will get to know about- Why you should study generative models and GANs?, How GAN works?, Research frontiers in GANs and more. 
* [GANs in Action: Deep learning with Generative Adversarial Networks](https://www.manning.com/books/gans-in-action) This book takes you from no knowledge of GANs to understanding and implementing some of the more advanced architectures at the practitioner level. Focus on applications and code.

### Blogs

* [Generative Adversarial Networks in 50 lines of code (PyTorch)](https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f)
* [Generative Models by OpenAI](https://blog.openai.com/generative-models/)
* [How to train a GAN? Tips and Tricks to make GANs work](https://github.com/soumith/ganhacks)
* [Generative Adversarial Networks in TensorFlow](http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/)
* [GANs, some open questions](http://www.offconvex.org/2017/03/15/GANs/)
* [An Introduction to GAN (TensorFlow)](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/)
* [BEGAN : State of the art generation of the faces with generative adversarial networks](https://blog.heuritech.com/2017/04/11/began-state-of-the-art-generation-of-faces-with-generative-adversarial-networks/)
* [SimGANs : a game changer in unsupervised learning, self-driving cars and more](https://blog.waya.ai/simgans-applied-to-autonomous-driving-5a8c6676e36b)
* [MNIST Generative Adversarial Model in Keras](https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/)


### Talks
* [NIPS 2016 :  Generative Adversarial Network by Ian Goodfellow [Video]](https://www.youtube.com/watch?v=AJVyzd0rqdc)
* [NIPS 2016 workshop on Adversarial Training [7 videos]](https://www.youtube.com/watch?v=RvgYvHyT15E&list=PLJscN9YDD1buxCitmej1pjJkR5PMhenTF)
* [Generalization and Equilibrium in Generative Adversarial Nets (GANs)[Video]](https://www.youtube.com/watch?v=V7TliSCqOwI)

## Datasets
* [CelebA : 202,599 number of face images](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
* [MNIST : 70,000 images of hand-written digits](http://yann.lecun.com/exdb/mnist/)
* [Imagenet](http://www.image-net.org/)
* [Cifar 10 and 100](https://www.cs.toronto.edu/~kriz/cifar.html)
* [COCO](http://mscoco.org/dataset/#overview)
* [LSUN](http://www.yf.io/p/lsun)

## Other Resources
* [Last chapter of Deep Learning Book : Deep Generative Models](https://www.deeplearningbook.org/contents/generative_models.html)
* [The GAN Zoo](https://deephunt.in/the-gan-zoo-79597dc8c347)
* [How to Train a GAN? Tips and tricks to make GANs work.](https://github.com/soumith/ganhacks)
* [Collection of generative models in PyTorch and TensorFlow](https://github.com/wiseodd/generative-models)
* [keras Implementation of Generative Adversarial Networks](https://github.com/eriklindernoren/Keras-GAN)

## Contributing
* Feel free to make pull requests or you can write me at **vsavan7@gmail.com**.
