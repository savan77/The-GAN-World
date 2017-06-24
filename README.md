# The GAN World
Everything about Generative Adversarial Networks

## Table of Contents
- [Introduction](#Introduction)
- [Papers and Code](#Papers)
- [Tutorials, Blogs and Talks](#Tutorials-Blogs-Talks)
- [Datasets](#Datasets)
- [Other Resources](#other)

## Introduction
Generative Adversarial Networks are very popular generative models which can be trained to generate synthetic data that is similar to the training data. Basic idea behind GANs is, we have two models, one called **Generator** and another called **Discriminator**. Generator takes some noise as an input and produces synthetic data. Then, this generated data(fake data) along with original data from training dataset is fed into disciminator. Here, discriminator tries to distinguish between original data and fake data. In other words, GANs learn a probability distribution of the training data which we can use later to sample the data from it. GANs are so popular that every week new paper on GAN is coming out.

## Papers and Code

### Generative Adversarial Networks [[Paper](https://arxiv.org/abs/1406.2661)]
* [Vanilla GAN implementation in PyTorch and TensorFlow](https://github.com/wiseodd/generative-models/tree/master/GAN/vanilla_gan)

### Deep Convolutional Generative Adversarial Networks [[Paper](https://arxiv.org/abs/1511.06434)]
* [A tensorflow implementation of "Deep Convolutional Generative Adversarial Networks"](https://github.com/carpedm20/DCGAN-tensorflow)
* [A torch implementation of DCGAN](https://github.com/soumith/dcgan.torch)

### Wasserstein GAN [[Paper](https://arxiv.org/abs/1701.07875)]
* [PyTorch implementation of Wasserstein GAN](https://github.com/martinarjovsky/WassersteinGAN)
* [TensorFlow implementation of Wasserstein GAN](https://github.com/shekkizh/WassersteinGAN.tensorflow)

### DiscoGAN [[Paper](https://arxiv.org/abs/1703.05192)]
* [PyTorch implementation of Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://github.com/carpedm20/DiscoGAN-pytorch)
* [TensorFlow and PyTorch implementation of DiscoGAN](https://github.com/wiseodd/generative-models/tree/master/GAN/disco_gan)

### Energy-based Generative Adversarial Network [[Paper](https://arxiv.org/abs/1609.03126)]
* [TensorFlow and PyTorch implementaion of EBGAN](https://github.com/wiseodd/generative-models/tree/master/GAN/ebgan)

### Boundary Equilibrium GAN [[Paper](https://arxiv.org/abs/1703.10717)]
* [TensorFlow implementation of Boundary Equilibrium Generative Adversarial Networks](https://github.com/carpedm20/BEGAN-tensorflow)

### Coupled Generative Adversarial Networks [[Paper](https://arxiv.org/abs/1606.07536)]
* [TensorFlow and PyTorch implementation of COGAN](https://github.com/wiseodd/generative-models/tree/master/GAN/coupled_gan)

### MAGAN: Margin Adaptation for Generative Adversarial Networks [[Paper](https://arxiv.org/abs/1704.03817)]
* [TensorFlow and PyTorch implementation of MAGAN](https://github.com/wiseodd/generative-models/tree/master/GAN/magan)

### InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets [[Paper](https://arxiv.org/abs/1606.03657)
* [TensorFlow and PyTorch implementation of InforGAN](https://github.com/wiseodd/generative-models/tree/master/GAN/infogan)

### Conditional Generative Adversarial Nets [[Paper](https://arxiv.org/abs/1411.1784)]
* [TensorFlow and PyTorch implementation of CGAN](https://github.com/wiseodd/generative-models/tree/master/GAN/conditional_gan)

### Boundary-Seeking Generative Adversarial Networks [[Paper](https://arxiv.org/abs/1702.08431)]
* [TensorFlow and PyTorch implementation of Boundary-Seeking GAN](https://github.com/wiseodd/generative-models/tree/master/GAN/boundary_seeking_gan)

## Tutorials, Blogs and Talks
* [NIPS 2016 Tutorial on Generative Adversarial Networks by Ian Goodfellow](https://arxiv.org/abs/1701.00160) - This tutorial by Ian Goodfellow (Inventor of GAN) covers almost everything you need to get started with Generative Adversarial Networks. You will get to know about- Why you should study generative models and GANs?, How GAN works?, Research frontiers in GANs and more. 
* [Generative Adversarial Networks in 50 lines of code (PyTorch)](https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f)
* [Generative Models by OpenAI](https://blog.openai.com/generative-models/)
* [Generative Adversarial Networks in TensorFlow](http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/)
* [GANs, some open questions](http://www.offconvex.org/2017/03/15/GANs/)
* [An Introduction to GAN (TensorFlow)](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/)
* [NIPS 2016 :  Generative Adversarial Network by Ian Goodfellow [Video]](https://www.youtube.com/watch?v=AJVyzd0rqdc)
* [NIPS 2016 workshop on Adversarial Training [7 videos]](https://www.youtube.com/watch?v=RvgYvHyT15E&list=PLJscN9YDD1buxCitmej1pjJkR5PMhenTF)

## Datasets
* [CelebA : 202,599 number of face images](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
* [MNIST : 70,000 images of hand-written digits](http://yann.lecun.com/exdb/mnist/)

## Other Resources
* [Last chapter of Deep Learning Book : Deep Generative Models](https://www.deeplearningbook.org/contents/generative_models.html)
