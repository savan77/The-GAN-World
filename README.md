# The GAN World
Everything about Generative Adversarial Networks

## Table of Contents
- [Introduction](#Introduction)
- [Papers](#Papers)
- [Tutorials, Blogs and Talks](#Tutorials-Blogs-Talks)
- [Implementations](#Implementations)
- [Datasets](#Datasets)
- [Other Resources](#other)

## Introduction
Generative Adversarial Networks are very popular generative models which can be trained to generate synthetic data that is similar to the training data. Basic idea behind GANs is, we have two models, one called **Generator** and another called **Discriminator**. Generator takes some noise as an input and tries to generate an output similar to the training data. Then, this generated data(fake data) along with original data from training dataset is fed into disciminator. Here, discriminator tries to distinguish between original data and fake data. 

## Papers
* [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) - The original paper on GAN by Ian Goodfellow et al(2014).

## Tutorials, Blogs and Talks
* [NIPS 2016 Tutorial on Generative Adversarial Networks by Ian Goodfellow](https://arxiv.org/abs/1701.00160) - This tutorial by Ian Goodfellow (Inventor of GAN) covers almost everything you need to get started with Generative Adversarial Networks. You will get to know about- Why you should study generative models and GANs?, How GAN works?, Research frontiers in GANs and more. 

## Implementations

### Vanilla GAN
* [Vanilla GAN implementation in PyTorch and TensorFlow](https://github.com/wiseodd/generative-models/tree/master/GAN/vanilla_gan)

## Datasets
* [CelebA : 202,599 number of face images](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
* [MNIST : 70,000 images of hand-written digits](http://yann.lecun.com/exdb/mnist/)

## Other Resources
* [Last chapter of Deep Learning Book : Deep Generative Models](https://www.deeplearningbook.org/contents/generative_models.html)
