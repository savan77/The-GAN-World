# The GAN World
Everything about Generative Adversarial Networks

## Table of Contents
- [Introduction](#Introduction)
- [Papers and Code](#Papers)
- [Tutorials, Blogs and Talks](#Tutorials-Blogs-Talks)
- [Datasets](#Datasets)
- [Other Resources](#other)
- [Contributing](#contribute)

## Introduction
Generative Adversarial Networks are very popular generative models which can be trained to generate synthetic data that is similar to the training data. Basic idea behind GANs is, we have two models, one called **Generator** and another called **Discriminator**. Generator takes noise as an input and produces synthetic data. Then, this generated data(fake data) along with original data from training dataset is fed into disciminator. Here, discriminator tries to distinguish between original data and fake data. As learning proceeds generator learns to generate more and more realistic data and discriminator learns to get better at distinguishing generated and fake data. Here, we have two networks(generator and discriminator) which we need to train simultaneously. In other words, GANs learn a probability distribution of the training data which we can use later to sample the data from it. GANs are also famous for their unstable training, they are hard to train. But we have made great progress in this field especially in image generation. As of now, we have GAN models which can generate high-resolution realistic images.  GANs are so popular that every week new paper on GAN is coming out. This repository contains various resources which can be used to learn or implement GAN. I will keep updating this repository with latest resources and I also intend to add jupyter notebooks on GAN soon.

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

### Softmax GAN [[Paper](https://arxiv.org/abs/1704.06191)]
* [TensorFlow and PyTorch implementation of Softmax GAN](https://github.com/wiseodd/generative-models/tree/master/GAN/softmax_gan)

### Cycle GAN [[Paper](https://arxiv.org/pdf/1703.10593.pdf)]
* [TensorFlow implementaion of Cycle GAN](https://github.com/XHUJOY/CycleGAN-tensorflow)
* [Torch implementation of Cycle GAN](https://github.com/junyanz/CycleGAN)

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
* [LSUN](http://www.yf.io/p/lsun)

## Other Resources
* [Last chapter of Deep Learning Book : Deep Generative Models](https://www.deeplearningbook.org/contents/generative_models.html)
* [The GAN Zoo](https://deephunt.in/the-gan-zoo-79597dc8c347)

## Contributing
* Feel free to make pull requests or you can write me at **vsavan7@gmail.com**.
