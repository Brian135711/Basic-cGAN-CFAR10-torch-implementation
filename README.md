This is a basic implementation of a conditional generative adversarial network (cGAN) to facilitate learning for both me and an opportunity for anyone who would like contribute. A generator generates images based on random noise as a normal GAN, but also takes in a vector of 10 class labels from the CFAR10 dataset ['horse', 'dog', ..., 'cat']. These labels are converted into an imbedding before insertion into the generator and discriminator.  The discriminator judges real or fake and updates weights of the generator as a normal GAN except the idea is that learning and generation can be controlled by these labels as opposed to a normal GAN where the user has no control over what type of image is generated. Here is a link to the original paper Conditional Generative Adversarial Nets by Mirza, and Osindero https://arxiv.org/abs/1411.1784, The notebook version works better and is largely based on https://github.com/bhiziroglu/Conditional-Generative-Adversarial-Network 

The resolution of the original images seems to limit the ability to produce clear images. The basic network structure probably slows the time to convergence, but makes the code somewhat easier to digest. I think that possible areas for improvement might be the way in which the embedding process is implemented as well. Good starting point for learning but, I will probably be looking at other more current methods. 

