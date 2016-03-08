# Quick-Neural-Art-Transfer
Theano/Lasagne based Neural artistic style transfer with Kivy GUI

based on https://github.com/Lasagne/Recipes/tree/master/examples/styletransfer an implementation of the algorithm described in "A Neural Algorithm of Artistic Style" (http://arxiv.org/abs/1508.06576) by Gatys, Ecker and Bethge. 

use VGG19 model http://www.robots.ox.ac.uk/~vgg/research/very_deep/

Several modification are made to speed up the process. 
1. Doing the transfer in pyramid manner to speed up and seems generating better results.
2. Has an option to use ADAM as the optimizer.
3. Use shared memory and can precompile theanot functions. 
4. Has a server and a Kivy GUI client, so you can capture image using a mobile device and the use the GPU on a remote machine.
