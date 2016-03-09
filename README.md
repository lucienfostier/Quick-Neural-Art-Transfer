# Quick-Neural-Art-Transfer
Theano/Lasagne based Neural artistic style transfer with Kivy GUI

based on https://github.com/Lasagne/Recipes/tree/master/examples/styletransfer an implementation of the algorithm described in "A Neural Algorithm of Artistic Style" (http://arxiv.org/abs/1508.06576) by Gatys, Ecker and Bethge. 

use VGG19 model http://www.robots.ox.ac.uk/~vgg/research/very_deep/

Several modification are made to speed up the process. 
1. Doing the transfer in pyramid manner to speed up and seems generating better results.
2. Has an option to use ADAM as the optimizer.
3. Use shared memory and can precompile theanot functions. 
4. Has a server and a Kivy GUI client, so you can capture image using a mobile device and the use the GPU on a remote machine.

content and styles
Many image files are from https://github.com/jcjohnson/neural-style and some are public domain

CC-By  http://agf81.deviantart.com/art/Ivy-Texture-7-326197006?q=gallery%3ACreative-Commons%2F26171837&qo=33
Ivy texture

CC-BY-SA   http://maadmann.deviantart.com/art/Fullcolor-437966862?q=gallery%3ACreative-Commons%2F26171826&qo=88
fullcolor

CC-BY https://www.flickr.com/photos/fontplaydotcom/24795772945/in/pool-734017@N25/
color squares

CCB-BY-SA 
http://oilsoaked.deviantart.com/art/holiday-188505340?q=gallery%3ACreative-Commons%2F26171826&qo=373
Holidy

CC-BY
https://www.flickr.com/photos/two/236883835/in/photolist-mW6j4-mW6kn-Jy6XP-Jy9fR-JxQD8-mW6gp-JxQGc-Jy5Xf-Jy3ES-mW6fr-Jy9eM-mW6hW-JxQqD-JxMdm-mW6nF-JxQtD-JxQxR-mW6mA-JxLZS-Jy72p-JxM2N-Jy3Hh-JxLU7-JxLWb-JxQsz-oH3ixB-4Nri37-JxQJr-JxQFe-Jy3AY-JxMcq-Jy3xG-JxQvT-Jy95i-Jy99n-JxQz2-a97MmE-JxQm6-Jy5V5-a97Nqb-a97PVu-Jy97c-a94ZyK-ADmwE6-B3g6KV-ADmtTV-BysQ61-a97Ptd-a94Zex-dwVdG
ndhu

CC-BY
https://www.flickr.com/photos/jphotos/5998521863/in/photolist-mW6gp-JxQGc-Jy5Xf-Jy3ES-mW6fr-Jy9eM-mW6hW-JxQqD-JxMdm-mW6nF-JxQtD-JxQxR-mW6mA-JxLZS-Jy72p-JxM2N-Jy3Hh-JxLU7-JxLWb-JxQsz-oH3ixB-4Nri37-JxQJr-JxQFe-Jy3AY-JxMcq-Jy3xG-JxQvT-Jy95i-Jy99n-JxQz2-a97MmE-JxQm6-Jy5V5-a97Nqb-a97PVu-Jy97c-a94ZyK-ADmwE6-B3g6KV-ADmtTV-BysQ61-a97Ptd-a94Zex-dwVdG-fJCm8H-ih4o2w-farbgP-ez8gpV-a244iL
ndhu2

CC-BY https://www.flickr.com/photos/kurtbudiarto/7257851556/in/photolist-c4mopE-jZZkdx-8bohuZ-bjLvVe-c4wRru-gkp48t-hPNbbL-qfySJz-c84E3h-fBV6ve-aijVu7-fCaHPs-qZ7XnX-f8zn77-eQRUQ5-695MbG-dQNbTm-rowN34-6o4FFz-d3MjtL-9htwN2-r6Yib1-fCau8q-fCaCh9-551xgk-bkueEX-fBVtqt-pbJvac-fCbdY9-aRQ2oH-oUai6n-fCbaRs-fKzB9m-fCbmqA-7kBUGU-eX5Ce9-fBVu9x-ftqE1w-aihyDg-bku5n4-54rawt-nixm7H-dDjykx-55JGBX-c4mnWy-bkudkH-pRMC8L-55JGWa-5A9m5g-6uijiE/
smile face
