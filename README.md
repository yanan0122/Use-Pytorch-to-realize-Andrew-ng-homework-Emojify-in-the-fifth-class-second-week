# Use-Pytorch-to-realize-Andrew-ng-homework-Emojify-in-the-fifth-class-second-week
Add an appropriate emoji to your sentence automatically by LSTM, realized by pytorch

## Project Introduction
Inspired by Andrew's homework Emojify, I hope to realize the model by myself by pytorch. 
This project uses LSTM to realize RNN. The embedding layer's weight, train dataset and test dataset is provided by Andrew's homework. 
Because the train dataset is small, so I do not update the embedding layer's weight in my training process.  
I got some pretty good result when epoch = 60000. The accuracy with different epochs is like this:    
  
*epoch=10000，train_accuracy = 97.727273, test_accuracy = 85.714286。  
epoch=5000，train_accuracy = 99.242424, test_accuracy = 91.071429。Great!  
epoch=3000，train_accuracy = 92.424242, test_accuracy = 78.571429。*
  
But every time I got different accuracy even if with the same epoch. That's reasonable of course. It won't take so much time to train. So have a try by yourself!
Try the model by change the *s* in function *predict*! **By the way, the length of your sentence should be shorter than 10.**
  
*Try like this:  
s = "I am so hungry"  
you will get:  
I am so hungry🍴  

s = "love you so much"  
you will get:  
love you so much❤️* 
  
Feel Free to change my code and share with us! I will be glad if you contact me to talk about this!  
Besides, I keep the best weight during my train. If you want, email me and I will give you that.  

## Hello guys!This is a simple self-introduction
I am a Chinese boy in sophomore year and just a freshman in AI and deep leaning. My code may have  a lot of problem. I am glad to receive your email to talk about the code and other interesting thing in deep learning. If you have some question about my code, feel free to email me. I will try my best to help you.  
**This is my email: 328375886@qq.com**
