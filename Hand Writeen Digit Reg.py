#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf


# In[2]:


mnist =tf.keras.datasets.mnist #for downloading setdata


# # spliting datasets into train data and test data

# In[3]:


(x_train,y_train),(x_test,y_test)= mnist.load_data()


# In[4]:


x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)


# In[5]:


model =tf.keras.models.Sequential()


# In[6]:


model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))


# In[7]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[49]:


model.fit(x_train, y_train, epochs=15
         )
model.save('handwritten.model')


# In[50]:


model=tf.keras.models.load_model('handwritten.model')


# In[51]:


loss,accuracy =model.evaluate(x_test,y_test)


# In[52]:


print(loss)


# In[53]:


print(accuracy)


# In[60]:


image_number=7


# In[61]:


while os.path.isfile(f"digits/{image_number}.jpg"):
    try:
        img=cv2.imread(f"digits/{image_number}.jpg")[:,:,0]
        img=np.invert(np.array([img]))
        prediction=model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
        image_number+=1
        
    except:
          print("Error!")
   # finally:
      #  image_number+=1
        


# In[ ]:





# In[ ]:





# In[ ]:




