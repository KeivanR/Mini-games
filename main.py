import tensorflow as tf
from tensorflow.keras import layers

moves = ['p','c','f']
learning_rate = .01
timestep = 16
batchsize = 10
# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
def loss(y_truth,y_pred):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred,y_truth))#(y_pred-y_truth)**2
# Given a callable model, inputs, outputs, and a learning rate...
def train(model, x, y_truth, learning_rate):

    with tf.GradientTape() as t:
    # Trainable variables are automatically tracked by GradientTape
        current_loss = loss(y_truth,model(x))
    # Use GradientTape to calculate the gradients with respect to W and b
    dweights = t.gradient(current_loss, model.trainable_weights)
    '''
    # Subtract the gradient scaled by the learning rate
    for dw,w in zip(dweights,model.trainable_weights):
        w.assign_sub(learning_rate * dw)
    '''
    # Update the weights of the model.
    optimizer.apply_gradients(zip(dweights, model.trainable_weights))
    return current_loss
def int2vec(x):
    return tf.cast(tf.equal(tf.range(3),x),tf.float32)

####CREATE MODEL###
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(timestep,6,)))
model.add(layers.GRU(16))
#model.add(tf.keras.Input(shape=(timestep*2,)))
#model.add(layers.Dense(8,activation='tanh'))
model.add(layers.Dense(3,activation = 'softmax'))

model.summary()    



auto = True
x = tf.zeros((batchsize,timestep,6))
y=tf.zeros((batchsize,3))
y_pred=tf.zeros((batchsize,3))
ylast = 0
number_of_moves = 10000
accuracy = 0
pause_size = 50
for move in range(number_of_moves):
    if not auto:
        ylast = input('Chi-Fou-Mi!')
        ylast = tf.cast(tf.where(tf.equal(moves,ylast))[0,0],tf.int32)
    else:
        #ylast+=1
        #ylast = ylast%3
        ylast = (int(tf.math.round(y_pred[-1,0]))-1)%3
    ylast = int2vec((ylast-1)%3)
    y = tf.concat((y[1:],[ylast]),0)
    y_pred = model(x)

    accuracy += tf.cast(tf.argmax(y[-1])==tf.argmax(y_pred[-1]),tf.float32)
    
    current_loss = train(model, x, y, learning_rate)
    
    xlast = tf.concat((x[-1,1:],[tf.concat((y_pred[-1],y[-1]),0)]),0)
    x = tf.concat((x[1:],[xlast]),0)
    if move%pause_size==0:
        print(int(100*accuracy/pause_size),'% of computer wins')
    if move%pause_size==0 or not auto:
        print('User ' ,moves[(tf.argmax(y[-1])+1)%3])
        print('Computer: ',moves[tf.argmax(y_pred[-1])])
        
        print('ypred' ,y_pred[-1])
        print('LOSS: ',current_loss)
        if 100*(accuracy/pause_size)>90:
            ans = input('If you want manual, press m\n') 
            auto = ans!='m'
        accuracy = 0

