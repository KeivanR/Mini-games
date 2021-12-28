import tensorflow as tf
from tensorflow.keras import layers

moves = ['p','c','f']
learning_rate = .0001
timestep = 16
# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
def loss(y_truth,y_pred):
    return (y_pred-y_truth)**2
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
    

####CREATE MODEL###
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(timestep,2,)))
model.add(layers.GRU(16))
#model.add(tf.keras.Input(shape=(timestep*2,)))
#model.add(layers.Dense(8,activation='tanh'))
model.add(layers.Dense(1))

model.summary()    



x = tf.zeros((1,timestep,2))
auto = True
y=0
y_pred = [[0]]
k=0
accuracy = 0
pause_size = 500
while k<100000:
    k +=1
    if not auto:
        y = input('Chi-Fou-Mi!')
        y = tf.cast(tf.where(tf.equal(moves,y))[0,0],tf.float32)
    else:
        #y+=1
        #y = y%3
        y = (int(tf.math.round(y_pred))-1)%3
    y_pred = model(x)
    current_loss = train(model, x, (y-1)%3, learning_rate)
    x = tf.concat((x[:,1:],[[[y_pred[0,0],y]]]),1)
    accuracy += moves[int((y-1)%3)]==moves[int(tf.math.round(y_pred))]
    if k%pause_size==0:
            print(int(100*accuracy/pause_size),'% of computer wins')
    if k%pause_size==0 or not auto:
        print('User ' ,moves[int(y)])
        print('Computer: ',moves[int(tf.math.round(y_pred))])
        
        print('ypred' ,y_pred)
        print('LOSS: ',current_loss)
        if 100*(accuracy/pause_size)>90:
            ans = input('If you want manual, press m') 
            auto = ans!='m'
        accuracy = 0
