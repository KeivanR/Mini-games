import rpc_model as rpcm
import tensorflow as tf
from tkinter import *
from PIL import Image, ImageTk

#import blind
windows_path = 'C:/Users/keiva/OneDrive/Documents/GitHub/Mini-games/'
paths = [
windows_path+'pierre.png',
windows_path+'ciseaux.jpg',
windows_path+'feuille.jpg',
windows_path+'table.png'
]
model_path = windows_path+'model'

class Interface(Frame):
    def __init__(self, fenetre,**kwargs):
        Frame.__init__(self, fenetre, width=1000, height=1200, **kwargs)
        self.winfo_toplevel().title("Keivchess")
        self.rowconfigure(0, weight=1)           
        self.columnconfigure(0, weight=1) 
        self.pack(fill=BOTH)
        # Création de nos widgets

        self.bouton_quitter = Button(self, text="Quitter", command=self.quit)
        self.bouton_quitter.grid(row=0, column=2)

        self.bouton_pierre = Button(self, text="Pierre", fg="Gray",command=lambda: self.play_move(0))
        self.bouton_pierre.grid(row=1, column=2,columnspan=1)
        self.bouton_feuille = Button(self, text="Feuille", fg="White",command=lambda: self.play_move(2))
        self.bouton_feuille.grid(row=2, column=2,columnspan=1)
        self.bouton_ciseaux = Button(self, text="Ciseaux", fg="Orange",command=lambda: self.play_move(1))
        self.bouton_ciseaux.grid(row=3, column=2,columnspan=1)
        self.bouton_load = Button(self, text="Load Model", fg="Black",command=lambda: self.load(model_path))
        self.bouton_load.grid(row=4, column=2,columnspan=1)
        self.bouton_save = Button(self, text="Save Model", fg="Black",command=lambda: self.save(model_path))
        self.bouton_save.grid(row=5, column=2,columnspan=1)
        
        self.moves = ['p','c','f']
        self.learning_rate = rpcm.learning_rate
        self.timestep = rpcm.timestep
        self.batchsize = rpcm.batchsize
        self.x = tf.zeros((self.batchsize,self.timestep,6))
        self.y=tf.zeros((self.batchsize,3))
        self.y_pred=tf.zeros((self.batchsize,3))
        self.score = [0,0]
        self.model = rpcm.model
        
        self.bkg = Image.open(paths[3])
        render = ImageTk.PhotoImage(self.bkg)
        img = Label(self, image=render)
        img.image = render
        img.grid(row=0, column=0)
    def display(self,ylast,y_pred):
        self.bkg = Image.open(paths[3])

        load = Image.open(paths[y_pred])
        load = load.resize((100, 100))
        self.bkg.paste(load,(100,10))

        load = Image.open(paths[ylast])
        load = load.resize((100, 100))
        self.bkg.paste(load,(350,160))

        render = ImageTk.PhotoImage(self.bkg)
        img = Label(self, image=render)
        img.image = render
        img.grid(row=0, column=0)
    def load(self,model_path):
        self.model.load_weights(model_path)
    def save(self,model_path):
        self.model.save_weights(model_path)
    def play_move(self,ylast):
        ylast_desired = rpcm.int2vec((ylast-1)%3)
        self.y = tf.concat((self.y[1:],[ylast_desired]),0)
        if rpcm.recurrent:
            xr = self.x
        else:
            xr = tf.reshape(self.x,(self.x.shape[0],self.x.shape[1]*self.x.shape[2]))
        self.y_pred = self.model(xr)
        
        self.display(ylast,tf.argmax(self.y_pred[-1]))
        
        current_loss = rpcm.train(self.model, xr, self.y, self.learning_rate)
        
        xlast = tf.concat((self.x[-1,1:],[tf.concat((self.y_pred[-1],self.y[-1]),0)]),0)
        self.x = tf.concat((self.x[1:],[xlast]),0)
        print('User ' ,self.moves[ylast])
        print('Computer: ',self.moves[tf.argmax(self.y_pred[-1])])
        
        print('ypred' ,self.y_pred[-1])
        print('LOSS: ',current_loss)
        
        last_score = rpcm.score(ylast,tf.argmax(self.y_pred[-1]))
        self.score[0] += last_score[0]
        self.score[1] += last_score[1]
        self.winfo_toplevel().title("Computer "+str(self.score[1])+" - "+str(self.score[0])+" User")
		

window = Tk()


window.geometry("700x800")
interface = Interface(window)
#window.bind("<Button-1>", interface.callback)
window.mainloop()
interface.destroy()
