import os
os.environ ['TF_CPP_MIN_LOG_LEVEL'] =' 2'

from ReadCsv import ReadCsv
from keras.utils import np_utils
''' EarlyStopping '''
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3)

''' Self-defined Callbacks '''
from keras.callbacks import Callback
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
    def on_epoch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

loss_history=LossHistory()

''' Useful Callback: ModelCheckpoint '''
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(filepath='model.h5',monitor='val_loss',mode='auto',save_best_only=True)

def LoadDate():
    #---------  read csv  ----------#
    X_test ,Y_test  = ReadCsv.ReadCsvToImageArray('mnist_test.csv',1,785,'float32',0,'int',28,28)
    X_train,Y_train = ReadCsv.ReadCsvToImageArray('mnist_train.csv',1,785,'float32',0,'int',28,28)
    X_test  = X_test/255
    X_train = X_train/255
    #---------  one-hot encoding   ----------#
    Y_train = np_utils.to_categorical(Y_train, 10)
    Y_test = np_utils.to_categorical(Y_test, 10)
    return X_train, Y_train, X_test, Y_test

def LoadDate2(do):
    #---------  read csv  ----------#
    X_test ,Y_test  = ReadCsv.ReadCsvToImageArray('mnist_test.csv',1,785,'float32',0,'int',28,28)
    X_test  = X_test/255
    #---------  one-hot encoding   ----------#
    if do ==1:
        Y_test = np_utils.to_categorical(Y_test, 10)
    return X_test, Y_test

def LoadDate3():
    #---------  read csv  ----------#
    X_test   = ReadCsv.ReadCsvToImageArray2('tesr123.csv',0,784,'float32',28,28)
    X_test  = X_test/255
    return X_test

def ModelCreate(x_train, y_train):
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.layers import MaxPooling2D, Flatten
    from keras.layers.convolutional import Conv2D
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(1, 28, 28), data_format="channels_first",activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),data_format="channels_first"))
    model.add(Conv2D(32, (3, 3), padding='same',data_format="channels_first", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    print("to compile")
    from keras.optimizers import Adam
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #from keras.optimizers import SGD, Adam, RMSprop, Adagrad
    #sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    #model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    print("compile end")
    TrainHistory = model.fit(x_train, y_train, batch_size=100, epochs=10, verbose=1, shuffle=True, validation_split=0.1,callbacks=[early_stopping, loss_history, checkpoint])
    return model, TrainHistory

def PlotHistory(History):
    loss_adam = History.history.get('loss')
    acc_adam = History.history.get('acc')
    val_loss_adam = History.history.get('val_loss')
    val_acc_adam = History.history.get('val_acc')
    loss = loss_history.loss
    acc = loss_history.acc
    val_loss = loss_history.val_loss
    val_acc = loss_history.val_acc
    import matplotlib.pyplot as plt
    plt.figure(0)
    plt.subplot(121)
    plt.plot(range(len(loss_adam)), loss_adam, label='Training')
    plt.plot(range(len(val_loss_adam)), val_loss_adam, label='Validation')
    plt.title('Loss history returned by fit function')
    plt.legend(loc='upper left')
    plt.subplot(122)
    plt.plot(range(len(loss)), loss, label='Training')
    plt.plot(range(len(val_loss)), val_loss, label='Validation')
    plt.title('Loss history from Callbacks')
    plt.savefig('09_callback.png', dpi=300, format='png')
    plt.close()

#----- start -------#
print("load date .....")
What_to_Do = 1
What_to_Do2 = 0
if What_to_Do == 1 :
    X_train, Y_train, X_test, Y_test = LoadDate()
    print("train .....")
    model,TrainHistory = ModelCreate(X_train, Y_train)
    print("PlotHistory .....")
    PlotHistory(TrainHistory)
    print("prediction rate.....")
    scores = model.evaluate(X_test, Y_test,verbose=0)
    print(scores)
    print(scores[1])
elif What_to_Do == 2:
    X_test, Y_test = LoadDate2(What_to_Do2)
    print("read model .....")
    from keras.models import load_model
    model_test = load_model("numswitch.h5")
    if What_to_Do2 ==1:
        scores = model_test.evaluate(X_test, Y_test, verbose=0)
        print(scores)
        print(scores[1])
    else:
        p = model_test.predict_classes(X_test)
        errornu=0
        for i in range(Y_test.shape[0]):
            if Y_test[i] != p[i]:
                print("TEST : " ,Y_test[i] ,", P : " ,p[i])
                errornu +=1
        print("error : " + str(errornu))
else:
    X_test = LoadDate3()
    print("read model .....")
    from keras.models import load_model
    model_test = load_model("numswitch.h5")
    p = model_test.predict_classes(X_test)

    #print("TEST1 : " +  str(p[0]))
    ReadCsv.WriteCsv(p)