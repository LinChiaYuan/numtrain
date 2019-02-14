import numpy as np

class ReadCsv:

    #   ('tesr123.csv',1,785,'float32',0,'int',28,28)
    def ReadCsvToImageArray(Filename,X,X2,X_type,Y,Y_type,ROW,COL):
        my_data = np.genfromtxt(Filename, delimiter=',', skip_header=0)
        X_train = my_data[:, X:X2]
        X_train = X_train.astype(X_type)
        Y_train = my_data[:, Y]
        Y_train = Y_train.astype(Y_type)
        X_train = X_train.reshape(X_train.shape[0], 1,ROW,COL)
        return X_train, Y_train

    #  ('tesr123.csv',0,784,'float32',28,28)
    def ReadCsvToImageArray2(Filename,X,X2,X_type,ROW,COL):
        my_data = np.genfromtxt(Filename, delimiter=',', skip_header=0)
        X_train = my_data[:, X:X2]
        X_train = X_train.astype(X_type)
        X_train = X_train.reshape(X_train.shape[0], 1,ROW,COL)
        return X_train

    def WriteCsv(EndNum):
        o = open("EndCFile.csv", "w")
        images = []
        image = []
        image.append("id")
        image.append("label")
        images.append(image)
        for i in range(10000):
            image =[]
            image.append(i)
            image.append(EndNum[i])
            images.append(image)
        for image in images:
            o.write(",".join(str(pix) for pix in image) + "\n")
        o.close()

