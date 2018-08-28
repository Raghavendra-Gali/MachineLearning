class knn(object):
    
    def __init__(self,neighbors=None,matrix1=None,matrix2=None):
        self.trainData_X = None
        self.trainData_Y = None
        self.neighbors = neighbors
        self.similaritiesMatrix1 = matrix1
        self.similaritiesMatrix2 = matrix2
        
    def fit(self,X,Y):
        self.trainData_X = X
        self.trainData_Y = Y
    
    def min_max(self,i):
        values = []
        for j in range(len(self.trainData_X)):
            values.append(self.trainData_X[j][i])
        return min(values),max(values)
    
    def similarities(self,x1,x2):     
        d= self.similaritiesMatrix1[x1[0]][x2[0]]
        d+= self.similaritiesMatrix2[x1[1]][x2[1]]
        for i in range(2,len(x1)):
            mn,mx = self.min_max(i)
            d+= np.square((x1[i]-x2[i]))
        return np.sqrt(d)   
    
    def Votes(self,lst):
        mode = [None]
        mx=0
        for i in range(len(lst)):
            c=0
            for j in range(len(lst)):
                if lst[i]==lst[j]:
                    c+=1
            if mx<c:
                if lst[i] in mode:
                    continue
                else:
                    mode[0] = lst[i]
                mx = c
            elif mx==c:
                if lst[i] in mode:
                    continue
                else:
                    mode.append(lst[i])
        if not mode[0]:
            return lst
        return mode[0]
     
    def predict(self,X_test,Y_test):
        if not self.neighbors:
            self.neighbors = 3
        predictClass = []
        for e in range(len(X_test)):
            distances={}
            for x_i in range(len(self.trainData_X)):
                dist = self.similarities(X_test[e],self.trainData_X[x_i])
                distances[x_i] = dist
            sorted_d = sorted(distances.items(),key=operator.itemgetter(1))
#             print(sorted_d[len(sorted_d)-self.neighbors:len(sorted_d)])
            kneighbors = [self.trainData_Y[x[0]] for x in sorted_d[:self.neighbors-1]]
#             print("kishore: ",kneighbors)
            kVotes = self.Votes(kneighbors)
#             print(kVotes)
            predictClass.append(kVotes)            
        return predictClass