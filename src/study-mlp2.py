import torch
from torch import nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
class MyNet(nn.Module):
    def  __init__(self,inputfeat,outputfeat,hidenfeat):
        super(MyNet,self).__init__()
        self.firtlinear=nn.Linear(inputfeat,hidenfeat)
        self.secondlinear=nn.Linear(hidenfeat,outputfeat)
        self.activefuc=nn.ReLU
    
    def forward(self,x):
        #print(" do forward ...")
        # result=self.firtlinear(input)
        # result=self.secondlinear(result)
        # result=self.activefuc(result)
        #x = torch.nn.functional.relu()
        x = torch.nn.functional.sigmoid(self.secondlinear(self.firtlinear(x)))
        #print("forward result ",result)
        return  x  
def main():
    # https://blog.csdn.net/qq_40491305/article/details/106756621
    print("Hello, mlp!")
    net =MyNet(4,3,8)
    loss_func=nn.CrossEntropyLoss()
    print("init params ",net)
    for param in net.parameters():
        nn.init.normal_(param,mean=0,std=0.01)

    optimizes= torch.optim.SGD(net.parameters(),lr=0.01)
    num_epoch=20000
    batch_size=256
    lr=0.1
    
    iris=load_iris()
    #print("load data \n",iris)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                    iris.target, 
                                                    test_size=0.2,  random_state=42)
    print("x train data")                                               
    print(X_train)
    print("y train data")     
    print(y_train)
    X_train=torch.tensor(X_train,dtype=torch.float32)
    X_test=torch.tensor(X_test,dtype=torch.float32)
    y_train=torch.tensor(y_train,dtype=torch.long)
    y_test=torch.tensor(y_test,dtype=torch.long)
    
    # Normalize the features
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    loss_list=[]
    for epoch in range(num_epoch):
        
        y_pred=net(X_train)
        #print("y pred \n",y_pred)
        loss=loss_func(y_pred,y_train)
        # clear gradients
        optimizes.zero_grad()
         # Backward pass and optimization
        loss.backward()
        #print("loss \t ",loss)
        loss_list.append(loss.item())
        # update parameters
        optimizes.step()
        
        # Print the loss every 100 epochs
        if (epoch+1) % 100 == 0:
           print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}')      
           
           
   # Evaluate the model
    with torch.no_grad():
      y_pred = net(X_test)
      _, predicted = torch.max(y_pred, dim=1)
      accuracy = (predicted == y_test).float().mean()
      print(f'Test Accuracy: {accuracy.item():.4f}')
    
    # Plotting the loss after each iteration
    plt.plot(loss_list, 'r')
    plt.tight_layout()
    plt.grid('True', color='y')
    plt.xlabel("Epochs/Iterations")
    plt.ylabel("Loss")
    plt.show()  
 
    
if __name__ == "__main__":
    main()

