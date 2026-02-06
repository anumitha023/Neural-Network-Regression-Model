# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: ANUMITHA M R
### Register Number: 212223040018
```

class Neuralnet(nn.Module):
   def __init__(self):
        super().__init__()
        self.n1=nn.Linear(1,10)
        self.n2=nn.Linear(10,20)
        self.n3=nn.Linear(20,1)
        self.relu=nn.ReLU()
        self.history={'loss': []}
   def forward(self,x):
        x=self.relu(self.n1(x))
        x=self.relu(self.n2(x))
        x=self.n3(x)
        return x


# Initialize the Model, Loss Function, and Optimizer
nithi=NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(nithi.parameters(),lr=0.001)

def train_model(nithi, X_train, y_train, criterion, optimizer, epochs=1000):
    # initialize history before loop
    nithi.history = {'loss': []}

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = nithi(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # record loss
        nithi.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```
## Dataset Information

<img width="209" height="532" alt="image" src="https://github.com/user-attachments/assets/652c18ee-6c0e-48f7-8799-83a54aa7d48c" />

## OUTPUT

<img width="402" height="140" alt="image" src="https://github.com/user-attachments/assets/ac83069c-0dae-44fd-976c-3661e3e975da" />


### Training Loss Vs Iteration Plot

<img width="773" height="532" alt="image" src="https://github.com/user-attachments/assets/60f17d29-eae5-4726-8c8d-db0010e55d12" />


### New Sample Data Prediction

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = nithi(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')

## RESULT

Successfully executed the code to develop a neural network regression model.

