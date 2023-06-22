from torch.nn import Linear, Conv2d, MaxPool2d
import torch.nn.functional as F
import torch


class fully_connected(torch.nn.Module):
    """
    Preprosta popolnoma povezava nevronska mreža.
    fully_connected(l1, ..., ln) ustvari NN z n plastmi, velikost i-te plasti je li.

    Input v mrežo je tabela oblike N x n_občin, ki se potem pretvori v en vektor dolžine N*n_obcin
    """

    def __init__(self,M, layer_sizes, out=12, N=14, n_obcin=193):
        '''
        out - število občin na outputu
        Vrne matriko dimenzije MxOut
        '''
        self.M = M
        self.out = out
        if n_obcin*N != layer_sizes[0]:
            print("Prvi layer more bit tok dolg kot št občin * N!")
            raise ValueError
        super().__init__()
        torch.manual_seed(420)
        self.layers = []
        out_size = out*M
        for i, l in enumerate(layer_sizes):
            if i == len(layer_sizes) - 1:
                self.layers.append(Linear(l, out_size))
            else:
                self.layers.append(Linear(l, layer_sizes[i + 1]))
        #self.layers = tuple(self.layers)
        self.layers = torch.nn.ParameterList(self.layers)

    def forward(self, x):
        # reshape
        x = torch.reshape(x, shape=(-1,))
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        x = torch.reshape(x, shape=(self.M, self.out))
        return x


class convolution(torch.nn.Module):
    """
        Uporaba konvolucije + ene linearne plasti na koncu

          Parametri
        ----------
        settings = touple of form (c, l, p), where
            c = kernel size
            l = št out channels
            p = (a,b) = pooling size..
        out = velikost izhoda

    Input v mrežo je tabela oblike N x n_občin
    """

    def __init__(self,M, settings, out=12, N=14, n_obcin=193) -> None:
        super().__init__()
        c, l, p = settings
        self.conv = Conv2d(1, l, c)
        self.pool = MaxPool2d((p[0], p[1]))
        h = int((N - c + 1) / p[0])
        w = int((n_obcin - c + 1) / p[1])
        self.lin = Linear(h * w * l, out*M)
        self.M = M
        self.out = out

    def forward(self, x):
        x = torch.reshape(x, (1,) + tuple(x.shape))
        x = self.pool(F.relu(self.conv(x)))

        x = torch.flatten(x)  # flatten all dimensions except batch
        x = self.lin(x)
        x = torch.reshape(x, shape=(self.M, self.out))
        return x


class double_convolution(torch.nn.Module):
    """
        Uporaba dveh konvolucij + ene linearne plasti na koncu

          Parametri
        ----------
        c1 = kernel size
        l1 = št out channels
        c2 = kernel size
        l3 = št out channels
        p = (a,b) = pooling size..
        out = velikost izhoda

    Input v mrežo je tabela oblike N x n_občin
    """

    def __init__(self,M, d_settings, out=12, N=14, n_obcin=193) -> None:
        super().__init__()
        # c1, l1, c2, l2, p1, p2 = d_settings
        c1, l1, p1, c2, l2, p2 = d_settings
        self.conv1 = Conv2d(1, l1, c1)
        self.pool1 = MaxPool2d((p1[0], p1[1]))

        h = (N - c1 + 1) / p1[0]
        w = (n_obcin - c1 + 1) / p1[1]

        self.conv2 = Conv2d(l1, l2, c2)

        h = h - c2 + 1
        w = w - c2 + 1

        self.pool2 = MaxPool2d((p2[0], p2[1]))

        h /= p2[0]
        w /= p2[1]
        h = int(h)
        w = int(w)
        self.lin = Linear(h * w  * l2, out*M)
        self.M = M
        self.out = out

    def forward(self, x):
        x = torch.reshape(x, (1,) + tuple(x.shape))
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = torch.flatten(x)  
        
        x = self.lin(x)
        x = torch.reshape(x, shape=(self.M, self.out))
        return x

class nn_test(torch.nn.Module):
    def __init__(self, a, b, c=12):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(a, b)
        self.lin2 = Linear(b, b)
        self.lin3 = Linear(b, c)

    def forward(self, x):
        x = torch.flatten(x)
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return x

def fit(config, params, X, y, n_epochs=10, batch_size=10):
    '''
    Fits model, described in config to data X,y
    '''
    model = config['algo']['model'](**params)
    loss_function = torch.nn.MSELoss()  
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) 
    def train(batch):   
        X_, y_ = batch
        opt.zero_grad()  
        
        # go throue the whole dataset
        for index, ex in enumerate(X_):
            tmp_out = model(ex)
            tmp_out = torch.reshape(tmp_out, (1,)+tuple(tmp_out.shape))
            if index == 0:
                out = tmp_out
            else:
                out = torch.cat((out, tmp_out))




        napaka = loss_function(out, y_) 
        napaka.backward()  
        opt.step() 
        return napaka
    model.train()
    for epoch in range(1, n_epochs):
        for batch in range(len(X) // batch_size):
            loss = train((X[batch: batch + batch_size], y[batch: batch + batch_size]))  
       # if epoch%10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    return model

def fit_model(model, X, y, n_epochs=10, batch_size=10):
    '''
    Fits model, described in config to data X,y
    '''
    loss_function = torch.nn.MSELoss()  
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) 
    def train(batch):   
        X_, y_ = batch
        opt.zero_grad()  
        
        # go throue the whole dataset
        for index, ex in enumerate(X_):
            tmp_out = model(ex)
            tmp_out = torch.reshape(tmp_out, (1,)+tuple(tmp_out.shape))
            if index == 0:
                out = tmp_out
            else:
                out = torch.cat((out, tmp_out))




        napaka = loss_function(out, y_) 
        napaka.backward()  
        opt.step() 
        return napaka
    model.train()
    for epoch in range(1, n_epochs):
        for batch in range(len(X) // batch_size):
            loss = train((X[batch: batch + batch_size], y[batch: batch + batch_size]))  
       # if epoch%10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    return model