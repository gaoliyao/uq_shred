import torch
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy

class SHRED(torch.nn.Module):
    '''SHRED model accepts input size (number of sensors), output size (dimension of high-dimensional spatio-temporal state, hidden_size, number of LSTM layers,
    size of fully-connected layers, and dropout parameter'''
    def __init__(self, input_size, output_size, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.0):
        super(SHRED,self).__init__()

        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                 num_layers=hidden_layers, batch_first=True)
        
        self.linear1 = torch.nn.Linear(hidden_size, l1)
        self.linear2 = torch.nn.Linear(l1, l2)
        self.linear3 = torch.nn.Linear(l2, output_size)

        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

        

    def forward(self, x):
        
        h_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float)
        c_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float)
        if next(self.parameters()).is_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out[-1].view(-1, self.hidden_size)

        output = self.linear1(h_out)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear2(output)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)
    
        output = self.linear3(output)

        return output


class SDN(torch.nn.Module):
    '''SDN model accepts input size (number of sensors), output size (dimension of high-dimensional spatio-temporal state,
    size of fully-connected layers, and dropout parameter'''
    def __init__(self, input_size, output_size, l1=350, l2=400, dropout=0.0):
        super(SDN,self).__init__()
        
        self.linear1 = torch.nn.Linear(input_size, l1)
        self.linear2 = torch.nn.Linear(l1, l2)
        self.linear3 = torch.nn.Linear(l2, output_size)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):

        output = self.linear1(x)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear2(output)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)
        
        output = self.linear3(output)

        return output

def fit(model, train_dataset, valid_dataset, batch_size=64, num_epochs=4000, lr=1e-3, verbose=False, patience=5):
    '''Function for training SHRED and SDN models'''
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    val_error_list = []
    patience_counter = 0
    best_params = model.state_dict()
    for epoch in range(1, num_epochs + 1):
        
        for k, data in enumerate(train_loader):
            model.train()
            outputs = model(data[0])
            optimizer.zero_grad()
            loss = criterion(outputs, data[1])
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_outputs = model(valid_dataset.X)
                val_error = torch.linalg.norm(val_outputs - valid_dataset.Y)
                val_error = val_error / torch.linalg.norm(valid_dataset.Y)
                val_error_list.append(val_error)

            if verbose == True:
                print('Training epoch ' + str(epoch))
                print('Error ' + str(val_error_list[-1]))

            if val_error == torch.min(torch.tensor(val_error_list)):
                patience_counter = 0
                best_params = deepcopy(model.state_dict())
            else:
                patience_counter += 1


            if patience_counter == patience:
                model.load_state_dict(best_params)
                return torch.tensor(val_error_list).cpu()

    model.load_state_dict(best_params)
    return torch.tensor(val_error_list).detach().cpu().numpy()


def forecast(forecaster, reconstructor, test_dataset):
    '''Takes model and corresponding test dataset, returns tensor containing the
    inputs to generate the first forecast and then all subsequent forecasts 
    throughout the test dataset.'''
    initial_in = test_dataset.X[0:1].clone()
    vals = []
    for i in range(0, test_dataset.X.shape[1]):
        vals.append(initial_in[0, i, :].detach().cpu().clone().numpy())

    for i in range(len(test_dataset.X)):
        scaled_output = forecaster(initial_in).detach().cpu().numpy()

        vals.append(scaled_output.reshape(test_dataset.X.shape[2]))
        temp = initial_in.clone()
        initial_in[0,:-1] = temp[0,1:]
        initial_in[0,-1] = torch.tensor(scaled_output)

    device = 'cuda' if next(reconstructor.parameters()).is_cuda else 'cpu'
    forecasted_vals = torch.tensor(np.array(vals), dtype=torch.float32).to(device)
    reconstructions = []
    for i in range(len(forecasted_vals) - test_dataset.X.shape[1]):
        recon = reconstructor(forecasted_vals[i:i+test_dataset.X.shape[1]].reshape(1, test_dataset.X.shape[1], 
                                    test_dataset.X.shape[2])).detach().cpu().numpy()
        reconstructions.append(recon)
    reconstructions = np.array(reconstructions)
    return forecasted_vals, reconstructions


class UQ_SHRED(torch.nn.Module):
    '''UQ-SHRED: SHRED with uncertainty quantification via engression.
    Same interface as SHRED, with added noise_dim parameter.'''
    
    def __init__(self, input_size, output_size, hidden_size=64, hidden_layers=2,
                 l1=350, l2=400, dropout=0.0, noise_dim=50):
        super(UQ_SHRED, self).__init__()
        
        self.noise_dim = noise_dim
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        
        # LSTM takes sensors + noise
        self.lstm = torch.nn.LSTM(input_size=input_size + noise_dim, hidden_size=hidden_size,
                                  num_layers=hidden_layers, batch_first=True)
        
        self.linear1 = torch.nn.Linear(hidden_size, l1)
        self.linear2 = torch.nn.Linear(l1, l2)
        self.linear3 = torch.nn.Linear(l2, output_size)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x, noise=None):
        batch_size, lags, _ = x.shape
        device = x.device
        
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=device)
        
        # Expand noise across lags: (batch, noise_dim) -> (batch, lags, noise_dim)
        noise_expanded = noise.unsqueeze(1).expand(-1, lags, -1)
        x_noisy = torch.cat([x, noise_expanded], dim=-1)
        
        # LSTM
        h_0 = torch.zeros(self.hidden_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.hidden_layers, batch_size, self.hidden_size, device=device)
        _, (h_out, _) = self.lstm(x_noisy, (h_0, c_0))
        h_out = h_out[-1].view(-1, self.hidden_size)
        
        # Decoder
        output = torch.nn.functional.relu(self.dropout(self.linear1(h_out)))
        output = torch.nn.functional.relu(self.dropout(self.linear2(output)))
        output = self.linear3(output)
        
        return output
    
    def sample(self, x, n_samples=100):
        '''Generate n_samples reconstructions for UQ.'''
        self.eval()
        with torch.no_grad():
            samples = torch.stack([self.forward(x) for _ in range(n_samples)])
        return samples  # (n_samples, batch, output_dim)
    
    def reconstruct(self, x, n_samples=100):
        '''Reconstruct with uncertainty (mean and std).'''
        samples = self.sample(x, n_samples)
        return samples.mean(dim=0), samples.std(dim=0)
    
    def reconstruct_quantiles(self, x, quantiles=[0.025, 0.5, 0.975], n_samples=100):
        '''Reconstruct with quantiles for confidence intervals.'''
        samples = self.sample(x, n_samples)
        return {q: torch.quantile(samples, q, dim=0) for q in quantiles}


class UQ_Forecaster(torch.nn.Module):
    '''UQ-Forecaster: predicts next sensor values with uncertainty.
    Same interface as a standard forecaster, with added noise_dim.'''
    
    def __init__(self, input_size, hidden_size=64, hidden_layers=2, noise_dim=50):
        super(UQ_Forecaster, self).__init__()
        
        self.noise_dim = noise_dim
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # LSTM takes sensors + noise
        self.lstm = torch.nn.LSTM(input_size=input_size + noise_dim, hidden_size=hidden_size,
                                  num_layers=hidden_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, input_size)  # predict next sensors
    
    def forward(self, x, noise=None):
        batch_size, lags, _ = x.shape
        device = x.device
        
        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=device)
        
        noise_expanded = noise.unsqueeze(1).expand(-1, lags, -1)
        x_noisy = torch.cat([x, noise_expanded], dim=-1)
        
        h_0 = torch.zeros(self.hidden_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.hidden_layers, batch_size, self.hidden_size, device=device)
        _, (h_out, _) = self.lstm(x_noisy, (h_0, c_0))
        h_out = h_out[-1].view(-1, self.hidden_size)
        
        return self.linear(h_out)
    
    def sample(self, x, n_samples=100):
        '''Generate n_samples predictions.'''
        self.eval()
        with torch.no_grad():
            samples = torch.stack([self.forward(x) for _ in range(n_samples)])
        return samples

    def predict(self, x, n_samples=100):
        '''Predict next sensor values with uncertainty (mean and std).'''
        samples = self.sample(x, n_samples)
        return samples.mean(dim=0), samples.std(dim=0)

    def sample_trajectory(self, x, horizon, n_samples=100):
        '''Generate n_samples trajectories over horizon steps.
        
        Args:
            x: (batch, lags, num_sensors) initial history
            horizon: number of future steps
            n_samples: number of trajectory samples
        
        Returns:
            trajectories: (n_samples, batch, horizon, num_sensors)
        '''
        self.eval()
        trajectories = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                history = x.clone()
                traj = []
                for t in range(horizon):
                    next_sensor = self.forward(history)
                    traj.append(next_sensor)
                    # Roll history: drop oldest, append newest
                    history = torch.cat([history[:, 1:, :], next_sensor.unsqueeze(1)], dim=1)
                trajectories.append(torch.stack(traj, dim=1))
        
        return torch.stack(trajectories)  # (n_samples, batch, horizon, num_sensors)
    
    def forecast(self, x, horizon, n_samples=100):
        '''Forecast with uncertainty (mean and std).'''
        traj = self.sample_trajectory(x, horizon, n_samples)
        return traj.mean(dim=0), traj.std(dim=0)
    
    def forecast_quantiles(self, x, horizon, quantiles=[0.025, 0.5, 0.975], n_samples=100):
        '''Forecast with quantiles.'''
        traj = self.sample_trajectory(x, horizon, n_samples)
        return {q: torch.quantile(traj, q, dim=0) for q in quantiles}


def forecast_uq(forecaster, reconstructor, test_dataset, horizon=None, n_samples=50):
    '''Forecast with UQ using UQ_Forecaster and UQ_SHRED.
    
    Args:
        forecaster: UQ_Forecaster model
        reconstructor: UQ_SHRED model
        test_dataset: test data
        horizon: forecast horizon (default: length of test set)
        n_samples: number of samples for UQ
    
    Returns:
        sensor_traj: (n_samples, horizon, num_sensors) forecasted sensors
        recon_mean: (horizon, output_dim) mean reconstruction
        recon_std: (horizon, output_dim) std reconstruction
    '''
    if horizon is None:
        horizon = len(test_dataset.X)
    
    initial = test_dataset.X[0:1]  # (1, lags, sensors)
    lags = initial.shape[1]
    
    # Sample sensor trajectories
    sensor_traj = forecaster.sample_trajectory(initial, horizon, n_samples)
    # (n_samples, 1, horizon, sensors) -> (n_samples, horizon, sensors)
    sensor_traj = sensor_traj.squeeze(1)
    
    # Reconstruct each trajectory
    recons = []
    reconstructor.eval()
    with torch.no_grad():
        for s in range(n_samples):
            recon_s = []
            history = initial.clone()
            for t in range(horizon):
                # Reconstruct current state
                out = reconstructor(history)
                recon_s.append(out)
                # Roll with this sample's forecasted sensor
                next_sensor = sensor_traj[s, t:t+1, :]
                history = torch.cat([history[:, 1:, :], next_sensor.unsqueeze(0)], dim=1)
            recons.append(torch.cat(recon_s, dim=0))
    
    recons = torch.stack(recons)  # (n_samples, horizon, output_dim)
    recon_mean = recons.mean(dim=0)
    recon_std = recons.std(dim=0)
    
    return sensor_traj, recon_mean, recon_std


def fit_uq(model, train_dataset, valid_dataset, batch_size=64, num_epochs=4000,
           lr=1e-3, verbose=False, patience=5, beta=1):
    '''Training function for UQ_SHRED or UQ_Forecaster with energy loss.'''
    from uq import energy_loss
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    val_error_list = []
    patience_counter = 0
    best_params = model.state_dict()
    
    for epoch in range(1, num_epochs + 1):
        for data in train_loader:
            model.train()
            x, y = data[0], data[1]
            
            # Two forward passes with different noise
            y_pred1 = model(x)
            y_pred2 = model(x)
            
            optimizer.zero_grad()
            loss = energy_loss(y, y_pred1, y_pred2, beta=beta)
            loss.backward()
            optimizer.step()
        
        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                if isinstance(model, UQ_Forecaster):
                    val_mean, _ = model.predict(valid_dataset.X, n_samples=20)
                else:
                    val_mean, _ = model.reconstruct(valid_dataset.X, n_samples=20)
                val_error = torch.linalg.norm(val_mean - valid_dataset.Y)
                val_error = val_error / torch.linalg.norm(valid_dataset.Y)
                val_error_list.append(val_error)
            
            if verbose:
                print(f'Epoch {epoch}: error={val_error:.4f}')
            
            if val_error == torch.min(torch.tensor(val_error_list)):
                patience_counter = 0
                best_params = deepcopy(model.state_dict())
            else:
                patience_counter += 1
            
            if patience_counter == patience:
                model.load_state_dict(best_params)
                return torch.tensor(val_error_list).cpu()
    
    model.load_state_dict(best_params)
    return torch.tensor(val_error_list).detach().cpu().numpy()