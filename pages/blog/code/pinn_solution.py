# Physics-Informed Neural Network (PINN) Solution
import torch
import torch.nn as nn
import numpy as np

# Training data (collocation points for t)
t_train = torch.linspace(0, 5, 100).view(-1, 1).requires_grad_(True)

# Initial condition
y0 = torch.tensor([[-1.0]])

# Define the PINN model
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t):
        return self.net(t)

model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()

    # Network prediction
    y_hat = model(t_train)

    # Compute derivative dy/dt using autograd
    dy_dt = torch.autograd.grad(
        y_hat, t_train,
        grad_outputs=torch.ones_like(y_hat),
        create_graph=True
    )[0]

    # Physics loss (ODE residual)
    residual = dy_dt - (2*t_train - y_hat)
    loss_phys = torch.mean(residual**2)

    # Initial condition loss
    loss_ic = (model(torch.tensor([[0.0]])) - y0).pow(2).mean()

    # Total loss
    loss = loss_phys + loss_ic
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.6f}')

# Evaluation
t_np = t_train.detach().numpy().flatten()
with torch.no_grad():
    y_pinn = model(t_train).numpy().flatten()

print("PINN training completed!")
print(f"Final solution at t=5: y_pinn = {y_pinn[-1]:.4f}")