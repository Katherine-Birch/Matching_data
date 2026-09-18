import torch

def sinkhorn():
    for _ in range(num_iters):
        matrix = matrix / matrix.sum(dim=-1, keepdim= True)
        matrix = matrix / matrix.sum(dim=-2, keepdim= True)
    return matrix
                        
def MLPSinkhorn(torch.nn.Module):
    def __init__(self, num_nodes=90, hidden_dim1=512, hidden_dim2=256, tau=0.1, n_iters=20):
        super().__init__()
        self.num_nodes = num_nodes
        self.tau = tau
        self.n_iters = n_iters
        graph_dim = num_nodes * num_nodes
        self.fc1 = torch.nn.Linear(graph_dim, hidden_dim1)
        self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.final = torch.nn.Linear(hidden_dim2, num_nodes * num_nodes)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        b = x.shape[0]
        x_flat = x.flatten(1, -1)

        h = self.relu(self.fc1(x_flat))
        h = self.relu(self.fc2(h))
        logits = self.final(h).view(b, self.num_nodes, self.num_nodes)

        kernel = torch.exp(logits / self.tau)
        soft_perm = sinkhorn(kernel, self.n_iters)
        return soft_perm

# def Graph_matching_model():

 