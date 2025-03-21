import os
import pickle
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from torch_geometric.data import Data, DataLoader as PyGDataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, BatchNorm
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import from_networkx

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold

import optuna  # <-- For hyperparameter optimization

# Dash imports for showing results in a dashboard
import dash
from dash import dcc
from dash import html
from dash import dash_table
import plotly.graph_objs as go

warnings.filterwarnings('ignore')

# =============================================================================
#                             DATASET & PREP
# =============================================================================

class MolDataset(Dataset):
    """
    A custom dataset that:
      - Reads external factors from CSV
      - Loads the corresponding pickle for the molecule's graph
      - Converts it into a PyG Data object
    """
    def __init__(
        self,
        raw_dataframe: pd.DataFrame,
        nx_graph_dict: dict,
        *,
        component_col: str,
        global_state_cols: list[str],
        label_col: str,
        transform=None
    ):
        """
        Args:
            raw_dataframe: The input dataframe containing molecule info.
            nx_graph_dict: Dictionary mapping component names to networkx graphs.
            component_col: Column name for the component.
            global_state_cols: List of columns representing external factors.
            label_col: Column name for the regression target.
            transform: Any transform to apply to each PyG Data object.
        """
        self.raw_dataframe = raw_dataframe
        self.nx_graph_dict = nx_graph_dict
        self.component_col = [component_col] if isinstance(component_col, str) else component_col
        self.global_state_cols = global_state_cols
        self.label_col = [label_col] if isinstance(label_col, str) else label_col
        self.transform = transform

        required_cols = set(self.global_state_cols + self.label_col + self.component_col)
        for col in required_cols:
            if col not in self.raw_dataframe.columns:
                raise ValueError(f"Missing column in DataFrame: '{col}'")

    def __len__(self):
        return len(self.raw_dataframe)

    def __getitem__(self, idx):
        row = self.raw_dataframe.iloc[idx]

        # 1. Load the molecule graph
        component_name = row[self.component_col[0]]  # e.g. "C23"
        pyg_data = self.nx_graph_dict[component_name]

        # 2. Prepare the external factors
        externals = torch.tensor(row[self.global_state_cols].values.astype(float), dtype=torch.float)
        externals = externals.unsqueeze(0)

        # 3. Prepare the label (regression target)
        label = torch.tensor([row[self.label_col][0]], dtype=torch.float)

        # 4. Attach externals & label to the Data object
        pyg_data.externals = externals  # shape [1, external_in_dim]
        pyg_data.y = label  # shape [1]

        if self.transform:
            pyg_data = self.transform(pyg_data)

        return pyg_data


def networkx_to_pyg(nx_graph):
    """
    Convert a networkx graph to a torch_geometric.data.Data object.
    This is a basic template; adjust for your actual node/edge features.
    """
    # Sort nodes to ensure consistent ordering
    node_mapping = {node: i for i, node in enumerate(nx_graph.nodes())}

    x_list = []
    edge_index_list = []
    edge_attr_list = []

    # Node features
    for node in nx_graph.nodes(data=True):
        original_id = node[0]
        attrs = node[1]
        symbol = attrs.get("symbol", "C")
        symbol_id = 0 if symbol == "C" else 1 if symbol == "H" else 2
        x_list.append([symbol_id])

    # Edge features
    for u, v, edge_attrs in nx_graph.edges(data=True):
        u_idx = node_mapping[u]
        v_idx = node_mapping[v]
        edge_index_list.append((u_idx, v_idx))
        bde_pred = edge_attrs.get("bde_pred", 0.0) or 0.0
        bdfe_pred = edge_attrs.get("bdfe_pred", 0.0) or 0.0
        edge_attr_list.append([bde_pred, bdfe_pred])

    x = torch.tensor(x_list, dtype=torch.float)
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


# =============================================================================
#                     BASE GNN MODEL (WITH DIM MATCH)
# =============================================================================

class GINE_Regression(nn.Module):
    """
    A GNN for regression using GINEConv layers + edge attributes,
    where all layers have the same hidden_dim (no dimension mismatch).
    """
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        external_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        # Encode edges from edge_in_dim to hidden_dim
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Encode nodes from node_in_dim to hidden_dim
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)

        # Multiple GINEConv layers & corresponding BatchNorm
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            conv = GINEConv(nn=net)
            self.convs.append(conv)
            self.bns.append(BatchNorm(hidden_dim))

        self.dropout = nn.Dropout(p=dropout)

        # Process external factors
        self.externals_mlp = nn.Sequential(
            nn.Linear(external_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Final regression
        self.final_regressor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Encode
        x = self.node_encoder(x)
        edge_emb = self.edge_encoder(edge_attr)

        # Pass through GINEConv layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_emb)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Global pooling
        graph_emb = global_mean_pool(x, batch)

        # Process external factors
        ext_emb = self.externals_mlp(data.externals)

        # Combine & regress
        combined = torch.cat([graph_emb, ext_emb], dim=-1)
        out = self.final_regressor(combined).squeeze(-1)
        return out


# =============================================================================
#                   SOFT CONSTRAINTS (HELPER FUNCTIONS)
# =============================================================================

def compute_c_neighbors(batch_data):
    """
    Compute the total (or average) number of carbon neighbors across all nodes in the batch.
    We'll do a simple approach:
      - For each node, check how many edges connect to a node with symbol_id == 0 (C).
      - Then average or sum across the graph.
    This is just an example aggregator.
    """
    edge_index = batch_data.edge_index
    symbol_ids = batch_data.x[:, 0]

    c_neighbor_count = torch.zeros_like(symbol_ids)
    num_edges = edge_index.size(1)

    for e in range(num_edges):
        src = edge_index[0, e]
        tgt = edge_index[1, e]
        if symbol_ids[tgt] == 0:  # 'C'
            c_neighbor_count[src] += 1
        if symbol_ids[src] == 0:
            c_neighbor_count[tgt] += 1

    avg_c_neighbors = c_neighbor_count.mean()
    return avg_c_neighbors


def compute_bde_bdfe(batch_data):
    """
    Compute an aggregated BDE/BDFe value for the entire graph.
    We'll just take the mean of bde_pred and bdfe_pred across edges as an example.
    """
    edge_attrs = batch_data.edge_attr
    if edge_attrs.size(0) == 0:
        return 0.0, 0.0
    bde_mean = edge_attrs[:, 0].mean()
    bdfe_mean = edge_attrs[:, 1].mean()
    return bde_mean, bdfe_mean


def compute_atom_energy(batch_data):
    """
    If you store 'energy' in node features or in a separate attribute,
    here's where you'd aggregate it.
    We'll do a dummy approach: sum the symbol IDs as a stand-in "energy".
    """
    energy = batch_data.x[:, 0].sum()
    return energy


def compute_soft_constraints(batch_data, preds):
    """
    Returns the three soft constraint losses:
      - BDE_BDFe loss
      - Energy loss
      - Num C neighbors loss
    Each is defined as mean(pred * aggregated_value).
    """
    # --- COMMENT OUT or set to zero the BDE/BDFE and C neighbors;
    #     keep only "atom energy" constraint active ---
    """
    bde_mean, bdfe_mean = compute_bde_bdfe(batch_data)
    aggregated_bde = bde_mean + bdfe_mean
    loss_bde = (preds * aggregated_bde).mean()

    avg_c_neighbors = compute_c_neighbors(batch_data)
    loss_c = (preds * avg_c_neighbors).mean()
    """

    # The only active constraint: "atom energy"
    aggregated_energy = compute_atom_energy(batch_data)
    loss_energy = (preds * aggregated_energy).mean()

    # Set the others to zero so they won't affect final loss
    loss_bde = torch.tensor(0.0, device=preds.device)
    loss_c = torch.tensor(0.0, device=preds.device)

    return loss_bde, loss_energy, loss_c


# =============================================================================
#                   TRAIN/VALID/EVALUATION UTILS (WITH SOFT CONSTRAINTS)
# =============================================================================

def train_one_epoch_with_constraints(
    model, loader, optimizer, device,
    base_criterion, lambda_bde=0.0, lambda_energy=0.001, lambda_c=0.0
):
    """
    Train for one epoch, combining MSE + the "atom energy" constraint.
    Other constraints are effectively disabled (set to 0).
    """
    model.train()
    total_loss = 0.0
    count = 0

    for batch_data in loader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        preds = model(batch_data)
        y = batch_data.y.to(device).view(-1)

        # Base regression loss
        base_loss = base_criterion(preds, y)

        # Soft constraints
        loss_bde, loss_energy, loss_c = compute_soft_constraints(batch_data, preds)

        # Only the "energy" term is active
        total_soft_loss = (lambda_bde * loss_bde
                           + lambda_energy * loss_energy
                           + lambda_c * loss_c)

        # Final combined loss
        loss = base_loss + total_soft_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_data.num_graphs
        count += batch_data.num_graphs

    return total_loss / count if count > 0 else 0.0


def validate_with_constraints(
    model, loader, device, base_criterion,
    lambda_bde=0.0, lambda_energy=0.001, lambda_c=0.0
):
    """
    Validate for one epoch, combining MSE + the "atom energy" constraint.
    Other constraints are effectively disabled.
    """
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)
            preds = model(batch_data)
            y = batch_data.y.to(device).view(-1)

            # Base regression loss
            base_loss = base_criterion(preds, y)

            # Soft constraints
            loss_bde, loss_energy, loss_c = compute_soft_constraints(batch_data, preds)

            # Only the "energy" term is active
            total_soft_loss = (lambda_bde * loss_bde
                               + lambda_energy * loss_energy
                               + lambda_c * loss_c)

            loss = base_loss + total_soft_loss

            total_loss += loss.item() * batch_data.num_graphs
            count += batch_data.num_graphs

    return total_loss / count if count > 0 else 0.0


def evaluate_model(model, loader, device):
    """
    Evaluate the model on a dataset loader and compute R² and RMSE.
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch)
            y_true.append(batch.y.cpu())
            y_pred.append(preds.cpu())

    y_true = torch.cat(y_true).numpy().squeeze()
    y_pred = torch.cat(y_pred).numpy().squeeze()

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return r2, rmse


# =============================================================================
#                             DATA PREPARATION
# =============================================================================

env_file = "/home/tingyi/GNN_chemicalENV-main/GNN molecules/graph_pickles/dataset02.xlsx"
data = pd.read_excel(env_file, engine='openpyxl').dropna(subset=['degradation_rate'])
data['seawater'] = data['seawater'].map({'art': 1, 'sea': 0})

folder_path = "/home/tingyi/GNN_chemicalENV-main/GNN molecules/graph_pickles/molecules"
graph_pickles = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]

base_dir = "/home/tingyi/GNN_chemicalENV-main/GNN molecules/graph_pickles/molecules"
if os.path.exists(base_dir):
    print("Directory exists:", base_dir)
    print("Files in directory:", os.listdir(base_dir))
else:
    print(f"Error: Directory {base_dir} does not exist!")

compounds = data.component.unique()
graphs_dict = {}
for compound, graph_pickle in zip(compounds, graph_pickles):
    with open(os.path.join(base_dir, graph_pickle), 'rb') as f:
        graph = pickle.load(f)
        graphs_dict[compound] = networkx_to_pyg(graph)

dataset = MolDataset(
    raw_dataframe=data,
    nx_graph_dict=graphs_dict,
    component_col="component",
    global_state_cols=["temperature", "concentration", "time", "seawater"],
    label_col="degradation_rate",
    transform=None
)

# =============================================================================
#                      OPTUNA HYPERPARAM OPT + CROSS-VALIDATION
# =============================================================================

from torch_geometric.data import DataLoader as PyGDataLoader

def objective(trial):
    """
    Optuna objective function:
      - Sample hyperparams
      - Do a 5-fold cross validation with early stopping
      - Return avg validation loss
    """
    # Hyperparameters to tune:
    hidden_dim = trial.suggest_int("hidden_dim", 8, 64, step=8)  # Example range
    lr = trial.suggest_categorical("learning_rate", [1e-3, 3e-3, 1e-4, 3e-4])
    dropout = trial.suggest_categorical("dropout", [0.1, 0.5])
    num_layers = trial.suggest_int("num_layers", 2, 6)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # Only "energy" is relevant, so set others to 0.0
    lambda_bde = 0.0
    lambda_c = 0.0

    # We'll tune only lambda_energy
    lambda_energy = trial.suggest_float("lambda_energy", 1e-4, 1e-1, log=True)

    # Early stopping parameters for objective function
    max_epochs = 100
    patience = 10
    min_delta = 1e-5

    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = PyGDataLoader(train_subset, batch_size=16, shuffle=True)
        val_loader = PyGDataLoader(val_subset, batch_size=16, shuffle=False)

        model = GINE_Regression(
            node_in_dim=1,
            edge_in_dim=2,
            external_in_dim=4,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        base_criterion = torch.nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0

        # Early stopping: train for max_epochs or until patience is exceeded
        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch_with_constraints(
                model,
                train_loader,
                optimizer,
                device,
                base_criterion,
                lambda_bde=lambda_bde,
                lambda_energy=lambda_energy,
                lambda_c=lambda_c
            )
            val_loss = validate_with_constraints(
                model,
                val_loader,
                device,
                base_criterion,
                lambda_bde=lambda_bde,
                lambda_energy=lambda_energy,
                lambda_c=lambda_c
            )
            trial.report(val_loss, step=epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_loss < (best_val_loss - min_delta):
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        fold_val_losses.append(best_val_loss)

    avg_val_loss = float(np.mean(fold_val_losses))
    return avg_val_loss

# Create and run the Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print("\n=== Optuna Study Results ===")
print(f"Best trial number: {study.best_trial.number}")
print(f"Best trial value (loss): {study.best_trial.value:.4f}")
print("Best hyperparameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# =============================================================================
#          RETRAIN FINAL MODEL WITH BEST HYPERPARAMS + REPORT TEST METRICS
# =============================================================================

best_params = study.best_params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_dim = best_params["hidden_dim"]
num_layers = best_params["num_layers"]
dropout = best_params["dropout"]
lr = best_params["lr"]
weight_decay = best_params["weight_decay"]
lambda_energy = best_params["lambda_energy"]

# Force bde and c to zero in final training
lambda_bde = 0.0
lambda_c = 0.0

# Early stopping parameters for final retraining
max_epochs = 300
patience = 10
min_delta = 1e-5

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []
early_stopping_data = []  # To store early stopping curves and stop epoch info

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n--- Retrain Fold {fold + 1} with Best Hyperparams ---")

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = PyGDataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = PyGDataLoader(val_subset, batch_size=16, shuffle=False)

    model = GINE_Regression(
        node_in_dim=1,
        edge_in_dim=2,
        external_in_dim=4,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    base_criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    train_losses = []
    val_losses = []
    epochs_ran = []

    stop_epoch = max_epochs  # default if never early stopped

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch_with_constraints(
            model,
            train_loader,
            optimizer,
            device,
            base_criterion,
            lambda_bde=lambda_bde,
            lambda_energy=lambda_energy,
            lambda_c=lambda_c
        )
        val_loss = validate_with_constraints(
            model,
            val_loader,
            device,
            base_criterion,
            lambda_bde=lambda_bde,
            lambda_energy=lambda_energy,
            lambda_c=lambda_c
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epochs_ran.append(epoch)

        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                stop_epoch = epoch
                print(f"Early stopping triggered at epoch {epoch}")
                break

        if epoch % 50 == 0:
            print(f"[Fold {fold + 1} Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    early_stopping_data.append({
        "fold": fold + 1,
        "epochs": epochs_ran,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "stop_epoch": stop_epoch
    })

    print(f"Evaluating fold {fold + 1} ...")
    r2, rmse = evaluate_model(model, val_loader, device)
    fold_results.append({"fold": fold + 1, "r2": r2, "rmse": rmse})

r2_scores = [res["r2"] for res in fold_results]
rmse_scores = [res["rmse"] for res in fold_results]

print("\n--- Final Cross-Validation Summary with Best Hyperparams ---")
for res in fold_results:
    print(f"Fold {res['fold']}: R² = {res['r2']:.4f}, RMSE = {res['rmse']:.4f}")

print(f"\nAverage R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")

# =============================================================================
#                           DASHBOARD FOR RESULTS
# =============================================================================

# Convert fold_results to a DataFrame for easy display
fold_results_df = pd.DataFrame(fold_results)

# Build the early stopping curves figure using Plotly
early_stopping_fig = go.Figure()
for fold_data in early_stopping_data:
    early_stopping_fig.add_trace(go.Scatter(
         x=fold_data["epochs"],
         y=fold_data["val_losses"],
         mode='lines+markers',
         name=f"Fold {fold_data['fold']} (Stop at epoch {fold_data['stop_epoch']})"
    ))
early_stopping_fig.update_layout(
    title="Early Stopping Validation Loss Curves",
    xaxis_title="Epoch",
    yaxis_title="Validation Loss"
)

# Build the Dash dashboard layout
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Optuna + GNN Training Results Dashboard"),
    html.H2("Best Hyperparameters Found by Optuna"),
    html.Ul([
        html.Li(f"{key}: {value}") for key, value in best_params.items()
    ]),
    html.P(f"Best trial value (loss): {study.best_trial.value:.4f}"),
    html.Br(),
    html.H2("Cross-Validation Results with Best Hyperparams"),
    dash_table.DataTable(
        id='fold-results-table',
        columns=[{"name": i, "id": i} for i in fold_results_df.columns],
        data=fold_results_df.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'}
    ),
    html.Br(),
    html.Div([
        html.H3("Averaged Metrics Across Folds:"),
        html.P(f"Average R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}"),
        html.P(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    ]),
    html.Br(),
    html.H2("Early Stopping Curves per Fold"),
    dcc.Graph(
        id='early-stopping-graph',
        figure=early_stopping_fig
    ),
    html.Div([
        html.H3("Early Stopping Summary:"),
        html.Ul([
            html.Li(f"Fold {fold_data['fold']} stopped at epoch {fold_data['stop_epoch']}")
            for fold_data in early_stopping_data
        ])
    ])
])

if __name__ == '__main__':
    # Run the Dash app
    # Note: The rest of the model training has already been performed above.
    app.run_server(debug=False)
