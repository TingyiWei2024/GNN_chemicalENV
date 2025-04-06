import os
import pickle
import pandas as pd
import numpy as np

import networkx as nx

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data, DataLoader as PyGDataLoader
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GINEConv, global_mean_pool, BatchNorm
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold

import warnings
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
    def __init__(self,
                 raw_dataframe: pd.DataFrame,
                 nx_graph_dict: dict,
                 *,
                 component_col: str,
                 global_state_cols: list[str],
                 label_col: str,
                 transform=None):
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
        self.component_col = [component_col] if type(component_col) is str else component_col
        self.global_state_cols = global_state_cols
        self.label_col = [label_col] if type(label_col) is str else label_col
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
    def __init__(self,
                 node_in_dim: int,
                 edge_in_dim: int,
                 external_in_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.1):
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
#                   TRAIN/VALID/EVALUATION UTILS
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    count = 0
    for batch_data in loader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        preds = model(batch_data)
        y = batch_data.y.to(device).view(-1)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_data.num_graphs
        count += batch_data.num_graphs
    return total_loss / count if count > 0 else 0.0


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)
            preds = model(batch_data)
            y = batch_data.y.to(device).view(-1)
            loss = criterion(preds, y)
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
#                      CROSS-VALIDATION (FIXED MODEL)
# =============================================================================
from torch_geometric.data import DataLoader as PyGDataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
k = 5  # number of folds
from sklearn.model_selection import KFold
kf = KFold(n_splits=k, shuffle=True, random_state=42)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n--- Fold {fold + 1} ---")

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = PyGDataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = PyGDataLoader(val_subset, batch_size=64, shuffle=False)

    model = GINE_Regression(
        node_in_dim=1,
        edge_in_dim=2,
        external_in_dim=4,
        hidden_dim=16,  # Example hidden_dim
        num_layers=5,
        dropout=0.1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()

    num_epochs = 1000
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        if epoch % 10 == 0:
            print(f"[Fold {fold + 1} Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    print(f"Evaluating fold {fold + 1} ...")
    r2, rmse = evaluate_model(model, val_loader, device)
    fold_results.append({"fold": fold + 1, "r2": r2, "rmse": rmse})

r2_scores = [res["r2"] for res in fold_results]
rmse_scores = [res["rmse"] for res in fold_results]

print("\n--- Cross-Validation Summary ---")
for res in fold_results:
    print(f"Fold {res['fold']}: R² = {res['r2']:.4f}, RMSE = {res['rmse']:.4f}")

print(f"\nAverage R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")


# =============================================================================
#             IMPROVED MODEL WITH TRAPEZOID DIMENSIONS & PROJECTIONS
# =============================================================================
import optuna
import optuna.visualization as vis

class GINE_RegressionTrapezoid(nn.Module):
    """
    A GINEConv-based regression model that uses a list of hidden dimensions
    to build layers with decreasing size (trapezoid architecture),
    ensuring dimension consistency with projection layers between convs.
    """
    def __init__(self,
                 node_in_dim: int,
                 edge_in_dim: int,
                 external_in_dim: int,
                 hidden_dims: list,
                 dropout: float = 0.1):
        super().__init__()

        # For the first layer, encode edges to hidden_dims[0], and encode nodes as well
        self.initial_edge_encoder = nn.Linear(edge_in_dim, hidden_dims[0])
        self.initial_node_encoder = nn.Linear(node_in_dim, hidden_dims[0])

        # We'll build each GINEConv to transform dimension: hidden_dims[i] -> hidden_dims[i].
        # After each conv i, if i < len(hidden_dims)-1, we project to hidden_dims[i+1].
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.projections = nn.ModuleList()  # for node features
        self.edge_projections = nn.ModuleList()  # for edge features

        for i in range(len(hidden_dims)):
            net = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i]),
                nn.ReLU(),
                nn.Linear(hidden_dims[i], hidden_dims[i])
            )
            conv = GINEConv(nn=net)
            self.convs.append(conv)
            self.bns.append(BatchNorm(hidden_dims[i]))

            if i < len(hidden_dims) - 1:
                self.projections.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                self.edge_projections.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            else:
                self.projections.append(None)
                self.edge_projections.append(None)

        self.dropout_layer = nn.Dropout(p=dropout)

        final_dim = hidden_dims[-1]
        self.externals_mlp = nn.Sequential(
            nn.Linear(external_in_dim, final_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(final_dim, final_dim)
        )

        self.final_regressor = nn.Sequential(
            nn.Linear(final_dim + final_dim, final_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(final_dim, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.initial_node_encoder(x)
        edge_emb = self.initial_edge_encoder(edge_attr)

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index, edge_emb)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout_layer(x)

            if i < len(self.projections) - 1 and self.projections[i] is not None:
                x = self.projections[i](x)
                edge_emb = self.edge_projections[i](edge_emb)

        graph_emb = global_mean_pool(x, batch)
        ext_emb = self.externals_mlp(data.externals)
        combined = torch.cat([graph_emb, ext_emb], dim=-1)
        out = self.final_regressor(combined).squeeze(-1)
        return out


def objective(trial):
    """
    Objective function for Optuna.
    We do k-fold cross validation using a new GNN model with a non-increasing
    (trapezoidal) architecture. The hyperparameters include learning rate, dropout,
    number of layers, and for each layer, a hidden dimension chosen from a fixed
    candidate set so that each subsequent layer's dimension is <= the previous layer's.
    We return the average RMSE (lower = better). R² is stored in user_attrs.
    Intermediate validation losses are reported for early stopping & dashboard visualization.
    """
    # Hyperparameter search space
    lr = trial.suggest_categorical("learning_rate", [1e-3, 3e-3, 1e-4, 3e-4])
    dropout = trial.suggest_categorical("dropout", [0.1, 0.5])
    num_layers = trial.suggest_int("num_layers", 2, 6)  # 层数限制

    # 固定候选集：所有层均使用相同的候选集
    candidate_values = [200, 128, 64, 32, 16]

    # 固定定义6个隐藏层参数
    hd0 = trial.suggest_categorical("NEW_hidden_dim_0", candidate_values)
    hd1 = trial.suggest_categorical("NEW_hidden_dim_1", candidate_values)
    hd2 = trial.suggest_categorical("NEW_hidden_dim_2", candidate_values)
    hd3 = trial.suggest_categorical("NEW_hidden_dim_3", candidate_values)
    hd4 = trial.suggest_categorical("NEW_hidden_dim_4", candidate_values)
    hd5 = trial.suggest_categorical("NEW_hidden_dim_5", candidate_values)
    hidden_dims_all = [hd0, hd1, hd2, hd3, hd4, hd5]

    # 只使用前 num_layers 个隐藏层
    hidden_dims = hidden_dims_all[:num_layers]

    # 检查是否满足非递增（梯形）结构
    for i in range(1, num_layers):
        if hidden_dims[i] > hidden_dims[i-1]:
            raise optuna.TrialPruned()

    # Early stopping 参数
    max_epochs = 500
    patience = 10
    min_delta = 1e-5

    kf_local = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    r2_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf_local.split(dataset)):
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = PyGDataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = PyGDataLoader(val_subset, batch_size=64, shuffle=False)

        # 构建使用选定梯形隐藏维度的 GNN 模型
        model = GINE_RegressionTrapezoid(
            node_in_dim=1,
            edge_in_dim=2,
            external_in_dim=4,
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = validate(model, val_loader, criterion, device)

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

        r2_fold, rmse_fold = evaluate_model(model, val_loader, device)
        rmse_scores.append(rmse_fold)
        r2_scores.append(r2_fold)

    avg_rmse = float(np.mean(rmse_scores))
    avg_r2 = float(np.mean(r2_scores))

    trial.set_user_attr("avg_r2", avg_r2)
    return avg_rmse


# =============================================================================
#                           OPTUNA STUDY & DASHBOARD
# =============================================================================
if __name__ == "__main__":
    # 使用 load_if_exists=False 以确保使用全新 study，避免旧数据冲突
    study = optuna.create_study(
        storage="sqlite:///gnn_mix_op11.sqlite3",
        study_name="GNN-mixed model different layers05",
        direction="minimize",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=19525, show_progress_bar=True)

    print("\n================= Optuna Study Results =================")
    best_trial = study.best_trial
    print(f"Best Trial Value (RMSE): {best_trial.value}")
    print("Best Hyperparameters:")
    for key, val in best_trial.params.items():
        print(f"  {key}: {val}")
    print(f"User Attrs (R², etc.): {best_trial.user_attrs}")

    try:
        fig1 = vis.plot_optimization_history(study)
        # 保存优化历史图像为 PNG 文件（需要安装 kaleido，例如：pip install kaleido）
        fig1.write_image("optimization_history.png")
    except Exception as e:
        print(f"Could not generate optimization history plot: {e}")

    try:
        fig2 = vis.plot_param_importances(study)
        fig2.write_image("param_importances.png")
    except Exception as e:
        print(f"Could not generate hyperparameter importance plot: {e}")

    try:
        fig3 = vis.plot_intermediate_values(study)
        fig3.write_image("intermediate_values.png")
    except Exception as e:
        print(f"Could not generate intermediate values plot: {e}")

    print("\n================= End of Optuna Tuning =================")
