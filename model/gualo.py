import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_models import GNN, NodeEncoder
from .gnn_crosstr import TR, SynthesisTransformerEncoder


LOG_VAR_MIN = -10.0
LOG_VAR_MAX = 10.0
MODEL_VERSION = "gualo-v3-prefusion-20261117"


def enable_dropout(model):

    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
        elif isinstance(m, nn.TransformerEncoderLayer):

            m.train()


class ProjectionHead(nn.Module):


    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class GUALO(nn.Module):


    def __init__(
        self,
        node_emb_size=2,
        gnnemb_size=128,
        synemb_size=128,
        batch_size=1,
        sm_size=64,
        lg_size=256,
        proj_dim=128,
        temperature=0.1,
        mc_samples=30,
        num_synthesis_actions=14,
        synth_num_layers=4,
        synth_num_heads=4,
        synth_ff_dim=512,
        synth_dropout=0.4,
        gnn_num_layers=5,
        graph_normalization="running",
    ):
        super().__init__()
        self.node_emb_size = node_emb_size
        self.gnnemb_size = gnnemb_size
        self.synemb_size = synemb_size
        self.batch_size = batch_size
        self.temperature = temperature
        self.mc_samples = mc_samples
        self.model_version = MODEL_VERSION
        self.gnn_num_layers = gnn_num_layers
        self.graph_normalization = graph_normalization

        self.nodeencoder = NodeEncoder(node_emb_size)
        self.gnn = GNN(self.nodeencoder, node_emb_size * 5, gnnemb_size,
                       n_layers=gnn_num_layers, normalization=graph_normalization)
        self.synth_transformer = SynthesisTransformerEncoder(
            num_synthesis_actions=num_synthesis_actions,
            embedding_dim=synemb_size,
            num_layers=synth_num_layers,
            num_heads=synth_num_heads,
            ff_dim=synth_ff_dim,
            dropout=synth_dropout,
        )


        self.transformer = TR(
            self.gnn,
            self.synth_transformer,
            batch_size,
            gnnemb_size,
            synemb_size,
            num_classes=1,
            sm_dim=sm_size,
            lg_dim=lg_size,
        )


        self.graph_proj_head = ProjectionHead(2 * gnnemb_size, hidden_dim=256, output_dim=proj_dim)
        self.recipe_proj_head = ProjectionHead(synemb_size, hidden_dim=256, output_dim=proj_dim)


    def _split_prediction(self, pred_2col):

        mu = pred_2col[:, :1]
        log_var = pred_2col[:, 1:].clamp(min=LOG_VAR_MIN, max=LOG_VAR_MAX)
        return mu, log_var


    def forward(self, batch, return_uncertainty=False, return_contrastive=False):

        if return_uncertainty:
            modes = [(module, module.training) for module in self.modules()]
            self.eval()
            enable_dropout(self)

            mus, log_vars = [], []
            graph_projs, recipe_projs = [], []
            try:
                with torch.no_grad():
                    for _ in range(self.mc_samples):
                        pred, graph_emb, recipe_emb = self.transformer(batch, return_embeddings=True)
                        mu_t, lv_t = self._split_prediction(pred)
                        mus.append(mu_t)
                        log_vars.append(lv_t)
                        if return_contrastive:
                            graph_projs.append(self.graph_proj_head(graph_emb))
                            recipe_projs.append(self.recipe_proj_head(recipe_emb))
            finally:
                for module, training in modes:
                    module.training = training

            mu_stack = torch.stack(mus)
            lv_stack = torch.stack(log_vars)
            mean_mu = mu_stack.mean(dim=0)
            epistemic_std = mu_stack.std(dim=0, unbiased=False)

            mean_log_var = 2 * torch.log(torch.exp(0.5 * lv_stack).mean(dim=0))

            if return_contrastive:
                gp = F.normalize(torch.stack(graph_projs).mean(dim=0), dim=-1)
                rp = F.normalize(torch.stack(recipe_projs).mean(dim=0), dim=-1)
                return mean_mu, mean_log_var, epistemic_std, gp, rp
            return mean_mu, mean_log_var, epistemic_std

        if return_contrastive:
            pred, graph_emb, recipe_emb = self.transformer(batch, return_embeddings=True)
            mu, log_var = self._split_prediction(pred)
            graph_proj = self.graph_proj_head(graph_emb)
            recipe_proj = self.recipe_proj_head(recipe_emb)
            return mu, log_var, graph_proj, recipe_proj

        pred = self.transformer(batch)
        mu, log_var = self._split_prediction(pred)
        return mu, log_var
