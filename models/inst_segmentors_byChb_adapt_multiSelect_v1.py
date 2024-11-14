import torch
from torch import nn


import models.encoders as encoders
from .decoders import  InnerProductDecoder
from .layers import MLP, MappingAndAffineModule
from .modules.domain_adv.domain_discriminator import DomainDiscriminator

class ImprovedBottomHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ImprovedBottomHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.layers(x)

class SFRGNNSegmentor(nn.Module):

    def __init__(self,
                 num_classes,
                 arch,
                 edge_attr_dim,
                 node_attr_dim,
                 edge_attr_emb,
                 node_attr_emb,
                 edge_grid_dim,
                 node_grid_dim,
                 edge_grid_emb,
                 node_grid_emb,
                 num_layers,
                 delta,
                 mlp_ratio=4,
                 drop=0.,
                 drop_path=0.,
                 head_hidden_dim=256,
                 conv_on_edge=True,
                 use_uv_gird=True,
                 use_edge_attr=True,
                 use_face_attr=True,
                 dim_node=256,
                 adaptation_method='madann'):
        super().__init__()

        self.use_uv_gird = use_uv_gird
        self.use_edge_attr = use_edge_attr
        self.use_face_attr = use_face_attr
        self.adaptation_method = adaptation_method
        # Mapping and Affine module
        self.ma_node_module = MappingAndAffineModule(node_attr_dim, 256, node_attr_dim)
        # Node attribute encoder
        self.node_attr_encoder = nn.Sequential(
            nn.Linear(node_attr_dim, node_attr_emb),
            nn.LayerNorm(node_attr_emb),
            nn.ReLU(inplace=True),
            nn.Linear(node_attr_emb, node_attr_emb),
            nn.LayerNorm(node_attr_emb),
            nn.Mish()
        )

        # Node grid encoder (if applicable)
        self.node_grid_encoder = None
        if node_grid_dim:
            self.node_grid_encoder = nn.Sequential(
                nn.Conv2d(node_grid_dim, node_grid_emb // 4, 3, 1, 1),
                nn.BatchNorm2d(node_grid_emb // 4),
                nn.Mish(),
                nn.Conv2d(node_grid_emb // 4, node_grid_emb // 2, 3, 1, 1),
                nn.BatchNorm2d(node_grid_emb // 2),
                nn.Mish(),
                nn.Conv2d(node_grid_emb // 2, node_grid_emb, 3, 1, 1),
                nn.BatchNorm2d(node_grid_emb),
                nn.Mish(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1)
            )

        # Edge attribute encoder
        self.edge_attr_encoder = nn.Sequential(
            nn.Linear(edge_attr_dim, edge_attr_emb),
            nn.LayerNorm(edge_attr_emb),
            nn.ReLU(inplace=True),
            nn.Linear(edge_attr_emb, edge_attr_emb),
            nn.LayerNorm(edge_attr_emb),
            nn.Mish()
        )

        node_emb = node_attr_emb + node_grid_emb
        edge_emb = edge_attr_emb + edge_grid_emb

        # Graph encoder
        encoder = getattr(encoders, arch)
        self.graph_encoder = encoder(node_emb, edge_emb,
                                     num_layers,
                                     delta, mlp_ratio,
                                     drop, drop_path,
                                     conv_on_edge)

        final_out_emb = 2 * node_emb

        # Feature extractor (包括节点嵌入和图嵌入)
        self.feature_extractor = nn.Identity()  # 可以根据需要定义特征提取器，此处简化处理

        # Unified segmentation head
        self.seg_head = MLP(num_layers=2,
                            input_dim=final_out_emb,
                            hidden_dim=head_hidden_dim,
                            output_dim=num_classes,
                            norm=nn.LayerNorm,
                            act=nn.Mish)

        # Instance segmentation head
        Wk = MLP(num_layers=2,
                 input_dim=final_out_emb,
                 hidden_dim=head_hidden_dim,
                 output_dim=head_hidden_dim,
                 norm=nn.LayerNorm,
                 last_norm=True,
                 act=nn.Mish)
        Wq = MLP(num_layers=2,
                 input_dim=final_out_emb,
                 hidden_dim=head_hidden_dim,
                 output_dim=head_hidden_dim,
                 norm=nn.LayerNorm,
                 last_norm=True,
                 act=nn.Mish)

        self.inst_head = InnerProductDecoder(Wq, Wk)
        self.bottom_head = ImprovedBottomHead(input_dim=final_out_emb, hidden_dim=head_hidden_dim)

        # 域判别器
        if self.adaptation_method in ['madann', 'dann']:
            use_sigmoid = True
            self.domain_discriminator = DomainDiscriminator(dim_node, hidden_size=512, use_sigmoid=use_sigmoid)
        else:
            self.domain_discriminator = None  # 对于 MMD 和 TCA，不需要域判别器

    def forward(self, batched_graph):
        """
        Forward pass of the model.
        Returns the segmentation, instance, bottom head outputs and the local-global feature embeddings for domain adaptation.
        """
        # Get input node and edge attributes
        input_node_attr = batched_graph.ndata["x"] if self.use_face_attr else torch.zeros_like(batched_graph.ndata["x"])
        input_node_grid = batched_graph.ndata["grid"] if self.use_uv_gird else torch.zeros_like(batched_graph.ndata["grid"])
        input_edge_attr = batched_graph.edata["x"] if self.use_edge_attr else torch.zeros_like(batched_graph.edata["x"])

        # Apply Mapping and Affine module to both node and edge attributes
        ma_node_attr = self.ma_node_module(input_node_attr)

        # Encode node features
        # node_feat = self.node_attr_encoder(input_node_attr)
        # Node feature encoding
        node_feat = self.node_attr_encoder(ma_node_attr)
        if self.node_grid_encoder:
            node_grid_feat = self.node_grid_encoder(input_node_grid)
            node_feat = torch.cat([node_feat, node_grid_feat], dim=1)

        # Encode edge features
        edge_feat = self.edge_attr_encoder(input_edge_attr)
        node_emb, graph_emb = self.graph_encoder(batched_graph, node_feat, edge_feat)

        # Concatenate node and graph embeddings (local-global features)
        num_nodes_per_graph = batched_graph.batch_num_nodes().to(graph_emb.device)
        graph_emb = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)
        local_global_feat = torch.cat((node_emb, graph_emb), dim=1)


        # Forward pass for segmentation head
        seg_out = self.seg_head(local_global_feat)

        # Forward pass for instance and bottom heads
        inst_out, feat_list = self.inst_head(batched_graph, local_global_feat)
        bottom_out = self.bottom_head(local_global_feat)

        # 返回所有输出，包括特征，用于域自适应
        return seg_out, inst_out, bottom_out, local_global_feat
