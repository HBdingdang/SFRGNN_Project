import torch
from torch import nn
import models.encoders as encoders
from .layers import MLP, MappingAndAffineModule
from .modules.domain_adv.domain_discriminator import DomainDiscriminator

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
                 adaptation_method='madann'):
        super().__init__()

        self.use_uv_gird = use_uv_gird
        self.use_edge_attr = use_edge_attr
        self.use_face_attr = use_face_attr
        self.adaptation_method = adaptation_method  # 新增

        # Mapping and Affine module
        self.ma_node_module = MappingAndAffineModule(node_attr_dim, 256, node_attr_dim)
        # self.ma_edge_module = MappingAndAffineModule(edge_attr_dim, 256, edge_attr_dim)

        # Node attribute encoder
        self.node_attr_encoder = nn.Sequential(
            nn.Linear(node_attr_dim, node_attr_emb),
            nn.LayerNorm(node_attr_emb),
            nn.ReLU(inplace=True),
            nn.Linear(node_attr_emb, node_attr_emb),
            nn.LayerNorm(node_attr_emb),
            nn.Mish()
        )
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
                nn.Flatten(1))
            node_grid_emb_dim = node_grid_emb
        else:
            node_grid_emb_dim = 0

        # Edge attribute encoder
        self.edge_attr_encoder = nn.Sequential(
            nn.Linear(edge_attr_dim, edge_attr_emb),
            nn.LayerNorm(edge_attr_emb),
            nn.ReLU(inplace=True),
            nn.Linear(edge_attr_emb, edge_attr_emb),
            nn.LayerNorm(edge_attr_emb),
            nn.Mish()
        )

        node_emb_dim = node_attr_emb + node_grid_emb_dim
        edge_emb_dim = edge_attr_emb + edge_grid_emb

        # Graph encoder
        encoder = getattr(encoders, arch)
        self.graph_encoder = encoder(node_emb_dim, edge_emb_dim,
                                     num_layers,
                                     delta, mlp_ratio,
                                     drop, drop_path,
                                     conv_on_edge)

        final_out_emb = 2 * node_emb_dim

        # Segmentation head
        self.seg_head = MLP(num_layers=2,
                            input_dim=final_out_emb,
                            hidden_dim=head_hidden_dim,
                            output_dim=num_classes,
                            norm=nn.LayerNorm,
                            act=nn.Mish)


        # 根据 adaptation_method 初始化域判别器
        if self.adaptation_method in ['dann', 'madann']:
            use_sigmoid = True
            # 将 dim_node 替换为 node_emb_dim
            self.domain_discriminator = DomainDiscriminator(node_emb_dim, hidden_size=512, use_sigmoid=use_sigmoid)
        else:
            self.domain_discriminator = None  # 对于 MMD 和 TCA，不需要域判别器

    def forward(self, batched_graph):
        input_node_attr = batched_graph.ndata["x"] if self.use_face_attr else torch.zeros_like(batched_graph.ndata["x"])
        input_node_grid = batched_graph.ndata["grid"] if self.use_uv_gird else torch.zeros_like(
            batched_graph.ndata["grid"])
        input_edge_attr = batched_graph.edata["x"] if self.use_edge_attr else torch.zeros_like(batched_graph.edata["x"])


        # Apply Mapping and Affine module to both node and edge attributes
        ma_node_attr = self.ma_node_module(input_node_attr)
        # ma_edge_attr = self.ma_edge_module(input_edge_attr)

        # Node feature encoding
        # node_feat = self.node_attr_encoder(input_node_attr)

        node_feat = self.node_attr_encoder(ma_node_attr)
        if self.node_grid_encoder:
            # 调整维度以匹配 Conv2d 的输入要求
            # input_node_grid = input_node_grid.unsqueeze(1)  # 假设 input_node_grid 原本是 [N, H, W]
            node_grid_feat = self.node_grid_encoder(input_node_grid)
            node_feat = torch.cat([node_feat, node_grid_feat], dim=1)

        # Edge feature encoding
        edge_feat = self.edge_attr_encoder(input_edge_attr)
        # edge_feat = self.edge_attr_encoder(ma_edge_attr)

        # Graph encoder
        node_emb, graph_emb = self.graph_encoder(batched_graph, node_feat, edge_feat)

        # Concatenate node and graph embeddings
        num_nodes_per_graph = batched_graph.batch_num_nodes().to(graph_emb.device)
        graph_emb_expanded = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0)
        local_global_feat = torch.cat((node_emb, graph_emb_expanded), dim=1)

        # Segmentation output
        seg_out = self.seg_head(local_global_feat)

        # 返回 segmentation 输出和节点嵌入
        return seg_out, node_emb
