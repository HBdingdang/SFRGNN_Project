import torch
from torch import nn

import models.encoders as encoders
from .decoders import  InnerProductDecoder
from .layers import  MLP


class ImprovedBottomHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ImprovedBottomHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # 使用GELU激活函数
            nn.Dropout(0.3),  # 添加Dropout层防止过拟合
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),  # 使用GELU激活函数
            nn.Dropout(0.3),  # 添加Dropout层防止过拟合
            nn.Linear(hidden_dim // 2, 1)  # 输出为1
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
                 dim_node=256):
        super().__init__()
        self.num_classes = num_classes

        self.use_uv_gird = use_uv_gird
        self.use_edge_attr = use_edge_attr
        self.use_face_attr = use_face_attr


        # 节点属性编码器
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

        # 边属性编码器
        self.edge_attr_encoder = nn.Sequential(
            nn.Linear(edge_attr_dim, edge_attr_emb),
            nn.LayerNorm(edge_attr_emb),
            nn.ReLU(inplace=True),
            nn.Linear(edge_attr_emb, edge_attr_emb),
            nn.LayerNorm(edge_attr_emb),
            nn.Mish()
        )
        if edge_grid_dim:#****************** 未实现curve grid 编码 ，实现后输出维度应该为64D******************************************
            # TODO
            pass

        node_emb = node_attr_emb + node_grid_emb
        edge_emb = edge_attr_emb + edge_grid_emb

        encoder = getattr(encoders, arch)
        self.graph_encoder = encoder(node_emb, edge_emb,
                                     num_layers,
                                     delta, mlp_ratio,
                                     drop, drop_path,
                                     conv_on_edge)

        final_out_emb = 2 * node_emb

        self.seg_head = MLP(num_layers=2,
                            input_dim=final_out_emb,
                            hidden_dim=head_hidden_dim,
                            output_dim=num_classes,
                            norm=nn.LayerNorm,
                            act=nn.Mish)
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

    def forward(self, batched_graph):
        input_node_attr = batched_graph.ndata["x"] if self.use_face_attr else torch.zeros_like(batched_graph.ndata["x"])
        input_node_grid = batched_graph.ndata["grid"] if self.use_uv_gird else torch.zeros_like(batched_graph.ndata["grid"])
        input_edge_attr = batched_graph.edata["x"] if self.use_edge_attr else torch.zeros_like(batched_graph.edata["x"])

        node_feat = self.node_attr_encoder(input_node_attr)
        if self.node_grid_encoder:
            node_grid_feat = self.node_grid_encoder(input_node_grid)
            node_feat = torch.concat([node_feat, node_grid_feat], dim=1)

        edge_feat = self.edge_attr_encoder(input_edge_attr)

        node_emb, graph_emb = self.graph_encoder(batched_graph, node_feat, edge_feat)
        # concatenated to the per-node embeddings
        num_nodes_per_graph = batched_graph.batch_num_nodes().to(graph_emb.device)
        graph_emb = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)
        local_global_feat = torch.cat((node_emb, graph_emb), dim=1)
        # Map to logits
        seg_out = self.seg_head(local_global_feat)
        inst_out, feat_list = self.inst_head(batched_graph, local_global_feat)

        bottom_out = self.bottom_head(local_global_feat)
        return seg_out, inst_out, bottom_out
        # return seg_out, inst_out, bottom_out, local_global_feat #应用于域自适应训练
