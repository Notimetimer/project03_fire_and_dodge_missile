import torch
import torch.nn as nn
import torch.nn.functional as F


# GRU-MLP
class GruMlp(nn.Module):
    def __init__(self, input_dim, gru_hidden_size, gru_num_layers=1, output_dim=None, batch_first=True):
        super(GruMlp, self).__init__()
        self.input_dim = input_dim
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers  # 添加GRU层数参数
        self.output_dim = output_dim

        # GRU层处理序列数据，增加num_layers参数
        self.gru = nn.GRU(input_dim, gru_hidden_size, num_layers=gru_num_layers, batch_first=batch_first)

        # MLP输出层
        self.mlp_out = nn.Linear(gru_hidden_size, output_dim)

        # # debug: 在 GruMlp 初始化完成后立即检查 self.gru._flat_weights 类型
        # if hasattr(self, "gru") and hasattr(self.gru, "_flat_weights") and isinstance(self.gru._flat_weights, list):
        #     import traceback
        #     raise RuntimeError("DEBUG: GruMlp created with gru._flat_weights as list; "
        #                        f"element types: {[type(w) for w in self.gru._flat_weights]}\\n"
        #                        f"stack:\\n{''.join(traceback.format_stack())}")

    def forward(self, x, h_0=None):
        # x: (B, S, D)
        B, S, D = x.shape

        # 如果没有提供初始隐藏状态，则使用None，让PyTorch自动初始化
        # h_0: (num_layers, B, gru_hidden_size) 或 None

        # # 检查 GRU 内部 flat weights 类型，若为 list 则抛出错误以便定位问题源
        # if hasattr(self.gru, "_flat_weights") and isinstance(self.gru._flat_weights, list):
        #     raise TypeError(
        #         "GRU internal _flat_weights is a list (unexpected). "
        #         "This indicates GRU parameters were mutated/ wrapped incorrectly. "
        #         f"_flat_weights element types: {[type(w) for w in self.gru._flat_weights]}"
        #     )
        # # 确保 h_0 是 Tensor 或 None
        # if h_0 is not None and not isinstance(h_0, torch.Tensor):
        #     raise TypeError(f"h_0 must be a torch.Tensor or None, got {type(h_0)}")
        # # 确保 h_0 与输入在同一 device
        # if h_0 is not None and isinstance(h_0, torch.Tensor) and h_0.device != x.device:
        #     raise RuntimeError(f"h_0.device ({h_0.device}) != x.device ({x.device}); move h_0 to input device before calling forward")

        # GRU处理
        gru_output, h_n = self.gru(x, h_0)  # gru_output: (B, S, gru_hidden_size), h_n: (num_layers, B, gru_hidden_size)

        # MLP输出
        output = self.mlp_out(gru_output)  # (B, S, output_dim)

        # 返回输出和最终隐藏状态
        return output, h_n


# ATT-MLP
class AttMlp(nn.Module):
    def __init__(self, input_dim, feature_embed_dim, attn_heads, output_dim,
                 dropout=0.0, bias=True, batch_first=False):
        super(AttMlp, self).__init__()
        self.input_dim = input_dim  # D
        self.feature_embed_dim = feature_embed_dim  # n1
        self.output_dim = output_dim
        self.batch_first = batch_first

        # 特征嵌入层
        self.feature_embedding_layer = nn.Linear(1, feature_embed_dim)

        # 多头自注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=feature_embed_dim,
                                               num_heads=attn_heads,
                                               dropout=dropout,
                                               bias=bias,
                                               batch_first=batch_first)

        # MLP输出层
        self.mlp_out = nn.Linear(feature_embed_dim, output_dim)

    def forward(self, x):
        # x: (B, S, D)
        B, S, D = x.shape

        # 为特征注意力准备 (B, S, D) -> (B*S, D, 1)
        x_reshaped = x.view(B * S, D, 1)

        # 嵌入每个1维特征到n1维 (B*S, D, n1)
        x_embedded = self.feature_embedding_layer(x_reshaped)

        # 多头自注意力
        if self.batch_first:
            # (B*S, D, n1)
            attn_output, _ = self.attention(x_embedded, x_embedded, x_embedded)
        else:
            # (D, B*S, n1)
            attn_input = x_embedded.permute(1, 0, 2)
            attn_output, _ = self.attention(attn_input, attn_input, attn_input)
            attn_output = attn_output.permute(1, 0, 2)  # (B*S, D, n1)

        # 池化/聚合特征注意力结果 (B*S, n1)
        pooled_features = attn_output.mean(dim=1)

        # 重塑回原始序列结构 (B, S, n1)
        final_output = pooled_features.view(B, S, self.feature_embed_dim)

        # MLP 输出 (B, S, output_dim)
        output = self.mlp_out(final_output)

        return output


# ATT-GRU-MLP
class AttGruMlp(nn.Module):
    def __init__(self, input_dim, feature_embed_dim, attn_heads, gru_hidden_size, gru_num_layers, output_dim,
                 dropout=0.0, bias=True, batch_first=True):
        super(AttGruMlp, self).__init__()
        # 初始化注意力模块，其输出维度为GRU的输入维度
        self.att = AttMlp(input_dim=input_dim,
                          feature_embed_dim=feature_embed_dim,
                          attn_heads=attn_heads,
                          output_dim=feature_embed_dim,  # ATT的输出作为GRU的输入
                          dropout=dropout,
                          bias=bias,
                          batch_first=batch_first)

        # 初始化GRU模块，其输入维度是注意力模块的输出维度
        self.gru_mlp = GruMlp(input_dim=feature_embed_dim,
                              gru_hidden_size=gru_hidden_size,
                              gru_num_layers=gru_num_layers,
                              output_dim=output_dim)

    def forward(self, x, h_0=None):
        # x: (B, S, D)

        # Step 1: Attention处理
        # att_output: (B, S, feature_embed_dim)
        att_output = self.att(x)

        # Step 2: GRU处理
        # output: (B, S, output_dim), h_n: (num_layers, B, gru_hidden_size)
        output, h_n = self.gru_mlp(att_output, h_0)

        return output, h_n


# GRU-ATT-MLP
class GruAttMlp(nn.Module):
    def __init__(self, input_dim, gru_hidden_size, gru_num_layers, feature_embed_dim, attn_heads, output_dim,
                 dropout=0.0, bias=True, batch_first=True):
        super(GruAttMlp, self).__init__()
        # 初始化GRU模块，其输出维度为Attention的输入维度
        self.gru_mlp = GruMlp(input_dim=input_dim,
                              gru_hidden_size=gru_hidden_size,
                              gru_num_layers=gru_num_layers,
                              output_dim=gru_hidden_size)  # GRU的输出作为ATT的输入

        # 初始化Attention模块，其输入维度是GRU模块的隐藏层大小
        self.att = AttMlp(input_dim=gru_hidden_size,
                          feature_embed_dim=feature_embed_dim,
                          attn_heads=attn_heads,
                          output_dim=output_dim,
                          dropout=dropout,
                          bias=bias,
                          batch_first=batch_first)

    def forward(self, x, h_0=None):
        # x: (B, S, D)

        # Step 1: GRU处理
        # gru_output: (B, S, gru_hidden_size)
        gru_output, h_n = self.gru_mlp(x, h_0)

        # Step 2: Attention处理
        # output: (B, S, output_dim)
        output = self.att(gru_output)

        return output, h_n
