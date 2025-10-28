import torch
import torch.nn as nn
import torch.nn.functional as F


# GRU-MLP
class GruMlp(nn.Module):
    def __init__(self, input_dim, gru_hidden_size, output_dim):
        super(GruMlp, self).__init__()
        self.input_dim = input_dim
        self.gru_hidden_size = gru_hidden_size
        self.output_dim = output_dim

        # GRU层处理序列数据
        self.gru = nn.GRU(input_dim, gru_hidden_size, batch_first=True)

        # MLP输出层
        self.mlp_out = nn.Linear(gru_hidden_size, output_dim)

    def forward(self, x):
        # x: (B, S, D)
        B, S, D = x.shape

        # GRU处理
        gru_output, _ = self.gru(x)  # (B, S, gru_hidden_size)

        # MLP输出
        output = self.mlp_out(gru_output)  # (B, S, output_dim)

        return output


# ATT-MLP
class AttMlp(nn.Module):
    def __init__(self, input_dim, feature_embed_dim, attn_heads, output_dim):
        super(AttMlp, self).__init__()
        self.input_dim = input_dim  # D
        self.feature_embed_dim = feature_embed_dim  # n1
        self.output_dim = output_dim

        # 特征嵌入层
        self.feature_embedding_layer = nn.Linear(1, feature_embed_dim)

        # 多头自注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=feature_embed_dim,
                                               num_heads=attn_heads,
                                               batch_first=False)

        # MLP输出层
        self.mlp_out = nn.Linear(feature_embed_dim, output_dim)

    def forward(self, x):
        # x: (B, S, D)
        B, S, D = x.shape

        # 为特征注意力准备 (B, S, D) -> (B*S, D, 1) -> (B*S, D, n1)
        x_reshaped = x.view(B * S, D, 1)
        x_embedded = self.feature_embedding_layer(x_reshaped)  # (B*S, D, n1)

        # 多头自注意力
        attn_input = x_embedded.permute(1, 0, 2)  # (D, B*S, n1)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        attn_output = attn_output.permute(1, 0, 2)  # (B*S, D, n1)

        # 池化/聚合特征注意力结果
        pooled_features = attn_output.mean(dim=1)  # (B*S, n1)

        # 重塑回原始序列结构
        mlp_input = pooled_features.view(B, S, self.feature_embed_dim)  # (B, S, n1)

        # MLP输出
        output = self.mlp_out(mlp_input)  # (B, S, output_dim)

        return output


# ATT-GRU-MLP
class AttGruMlp(nn.Module):
    def __init__(self, input_dim, feature_embed_dim, attn_heads, gru_hidden_size, output_dim):
        super(AttGruMlp, self).__init__()
        self.input_dim = input_dim  # Original D
        self.feature_embed_dim = feature_embed_dim  # n1
        self.gru_hidden_size = gru_hidden_size
        self.output_dim = output_dim

        # Step 2: 嵌入层升维
        # Original idea: (B*S, D, 1) -> (B*S, D, n1)
        # More direct: (B*S, D) -> (B*S, D, n1) by treating each D element as an input for the linear layer
        # A simple linear layer can process the last dimension (D) and project it to feature_embed_dim
        # No, a better way to think about (B*S, D, 1) -> (B*S, D, n1) is each feature (1) is embedded.
        # This implies a linear layer that takes 1 as input and outputs n1.
        # It's more common to have (B*S, D) and then project each "feature item" D to n1.
        # So we treat each D as a sequence element of n1 features
        self.feature_embedding_layer = nn.Linear(1, feature_embed_dim)
        # Alternatively, if D is itself considered an embed_dim for a sequence of 1,
        # and we want to expand the '1' to 'n1' for each feature.
        # More likely, we view D as sequence_length, and each feature has an original embed_dim=1.

        # Let's adjust for the (B*S, D, 1) -> (B*S, D, n1) interpretation:
        # Each "feature value" in D becomes a single-element sequence (embedding dim of 1).
        # We want to embed this single element into feature_embed_dim.
        # So the linear layer will operate on the *last* dimension (1).

        # Step 3: 多头自注意力
        # input: (Batch*SeqLen, D, n1) -> (SeqLen (D), Batch*SeqLen, EmbedDim (n1)) for MHA
        self.attention = nn.MultiheadAttention(embed_dim=feature_embed_dim, num_heads=attn_heads,
                                               batch_first=False)  # Or batch_first=True if you permute to (B*S, n1, D)

        # Step 4 & 5: GRU 处理 (GRU takes (Batch, SeqLen, Feature_Dim))
        # After pooling, output will be (B*S, n1), reshaped to (B, S, n1)
        self.gru = nn.GRU(feature_embed_dim, gru_hidden_size, batch_first=True)

        # Step 6: MLP 输出
        self.mlp_out = nn.Linear(gru_hidden_size, output_dim)

    def forward(self, x):
        # x: (B, S, D)

        B, S, D = x.shape

        # Step 2: 为特征注意力准备 (B, S, D) -> (B*S, D, 1) -> (B*S, D, n1)
        # Reshape to (B*S, D) then unsqueeze to (B*S, D, 1)
        x_reshaped = x.view(B * S, D, 1)  # Each feature value is now a single-dim embedding

        # Embed each 1-dim feature into n1-dim
        # x_embedded: (B*S, D, n1)
        x_embedded = self.feature_embedding_layer(x_reshaped)

        # Step 3: 多头自注意力
        # MultiheadAttention expects (SeqLen, Batch, EmbedDim) if batch_first=False
        # Our current x_embedded is (B*S, D, n1).
        # D is our sequence_length, B*S is our "batch_size" for MHA, n1 is embed_dim
        # So we need to permute to (D, B*S, n1)
        attn_input = x_embedded.permute(1, 0, 2)  # (D, B*S, n1)

        # attn_output: (D, B*S, n1), attn_weights: (B*S, D, D)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)

        # Permute back to (B*S, D, n1) for pooling
        attn_output = attn_output.permute(1, 0, 2)  # (B*S, D, n1)

        # Step 4: 池化/聚合特征注意力结果
        # Mean pool along D dimension: (B*S, D, n1) -> (B*S, n1)
        pooled_features = attn_output.mean(dim=1)  # (B*S, n1)

        # Step 5: 重塑回原始序列结构
        # (B*S, n1) -> (B, S, n1)
        gru_input = pooled_features.view(B, S, self.feature_embed_dim)

        # Step 6: GRU 处理
        gru_output, _ = self.gru(gru_input)  # (B, S, gru_hidden_size)

        # Step 7: MLP 输出
        output = self.mlp_out(gru_output)  # (B, S, output_dim)

        return output


# GRU-ATT-MLP
class GruAttMlp(nn.Module):
    def __init__(self, input_dim, gru_hidden_size, feature_embed_dim, attn_heads, output_dim):
        super(GruAttMlp, self).__init__()
        self.input_dim = input_dim
        self.gru_hidden_size = gru_hidden_size  # G
        self.feature_embed_dim = feature_embed_dim  # n1
        self.output_dim = output_dim

        # Step 2: GRU 处理
        self.gru = nn.GRU(input_dim, gru_hidden_size, batch_first=True)

        # Step 3: 为特征注意力准备 (B*S, G, 1) -> (B*S, G, n1)
        self.feature_embedding_layer = nn.Linear(1, feature_embed_dim)

        # Step 4: 多头自注意力
        # input: (G, B*S, n1) if batch_first=False
        self.attention = nn.MultiheadAttention(embed_dim=feature_embed_dim, num_heads=attn_heads, batch_first=False)

        # Step 6: MLP 输出 (after pooling (B*S, n1) and reshaping (B, S, n1))
        self.mlp_out = nn.Linear(feature_embed_dim, output_dim)

    def forward(self, x):
        # x: (B, S, D)

        B, S, D = x.shape

        # Step 2: GRU 处理
        gru_output, _ = self.gru(x)  # (B, S, gru_hidden_size (G))
        G = gru_output.shape[2]  # Get actual G

        # Step 3: 为特征注意力准备 (B, S, G) -> (B*S, G, 1) -> (B*S, G, n1)
        x_reshaped = gru_output.view(B * S, G, 1)
        x_embedded = self.feature_embedding_layer(x_reshaped)  # (B*S, G, n1)

        # Step 4: 多头自注意力
        attn_input = x_embedded.permute(1, 0, 2)  # (G, B*S, n1)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        attn_output = attn_output.permute(1, 0, 2)  # (B*S, G, n1)

        # Step 5: 池化/聚合特征注意力结果
        pooled_features = attn_output.mean(dim=1)  # (B*S, n1)

        # Step 6: 重塑回原始序列结构 (Optional if not passing through another GRU)
        # pooled_features is already (B*S, n1), for MLP needs (B, S, n1)
        final_attn_output = pooled_features.view(B, S, self.feature_embed_dim)

        # Step 7: MLP 输出
        output = self.mlp_out(final_attn_output)  # (B, S, output_dim)

        return output
