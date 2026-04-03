import torch 
import torch.nn as nn
import torch.nn.init as init

class VisualProjection(nn.Module):
    """
    独立的视觉特征投影模块
    将不同维度的视觉特征（CLIP/DINOv3/SigLIP）投影到统一的输出空间
    """
    def __init__(self, vis_dim=768, output_dim=768):
        """
        Args:
            vis_dim: 视觉特征输入维度 (CLIP: 768, DINOv3: 1024, SigLIP: 1152)
            output_dim: 输出特征维度 (对齐到 CLIP 空间，通常是 768)
        """
        super(VisualProjection, self).__init__()
        
        self.vis_dim = vis_dim
        self.output_dim = output_dim
        
        # 如果视觉特征维度与输出维度不同，添加两层MLP投影层
        if vis_dim != output_dim:
            # 计算中间维度：视觉维度和输出维度的平均值
            mid_dim = (vis_dim + output_dim) // 2
            self.projection = nn.Sequential(
                nn.Linear(vis_dim, mid_dim),
                nn.LeakyReLU(),
                nn.Linear(mid_dim, output_dim),
                nn.LeakyReLU()
            )
            print(f"🔧 VisualProjection: Created 2-layer MLP {vis_dim} -> {mid_dim} -> {output_dim}")
        else:
            self.projection = nn.Identity()
            print(f"✓ VisualProjection: Visual dim matches output dim ({vis_dim})")
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        """
        Args:
            x: 视觉特征 [..., vis_dim]
        Returns:
            投影后的特征 [..., output_dim]
        """
        return self.projection(x)


class SoftPrompt(nn.Module):
    def __init__(self, query_dim=768, output_dim=768):
        """
        SoftPrompt 模块（不再包含视觉投影层）
        
        Args:
            query_dim: Query 特征维度 (BLIP Q-former 输出，通常是 768)
            output_dim: 输出特征维度 (对齐到 CLIP 空间，通常是 768)
        
        Note:
            视觉特征投影现在由独立的 VisualProjection 模块处理
        """
        super(SoftPrompt, self).__init__()
        
        self.query_dim = query_dim
        self.output_dim = output_dim
        
        #init num of learnable tokens
        self.pos_prompt = nn.Parameter(torch.randn(1, 24, output_dim))
        self.neg_prompt = nn.Parameter(torch.randn(1, 24, output_dim))
        # self.soft_prompt.requires_grad = True
        self.pos_attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=1)
        self.neg_attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=1)

        self.layer1 = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Dropout(0.2),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1, 24),
        )
        
        self.pos_layer0 = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU()
        )

        self.pos_layer1 = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Dropout(0.2),
            nn.LeakyReLU()
        )
        self.pos_layer2 = nn.Sequential(
            nn.Linear(16, 24),
            nn.Dropout(0.1),
            nn.LeakyReLU()
        )

        self.neg_layer0 = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU()
        )
        self.neg_layer1 = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Dropout(0.2),
            nn.LeakyReLU()
        )
        self.neg_layer2 = nn.Sequential(
            nn.Linear(16, 24),
            nn.Dropout(0.1),
            nn.LeakyReLU()
        )

        self.attn_pos_layer0 = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU()
        )

        self.attn_pos_layer1 = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Dropout(0.2),
            nn.LeakyReLU()
        )
        self.attn_pos_layer2 = nn.Sequential(
            nn.Linear(16, 24),
        )

        self.attn_neg_layer0 = nn.Sequential(
            nn.Linear(1, 16),
            # nn.Dropout(0.1),
            nn.LeakyReLU()
        )
        self.attn_neg_layer1 = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Dropout(0.2),
            nn.LeakyReLU()
        )
        self.attn_neg_layer2 = nn.Sequential(
            nn.Linear(16, 24),
        )

        self.blip_pos_layer1 = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.Dropout(0.2),
            nn.LeakyReLU()
        )
        self.blip_pos_layer2 = nn.Sequential(
            nn.Linear(16, 16),
        )

        self.blip_neg_layer1 = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.Dropout(0.2),
            nn.LeakyReLU()
        )
        self.blip_neg_layer2 = nn.Sequential(
            nn.Linear(16, 16),
        )

        self.blip_pos_proj1 = nn.Sequential(
            nn.Linear(16, 8),
            nn.LeakyReLU()
        )
        self.blip_pos_proj2 = nn.Sequential(
            nn.Linear(8, 4),
            nn.LeakyReLU()
        )

        self.blip_neg_proj1 = nn.Sequential(
            nn.Linear(16, 8),
        )
        self.blip_neg_proj2 = nn.Sequential(
            nn.Linear(8, 4),
            nn.LeakyReLU()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                init.constant_(m.bias.data,0)
    
    def forward(self, query, vis_token):
        """
        Args:
            query: BLIP Q-former 查询特征 [B, 16, query_dim]
            vis_token: 已投影到output_dim的视觉特征 [B, output_dim]
        
        Note:
            vis_token 应该已经通过 VisualProjection 投影到 output_dim
        """
        query = query.to(torch.float32)
        vis_token = vis_token.to(torch.float32)

        vis_prompt = self.layer1(vis_token)
        vis_prompt = self.layer2(vis_prompt.T)

        pos_vq_prompt = self.pos_layer0(vis_token.T).T
        pos_vis_prompt = self.pos_layer1(pos_vq_prompt)
        pos_vis_prompt = self.pos_layer2(pos_vis_prompt.T)


        neg_vq_prompt = self.neg_layer0(vis_token.T).T
        neg_vis_prompt = self.neg_layer1(neg_vq_prompt)
        neg_vis_prompt = self.neg_layer2(neg_vis_prompt.T)

        pos_query = self.blip_pos_layer1(query[0])
        pos_query = self.blip_pos_layer2(pos_query.T)

        neg_query = self.blip_neg_layer1(query[0])
        neg_query = self.blip_neg_layer2(neg_query.T)


        pos_attn_prompt1, _ = self.pos_attention(vis_token, pos_query.T, pos_query.T)

        pos_attn_prompt = self.attn_pos_layer0(pos_attn_prompt1.T).T
        pos_attn_prompt = self.attn_pos_layer1(pos_attn_prompt)
        pos_attn_prompt = self.attn_pos_layer2(pos_attn_prompt.T)



        neg_attn_prompt1, _ = self.neg_attention(vis_token, neg_query.T, neg_query.T)

        neg_attn_prompt = self.attn_neg_layer0(neg_attn_prompt1.T).T
        neg_attn_prompt = self.attn_neg_layer1(neg_attn_prompt)
        neg_attn_prompt = self.attn_neg_layer2(neg_attn_prompt.T)

        pos_prompt = (self.pos_prompt + vis_prompt.T + pos_attn_prompt.T + pos_vis_prompt.T)
        neg_prompt = (self.neg_prompt + vis_prompt.T + neg_attn_prompt.T + neg_vis_prompt.T)


        cat_pos_query = self.blip_pos_proj1(query.permute(0,2,1))
        cat_pos_query = self.blip_pos_proj2(cat_pos_query)
        cat_neg_query = self.blip_neg_proj1(query.permute(0,2,1))
        cat_neg_query = self.blip_neg_proj2(cat_neg_query)

        prompt_query_1 = torch.hstack((pos_prompt, cat_pos_query.permute(0,2,1)))
        prompt_query_2 = torch.hstack((neg_prompt, cat_neg_query.permute(0,2,1)))


        return prompt_query_1, prompt_query_2