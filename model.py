import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

class CrossModalAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(CrossModalAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.scaled_dot_attn = nn.MultiheadAttention(d_model, nhead)
        
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feat1, feat2):
        # feat1 and feat2 are of shape (batch_size, seq_length, d_model)
        if len(feat1.shape) == 2:
            feat1 = feat1.unsqueeze(1)
        if len(feat2.shape) == 2:
            feat2 = feat2.unsqueeze(1)
        # Project inputs to queries, keys, and values
        queries = self.query_proj(feat1)
        keys = self.key_proj(feat2)
        values = self.value_proj(feat2)
        
        # Transpose for scaled dot-product attention
        queries = queries.permute(1, 0, 2)   # (seq_length, batch_size, d_model)
        keys = keys.permute(1, 0, 2)         # (seq_length, batch_size, d_model)
        values = values.permute(1, 0, 2)     # (seq_length, batch_size, d_model)
        
        # Apply Scaled Dot-Product Attention
        attn_output, _ = self.scaled_dot_attn(queries, keys, values)
        
        # Transpose back to original shape for output projection
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)

        # Project output and apply normalization
        output = self.output_proj(attn_output)
        output = self.norm(output + feat1)
        
        return output

class ConViTac(nn.Module):
    def __init__(self, num_class=100, n_list=2):
        super(ConViTac, self).__init__()
        self.touch_extractor = CLIPVisionModelWithProjection.from_pretrained("path_to_your_ckpt")
        self.vis_extractor = CLIPVisionModelWithProjection.from_pretrained("path_to_your_ckpt")
        self.tou_proj = nn.Linear(n_list*512, 512)
        self.touch_extractor.requires_grad_(True)
        self.vis_extractor.requires_grad_(True)

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("path_to_your_ckpt")
        self.feature_extractor = CLIPImageProcessor.from_pretrained("path_to_your_ckpt")
        self.image_encoder.requires_grad_(False)
        self.clip_image_mean = torch.as_tensor(self.feature_extractor.image_mean)[:,None,None].cuda()
        self.clip_image_std = torch.as_tensor(self.feature_extractor.image_std)[:,None,None].cuda()

        self.fc = nn.Linear(1024, num_class)

        self.cross_modal_attention = CrossModalAttention(d_model=1024, nhead=8)

    def forward(self, x, vis):
        B, frames, C, H, W = x.shape
        processed_features = []
        
        for frame in range(frames):
            frame_features = self.touch_extractor(x[:, frame, :, :, :]).image_embeds  # Shape: [16, 512]
            processed_features.append(frame_features)
                
        touch_feat = self.tou_proj(torch.concat(processed_features, dim=1))
        vis_feat = self.vis_extractor(vis).image_embeds

        vis_in_proc = TF.resize(vis, (self.feature_extractor.crop_size['height'], self.feature_extractor.crop_size['width']), interpolation=InterpolationMode.BICUBIC, antialias=True)
        vis_in_proc = ((vis_in_proc.float() - self.clip_image_mean) / self.clip_image_std)

        vis_clip = self.image_encoder(vis_in_proc).image_embeds # shape: 16, 512
        
        tou_in_proc = TF.resize(x[:, 0, :, :, :], (self.feature_extractor.crop_size['height'], self.feature_extractor.crop_size['width']), interpolation=InterpolationMode.BICUBIC, antialias=True)
        tou_in_proc = ((tou_in_proc.float() - self.clip_image_mean) / self.clip_image_std)

        tou_clip = self.image_encoder(tou_in_proc).image_embeds # shape: 16, 512

        condition = torch.concat([vis_clip, tou_clip], dim=1) # shape: 16, 1024
        
        vt_feat = torch.cat([vis_feat, touch_feat], dim=1)        

        # Apply cross-modal attention
        out = self.cross_modal_attention(condition, vt_feat)

        # Flatten for fully connected layer
        out = out.view(B, -1)  # Shape: [B, C*H*W]
        
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

