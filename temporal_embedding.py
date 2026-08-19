import torch
import torch.nn as nn

class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time  # e.g., 288 or 48
        self.features = features

        # Learned temporal embeddings
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

        # Cycle-based embeddings projected to feature space
        self.cycle_day_proj = nn.Linear(1, features)
        self.cycle_week_proj = nn.Linear(1, features)

    def forward(self, x):
        # x shape: [B, C, N, T]
        B, C, N, T = x.shape

        if C >= 3:
            # ----- Time of day (channel 1 at current/last time step) -----
            day_raw = x[:, 1, :, -1]  # [B, N]
            day_idx = (
                (day_raw * self.time).long()
                if day_raw.max() <= 1.0
                else day_raw.long()
            ).clamp(0, self.time - 1)
            time_day = self.time_day[day_idx]  # [B, N, F]
            time_day = time_day.permute(0, 2, 1).unsqueeze(-1)  # [B, F, N, 1]

            # ----- Day of week (channel 2 at current/last time step) -----
            week_raw = x[:, 2, :, -1]  # [B, N]
            week_idx = week_raw.long().clamp(0, 6)
            time_week = self.time_week[week_idx]  # [B, N, F]
            time_week = time_week.permute(0, 2, 1).unsqueeze(-1)  # [B, F, N, 1]

            # ----- Cycle-based embeddings -----
            norm_day = day_raw.unsqueeze(-1)  # [B, N, 1]
            norm_week = week_raw.unsqueeze(-1)  # [B, N, 1]

            angle_day = (
                norm_day if norm_day.max() <= 1.0 else norm_day / self.time
            ) * 2 * torch.pi
            angle_week = norm_week * 2 * torch.pi / 7.0

            sincos_day = torch.sin(angle_day) + torch.cos(angle_day)  # [B, N, 1]
            sincos_week = torch.sin(angle_week) + torch.cos(angle_week)  # [B, N, 1]

            cycle_day = self.cycle_day_proj(sincos_day).permute(0, 2, 1).unsqueeze(-1)  # [B, F, N, 1]
            cycle_week = self.cycle_week_proj(sincos_week).permute(0, 2, 1).unsqueeze(-1)  # [B, F, N, 1]

            # ----- Combine -----
            tem_emb = time_day + time_week + cycle_day + cycle_week  # [B, F, N, 1]
        else:
            tem_emb = torch.zeros(B, self.features, N, 1, device=x.device)

        return tem_emb
