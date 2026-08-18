import re

with open("colab_train_v33_powerlaw_epidemic_engine.py", "r") as f:
    code = f.read()

# Replace Phase Title
code = code.replace("PHASE 32 — CURVATURE-COMPLETE MONOTONIC SPECTRAL ENGINE", "PHASE 33 — POWER-LAW EPIDEMIC SPECTRAL ENGINE")
code = code.replace("PHASE 32", "PHASE 33")

# Phase 33 Engine Components
new_components = """# ═══════════════════════════════════════════════════════════════
# PHASE 33 ENGINE COMPONENTS
# ═══════════════════════════════════════════════════════════════

class PowerlawSpectralHead(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable positive weights (monotonic constraint)
        self.raw_w1 = nn.Parameter(torch.tensor(0.5))
        self.raw_w2 = nn.Parameter(torch.tensor(0.2))
        self.raw_w3 = nn.Parameter(torch.tensor(0.1))
        self.raw_w4 = nn.Parameter(torch.tensor(0.2))
        
        # Alpha explicitly defaults to > 1.2
        self.raw_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, lam1, lam2):
        # Explicit epidemic growth modeling (alpha >= 1.2)
        alpha = F.softplus(self.raw_alpha) + 1.2
        
        phi1 = lam1
        phi2 = lam1 ** 2
        phi3 = lam1 * lam2
        
        # Protect against NaN when taking power of exactly 0
        eps = 1e-6
        phi4 = (lam1 + eps) ** alpha
        
        w1 = F.softplus(self.raw_w1)
        w2 = F.softplus(self.raw_w2)
        w3 = F.softplus(self.raw_w3)
        w4 = F.softplus(self.raw_w4)
        
        score = w1*phi1 + w2*phi2 + w3*phi3 + w4*phi4
        
        # Normalize across pens (batch-wise)
        max_score = score.max(dim=1, keepdim=True)[0]
        score = score / (max_score + 1e-6)
        
        return score, alpha

class RegimeAmplification(nn.Module):
    def __init__(self):
        super().__init__()
        self.regime_emb = nn.Embedding(4, 8)
        self.mlp = nn.Sequential(
            nn.Linear(8 + 3, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, regime_idx, outbreak, vacc_rank, seed_density):
        r = self.regime_emb(regime_idx)
        
        if seed_density.dim() == 3: seed_density = seed_density.squeeze(-1)
        if r.dim() == 3: r = r.squeeze(1)
        
        outbreak = outbreak.unsqueeze(1) if outbreak.dim() == 1 else outbreak
        vacc_rank = vacc_rank.unsqueeze(1) if vacc_rank.dim() == 1 else vacc_rank
            
        x = torch.cat([outbreak, vacc_rank, r, seed_density], dim=-1)
        amp = self.mlp(x)
        return F.softplus(amp)

class Phase33Engine(nn.Module):
    def __init__(self, backbone, max_pens=MAX_PENS, gat_dim=GAT_DIM):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters(): p.requires_grad = False
        self.backbone.eval()
        self.max_pens = max_pens
        self.gat_dim = gat_dim
        self.struct_enc = StructuralPenEncoder(in_dim=gat_dim, hidden=64, out_dim=64)
        self.powerlaw_head = PowerlawSpectralHead()
        self.amp_head = RegimeAmplification()

    def compute_pen_outcomes(self, pen_struct_basis, regime_idx, outbreak, vacc_rank, seed_density):
        B, P, _ = pen_struct_basis.shape

        lam1 = pen_struct_basis[..., 0]
        lam2 = pen_struct_basis[..., 1]
        
        structural, alpha = self.powerlaw_head(lam1, lam2) # [B, P], [1]
        
        amp_scalar = self.amp_head(regime_idx, outbreak, vacc_rank, seed_density) # [B, 1]
        amp_exp = amp_scalar.expand(B, P) if amp_scalar.dim() == 2 else amp_scalar
        
        outcome = structural * amp_exp
        return structural, amp_exp, outcome, alpha"""

code = re.sub(r'# ═══════════════════════════════════════════════════════════════\n# PHASE 32 ENGINE COMPONENTS.*def compute_pen_outcomes\(self, pen_struct_basis, regime_idx, seed_density, vacc_ratio\):\n        B, P, _ = pen_struct_basis\.shape\n\n        lam1 = pen_struct_basis\[\.\.\., 0\]\n        lam2 = pen_struct_basis\[\.\.\., 1\]\n        \n        structural = self\.curvature_head\(lam1, lam2\) # \[B, P\]\n        \n        amp_scalar = self\.amp_head\(regime_idx, seed_density, vacc_ratio\) # \[B, 1\]\n        amp_exp = amp_scalar\.expand\(B, P\) if amp_scalar\.dim\(\) == 2 else amp_scalar\n        \n        outcome = structural \* amp_exp\n        return structural, amp_exp, outcome', new_components, code, flags=re.DOTALL)


forward_pass = """        structural_score, dynamics_embed, pen_out, alpha = self.compute_pen_outcomes(
            pen_struct_basis,
            regime_idx,
            bb_out["outbreak"],
            bb_out["vacc_rank"].mean(dim=1),
            kwargs.get("seed_density", torch.zeros_like(regime_idx).float().unsqueeze(-1))
        )
                
        return {
            "delta_R0": bb_out["delta_R0"], "vacc_rank": bb_out["vacc_rank"],
            "outbreak": bb_out["outbreak"], "breakdown": bb_out["breakdown"], 
            "intensity": bb_out["intensity"],
            "H_herd": bb_out["H_herd"],
            "pen_struct_basis": pen_struct_basis,
            "pen_outcome": pen_out,
            "pen_amplification": dynamics_embed,
            "structural_score": structural_score,
            "alpha": alpha
        }"""
        
code = re.sub(r'        structural_score, dynamics_embed, pen_out = self.compute_pen_outcomes\(.*?"structural_score": structural_score\n        \}', forward_pass, code, flags=re.DOTALL)


# Phase 33 Loss Logic
loss_logic = """            structural_pred = o['structural_score']
            outcome_pred = o['pen_outcome']
            amp_pred_batch = o['pen_amplification']
            alpha_val = o['alpha']
            
            Loss_struct = sum(F.mse_loss(struct_basis_pred[p_mask_global, i], s_target_arr[p_mask_global, i]) for i in range(3))
            
            Loss_amp = 0.0; Loss_outcome = 0.0
            p_mask_all = []
            
            for b_i in range(ns.shape[0]):
                p_mask = p_target[b_i] > 0
                if p_mask.sum() >= 2:
                    p_mask_all.append(True)
                    s_cat.extend(struct_basis_pred[b_i][p_mask][:, 0].cpu().detach().numpy())
                    s_t_cat.extend(s_target_arr[b_i][p_mask][:, 0].cpu().numpy())
                    out_p.extend(outcome_pred[b_i][p_mask].cpu().detach().numpy())
                    out_t.extend(p_target[b_i][p_mask].cpu().numpy())
                    amp_pred_c.append(amp_pred_batch[b_i].norm(dim=-1).item())
                    amp_tgt_c.append(amp_tgt_arr[b_i].item())
                else:
                    p_mask_all.append(False)
                    
            # PHASE 33 LOSS EVALUATION 
            Loss_outcome = F.mse_loss(outcome_pred[p_mask_global], p_target[p_mask_global])
            
            std_pred = outcome_pred[p_mask_global].std(unbiased=False)
            var_floor = F.relu(0.15 - std_pred)
            
            # Use raw struct basis lam1 positive mapping for strict structural penalty derivative
            if p_mask_global.sum() >= 2:
                lam1_batch = struct_basis_pred[..., 0] 
                lam1_pos = F.softplus(lam1_batch)
                
                lam1_pos_v = lam1_pos[p_mask_global]
                pen_out_v = outcome_pred[p_mask_global]
                
                diff_lam = lam1_pos_v.unsqueeze(1) - lam1_pos_v.unsqueeze(0)
                diff_out = pen_out_v.unsqueeze(0) - pen_out_v.unsqueeze(1)
                
                monotonic_penalty = F.relu(diff_lam * diff_out).mean()
            else:
                monotonic_penalty = torch.tensor(0.0, device=device)

            if not is_stage_b:
                loss = Loss_struct
            else:
                loss = Loss_outcome + 0.6 * Loss_struct + 0.1 * monotonic_penalty + 0.05 * var_floor
"""

code = re.sub(r'            structural_pred = o\[\'structural_score\'\].*?loss = Loss_struct \+ Loss_outcome \+ 5\.0 \* var_floor \+ 0\.5 \* monotonic_penalty\n', loss_logic, code, flags=re.DOTALL)

code = code.replace("Phase32Engine(", "Phase33Engine(")
code = code.replace("v32_curvature_monotonic_engine.pth", "v33_powerlaw_epidemic_engine.pth")
code = code.replace("v32_config.json", "v33_config.json")

# Optimizer initialization
opt_init = """    opt_struct = torch.optim.AdamW([
        {'params': model.struct_enc.parameters()},
        {'params': model.powerlaw_head.parameters()}
    ], lr=3e-4, weight_decay=1e-4)
    opt_outcome = torch.optim.AdamW(model.amp_head.parameters(), lr=1e-4, weight_decay=1e-4)"""
code = re.sub(r'    opt_struct = torch.optim.AdamW\(\[\s+\{.*?weight_decay=1e-4\)', opt_init, code, flags=re.DOTALL)

# Stage toggle inner
stage_b_toggle = """        if not is_stage_b:
            opt = torch.optim.AdamW([
                {'params': model.struct_enc.parameters()},
                {'params': model.powerlaw_head.parameters()}
            ], lr=3e-4, weight_decay=1e-4)
            for p in model.amp_head.parameters(): p.requires_grad = False
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            for p in model.amp_head.parameters(): p.requires_grad = True"""
code = re.sub(r'        if not is_stage_b:.*?p.requires_grad = True', stage_b_toggle, code, flags=re.DOTALL)


# Print telemetry addition
alpha_log_str = r"""        logger.info(f"   ↳ [Telemetry] std_p:{m_out_p.std():.4f} | std_t:{m_out_t.std():.4f} | Monotonic Violations: {mono_metric:.4f} | Alpha: {alpha_val.item():.4f}")"""
code = re.sub(r'        logger.info\(f"   ↳ \[Telemetry\] std_p:\{m_out_p\.std\(\):\.4f\} \| std_t:\{m_out_t\.std\(\):\.4f\} \| Monotonic Violations: \{mono_metric:\.4f\}"\)', alpha_log_str, code)


with open("colab_train_v33_powerlaw_epidemic_engine.py", "w") as f:
    f.write(code)

print("Patching successful.")
