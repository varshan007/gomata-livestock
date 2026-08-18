import re

with open("colab_train_v32_curvature_monotonic_engine.py", "r") as f:
    code = f.read()

# Replace Phase Title
code = code.replace("PHASE 31 — REGIME-CONDITIONED MONOTONIC SPECTRAL ENGINE", "PHASE 32 — CURVATURE-COMPLETE MONOTONIC SPECTRAL ENGINE")
code = code.replace("PHASE 31", "PHASE 32")

# Insert New Components
new_components = """# ═══════════════════════════════════════════════════════════════
# PHASE 32 ENGINE COMPONENTS
# ═══════════════════════════════════════════════════════════════

class CurvatureSpectralHead(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable positive weights (monotonic constraint)
        self.raw_w1 = nn.Parameter(torch.tensor(0.5))
        self.raw_w2 = nn.Parameter(torch.tensor(0.2))
        self.raw_w3 = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(3)

    def forward(self, lam1, lam2):
        # Spectral basis
        phi1 = lam1
        phi2 = lam1 ** 2
        phi3 = lam1 * lam2

        basis = torch.stack([phi1, phi2, phi3], dim=-1)
        basis = self.norm(basis)

        # Enforce positivity
        w1 = F.softplus(self.raw_w1)
        w2 = F.softplus(self.raw_w2)
        w3 = F.softplus(self.raw_w3)

        score = w1*basis[...,0] + w2*basis[...,1] + w3*basis[...,2]
        return score

class RegimeAmplification(nn.Module):
    def __init__(self):
        super().__init__()
        self.regime_emb = nn.Embedding(4, 8)
        self.mlp = nn.Sequential(
            nn.Linear(8 + 2, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, regime_idx, seed_density, vacc_ratio):
        r = self.regime_emb(regime_idx)
        # Squeeze down to match (B, 10).
        if seed_density.dim() == 3: seed_density = seed_density.squeeze(-1)
        if vacc_ratio.dim() == 3: vacc_ratio = vacc_ratio.squeeze(-1)
        if r.dim() == 3: r = r.squeeze(1)
            
        x = torch.cat([r, seed_density, vacc_ratio], dim=-1)
        amp = self.mlp(x)
        return F.softplus(amp)
"""

code = re.sub(r'class DualHeadPenEngine.*?def compute_pen_outcomes', new_components + '\nclass Phase32Engine(nn.Module):\n    def __init__(self, backbone, max_pens=MAX_PENS, gat_dim=GAT_DIM):\n        super().__init__()\n        self.backbone = backbone\n        for p in self.backbone.parameters(): p.requires_grad = False\n        self.backbone.eval()\n        self.max_pens = max_pens\n        self.gat_dim = gat_dim\n        self.struct_enc = StructuralPenEncoder(in_dim=gat_dim, hidden=64, out_dim=64)\n        self.curvature_head = CurvatureSpectralHead()\n        self.amp_head = RegimeAmplification()\n\n    def compute_pen_outcomes', code, flags=re.DOTALL)

# Re-implement compute_pen_outcomes and forward
new_compute = """    def compute_pen_outcomes(self, pen_struct_basis, regime_idx, seed_density, vacc_ratio):
        B, P, _ = pen_struct_basis.shape

        lam1 = pen_struct_basis[..., 0]
        lam2 = pen_struct_basis[..., 1]
        
        structural = self.curvature_head(lam1, lam2) # [B, P]
        
        amp_scalar = self.amp_head(regime_idx, seed_density, vacc_ratio) # [B, 1]
        amp_exp = amp_scalar.expand(B, P) if amp_scalar.dim() == 2 else amp_scalar
        
        outcome = structural * amp_exp
        return structural, amp_exp, outcome
        
    def forward(self, ns, adj, pen_map, regime_idx, **kwargs):
        self.backbone.eval()
        with torch.no_grad(): 
            bb_out = self.backbone(ns, adj)
            
        B, N, _ = bb_out["H_node"].shape
        
        adj_bin = (adj > 0).float()
        pen_struct_basis = torch.zeros(B, self.max_pens, 3, device=ns.device, dtype=ns.dtype)
        
        for b in range(B):
            bb_h_node = bb_out["H_node"][b] # [N, 96]
            H_herd = bb_out["H_herd"][b].unsqueeze(0) # [1, 96]
            A_b = adj_bin[b]
            pm_b = pen_map[b]
            
            for p in range(self.max_pens):
                mask = (pm_b == p)
                N_p = mask.sum().item()
                if N_p < 2: continue
                
                idx_p = torch.nonzero(mask).squeeze(-1)
                H_pen = bb_h_node[idx_p] # [N_p, 96]
                A_pen = A_b[idx_p][:, idx_p] # [N_p, N_p]
                
                H_feat = H_pen.unsqueeze(0) # [1, N_p, 96]
                A_feat = A_pen.unsqueeze(0) # [1, N_p, N_p]
                
                s_raw, _ = extract_structural_features(A_pen, self.max_pens)
                s_raw = s_raw.unsqueeze(0).to(ns.device)
                
                struct_basis_pred = self.struct_enc(H_feat, A_feat, s_raw).squeeze(0) # [3]
                pen_struct_basis[b, p] = struct_basis_pred

        structural_score, dynamics_embed, pen_out = self.compute_pen_outcomes(
            pen_struct_basis,
            regime_idx,
            kwargs.get("seed_density", torch.zeros_like(regime_idx).float().unsqueeze(-1)),
            kwargs.get("vacc_ratio", torch.zeros_like(regime_idx).float().unsqueeze(-1))
        )
                
        return {
            "delta_R0": bb_out["delta_R0"], "vacc_rank": bb_out["vacc_rank"],
            "outbreak": bb_out["outbreak"], "breakdown": bb_out["breakdown"], 
            "intensity": bb_out["intensity"],
            "H_herd": bb_out["H_herd"],
            "pen_struct_basis": pen_struct_basis,
            "pen_outcome": pen_out,
            "pen_amplification": dynamics_embed,
            "structural_score": structural_score
        }"""

code = re.sub(r'    def compute_pen_outcomes.*?return \{(.*?)\}', new_compute, code, flags=re.DOTALL)

# Phase 32 Loss logic in training loop
new_loss_logic = """            structural_pred = o['structural_score']
            outcome_pred = o['pen_outcome']
            amp_pred_batch = o['pen_amplification']
            
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
                    
            # PHASE 32 LOSS EVALUATION (Applied across both stages)
            Loss_outcome = F.mse_loss(outcome_pred[p_mask_global], p_target[p_mask_global])
            
            std_pred = outcome_pred[p_mask_global].std(unbiased=False)
            var_floor = F.relu(0.15 - std_pred)
            
            structural_mask = structural_pred[p_mask_global]
            monotonic_penalty = F.relu(-torch.mean(structural_mask))

            loss = Loss_struct + Loss_outcome + 5.0 * var_floor + 0.5 * monotonic_penalty

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("⚠️ Numerical instability detected (NaN/Inf Loss) — skipping batch")
                continue
"""

code = re.sub(r'            Loss_struct = sum\(F.mse_loss.*?continue\n', new_loss_logic, code, flags=re.DOTALL)

code = code.replace("model = DualHeadPenEngine(", "model = Phase32Engine(")
code = code.replace("v31_regime_conditioned_engine.pth", "v32_curvature_monotonic_engine.pth")
code = code.replace("v31_config.json", "v32_config.json")

# Replace optimizer initialization and stage B toggle
opt_init = """    opt_struct = torch.optim.AdamW([
        {'params': model.struct_enc.parameters()},
        {'params': model.curvature_head.parameters()}
    ], lr=3e-4, weight_decay=1e-4)
    opt_outcome = torch.optim.AdamW(model.amp_head.parameters(), lr=1e-4, weight_decay=1e-4)"""
code = re.sub(r'    opt_struct = torch.optim.AdamW\(model.struct_enc.*weight_decay=1e-4\)', opt_init, code)

stage_b_toggle = """        if not is_stage_b:
            opt = torch.optim.AdamW([
                {'params': model.struct_enc.parameters()},
                {'params': model.curvature_head.parameters()}
            ], lr=3e-4, weight_decay=1e-4)
            for p in model.amp_head.parameters(): p.requires_grad = False
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            for p in model.amp_head.parameters(): p.requires_grad = True"""
code = re.sub(r'        if not is_stage_b:.*?p.requires_grad = True', stage_b_toggle, code, flags=re.DOTALL)

with open("colab_train_v32_curvature_monotonic_engine.py", "w") as f:
    f.write(code)

print("Patching successful.")
