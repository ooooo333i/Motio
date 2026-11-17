"""
batch_fit2d_v4 (실험1): 손목 overlay 정확도 ↑ + 스파이크 제거.
핵심 변경 (vs batch_fit2d):
  - 2D: 강한 2D + gmof robust(σ=100) → 튀는 2D 관측 자동 downweight.
  - 평활 축분리(local axis-angle, X=axial / Y=dev / Z=flex):
      * swing(Y,Z=comp1,2): Huber 평활 → 빠른 스윙은 보존(L1), jitter만 죽임.
      * axial(X=comp0)     : 강한 L2 평활 + GVHMR로 soft anchor.
        (axial은 2D invisible → GVHMR의 물리적 값으로 당겨도 overlay 손해 0)
  - 손가락은 그대로 고정(=postprocess된 _post).
입력: <clip_dir>/smplx_merged_hamer_post.pt , <clip_dir>/preprocess/wholebody_kp2d.npy
출력: <clip_dir>/fit2d_v4.pt
usage: python batch_fit2d_v4.py <clip_dir> [iters]
주의: _post의 손목 = raw GVHMR init 이라 comp0(axial)의 GVHMR 기준 = bp0 손목 그대로.
"""
import sys, numpy as np, copy, time
sys.path.insert(0, '/workspace/capstone_deploy')
import torch, smplx

clip_dir = sys.argv[1].rstrip('/')
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
base_pt = f'{clip_dir}/smplx_merged_hamer_post.pt'
out_pt = f'{clip_dir}/fit2d_v4.pt'
dev = 'cuda'

# ---- 가중치 (튜닝 포인트) ----
W2D = 1.0           # 2D 항 (batch와 동일 스케일 /1000)
SIGMA = 100.0       # gmof robust scale (px), 01_faithful과 동일
W_SWING = 8.0       # swing(Y,Z) Huber 평활 — WSM=20보다 낮춰 lag↓
HUBER_DELTA = 0.04  # rad(~2.3°): 이하=L2(jitter), 이상=L1(빠른 스윙 보존)
W_AXIAL_SM = 30.0   # axial(X) L2 평활 — invisible이라 강하게
W_AXIAL_ANCHOR = 4.0  # axial(X) → GVHMR soft anchor

t0 = time.perf_counter()
src = torch.load(base_pt, map_location='cpu')
inc = src['smpl_params_incam']
bp_raw = inc['body_pose']; F = bp_raw.shape[0]
bp0 = bp_raw.reshape(F, 21, 3).clone()
K = src['K_fullimg']
if K.dim() == 2: K = K.unsqueeze(0).expand(F, -1, -1)
K = K.float().numpy()
WB = np.load(f'{clip_dir}/preprocess/wholebody_kp2d.npy')   # (F,133,3)
assert WB.shape[0] == F, f'WB {WB.shape[0]} != pose {F}'

def fix(t):
    t = t if torch.is_tensor(t) else torch.tensor(t)
    t = t.float()
    if t.dim() == 1: t = t.unsqueeze(0)
    if t.shape[0] == 1 and F > 1: t = t.expand(F, -1).contiguous()
    return t.to(dev)
betas = fix(inc['betas']); go = fix(inc['global_orient']); tr = fix(inc['transl'])
lhp = fix(inc['left_hand_pose']); rhp = fix(inc['right_hand_pose'])

m = smplx.create('/workspace/hmp_test/data/body_models/smplx/SMPLX_NEUTRAL.npz',
                 model_type='smplx', gender='neutral', use_pca=False,
                 flat_hand_mean=False, num_betas=betas.shape[1], batch_size=F).to(dev)

wr_R = bp0[:, 20].clone().to(dev).requires_grad_(True)
wr_L = bp0[:, 19].clone().to(dev).requires_grad_(True)
bp_fixed = bp0.clone().to(dev)
# GVHMR axial 기준 (comp0) — anchor 대상
gvax_R = bp0[:, 20, 0].clone().to(dev)
gvax_L = bp0[:, 19, 0].clone().to(dev)

RID_S = [21, 40, 43, 46, 49, 52]; RID_W = [112 + i for i in [0, 5, 9, 17, 13, 2]]
LID_S = [20, 25, 28, 31, 34, 37]; LID_W = [91 + i for i in [0, 5, 9, 17, 13, 2]]
Kt = torch.tensor(K, dtype=torch.float32, device=dev)
WBt = torch.tensor(WB, dtype=torch.float32, device=dev)

def proj(J):
    z = J[..., 2].clamp(min=1e-4)
    u = Kt[:, 0, 0:1] * J[..., 0] / z + Kt[:, 0, 2:3]
    v = Kt[:, 1, 1:2] * J[..., 1] / z + Kt[:, 1, 2:3]
    return torch.stack([u, v], -1)

def gmof(res2):  # robust: 큰 잔차 포화
    return (SIGMA**2 * res2) / (SIGMA**2 + res2)

def huber(d, delta):
    a = d.abs()
    return torch.where(a < delta, 0.5 * d**2, delta * (a - 0.5 * delta))

opt = torch.optim.Adam([wr_R, wr_L], lr=0.02)
for it in range(ITERS):
    bp = bp_fixed.clone(); bp[:, 20] = wr_R; bp[:, 19] = wr_L
    out = m(betas=betas, body_pose=bp.reshape(F, 63), global_orient=go, transl=tr,
            left_hand_pose=lhp, right_hand_pose=rhp)
    J = out.joints
    # 2D (robust gmof)
    loss_2d = 0.0
    for SID, WID in [(RID_S, RID_W), (LID_S, LID_W)]:
        pj = proj(J[:, SID]); tgt = WBt[:, WID, :2]; conf = WBt[:, WID, 2:3].clamp(0, 1)
        res2 = ((pj - tgt)**2).sum(-1, keepdim=True)     # (F,6,1)
        loss_2d = loss_2d + (conf * gmof(res2)).mean()
    # 평활 축분리
    sw_R = huber(wr_R[1:, 1:] - wr_R[:-1, 1:], HUBER_DELTA).mean()
    sw_L = huber(wr_L[1:, 1:] - wr_L[:-1, 1:], HUBER_DELTA).mean()
    ax_R = ((wr_R[1:, 0] - wr_R[:-1, 0])**2).mean()
    ax_L = ((wr_L[1:, 0] - wr_L[:-1, 0])**2).mean()
    # axial anchor → GVHMR
    an_R = ((wr_R[:, 0] - gvax_R)**2).mean()
    an_L = ((wr_L[:, 0] - gvax_L)**2).mean()
    loss = (W2D * loss_2d / 1000.0
            + W_SWING * (sw_R + sw_L)
            + W_AXIAL_SM * (ax_R + ax_L)
            + W_AXIAL_ANCHOR * (an_R + an_L))
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 50 == 0 or it == ITERS - 1:
        with torch.no_grad():
            rmse = (loss_2d.item() / 4)**0.5 * 0  # placeholder; 실 rmse 아래 진단서 계산
        print('  v4 it%3d  2d=%.4f swing=%.4f axialSM=%.4f anchor=%.4f'
              % (it, loss_2d.item(), (sw_R + sw_L).item(), (ax_R + ax_L).item(), (an_R + an_L).item()))

out_dict = copy.deepcopy(src)
bpn = bp0.clone(); bpn[:, 20] = wr_R.detach().cpu(); bpn[:, 19] = wr_L.detach().cpu()
for sp in ('smpl_params_incam', 'smpl_params_global'):
    if sp in out_dict:
        out_dict[sp]['body_pose'] = bpn.reshape(F, 21, 3) if out_dict[sp]['body_pose'].dim() == 3 else bpn.reshape(F, 63)
torch.save(out_dict, out_pt)
print(f'[fit2d_v4] saved {out_pt}  ({time.perf_counter()-t0:.1f}s)')

# ---- 진단: 손목 각속도 + ROM 위반 프레임 (physical_limits 기준) ----
from scipy.spatial.transform import Rotation
from physical_limits import check_wrist_rotation, MAX_WRIST_DEG_PER_FRAME
bpc = bpn.numpy()
def vel_list(wr):
    Rr = Rotation.from_rotvec(bpc[:, wr]).as_matrix()
    return np.array([0.0] + [np.degrees(np.linalg.norm(
        Rotation.from_matrix(Rr[f-1].T @ Rr[f]).as_rotvec())) for f in range(1, F)])
for name, wr in [('R', 20), ('L', 19)]:
    v = vel_list(wr)
    spike = np.where(v > 33.3)[0]
    # ROM hard 위반
    rom_bad = [f for f in range(F) if not all(c['in_range'] for c in check_wrist_rotation(bpc[f, wr], 'hard').values())]
    print(f'[diag {name}] vel max={v.max():.0f}deg/f  >33.3:{len(spike)}f  '
          f'ROM-hard위반:{len(rom_bad)}f')
    if len(spike): print(f'   spike frames(>33): {spike[:30].tolist()}')
    if rom_bad: print(f'   ROM-bad frames: {rom_bad[:30]}')
