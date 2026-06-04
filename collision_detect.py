"""
충돌 검출기: SMPL-X 메쉬의 self-interpenetration 검출 (닿음 vs 관통 구분).
부위 = 스키닝 가중치 argmax. 부호거리(정점이 상대 표면 안쪽이면 관통)로 관통깊이/개수 산출.
검사 쌍: L팔↔R팔, L팔↔몸통, R팔↔몸통, (참고)L손↔R손.
usage: python collision_detect.py <clip_dir> [pt_name=fit2d.pt]
out  : 콘솔 리포트 + <clip_dir>/collision_<pt>.json (프레임별 관통 메트릭)
"""
import sys, os, json, time
from pathlib import Path
import numpy as np, torch, smplx, trimesh
from scipy.spatial import cKDTree
sys.path.insert(0, '/workspace/capstone_deploy')

clip = sys.argv[1].rstrip('/'); pt = sys.argv[2] if len(sys.argv) > 2 else 'fit2d.pt'
dev = 'cuda'; EPS = 0.005  # 5mm 이상 안쪽 = 관통
THICK = 0.04  # 진짜 관통은 nearest 거리 < 팔두께(40mm). 멀리 떨어진 false positive 제거
t0 = time.perf_counter()
src = torch.load(f'{clip}/{pt}', map_location='cpu'); inc = src['smpl_params_incam']
F = inc['body_pose'].shape[0]
def fix(t):
    t = (t if torch.is_tensor(t) else torch.tensor(t)).float()
    if t.dim() == 1: t = t.unsqueeze(0)
    if t.shape[0] == 1 and F > 1: t = t.expand(F, -1).contiguous()
    return t.to(dev)
SX = '/workspace/hmp_test/data/body_models/smplx/SMPLX_NEUTRAL.npz'
betas = fix(inc['betas'])
m = smplx.create(SX, model_type='smplx', gender='neutral', use_pca=False,
                 flat_hand_mean=False, num_betas=betas.shape[1], batch_size=F).to(dev)
with torch.no_grad():
    out = m(betas=betas, body_pose=fix(inc['body_pose']).reshape(F, 63),
            global_orient=fix(inc['global_orient']), transl=fix(inc['transl']),
            left_hand_pose=fix(inc['left_hand_pose']), right_hand_pose=fix(inc['right_hand_pose']))
    V = out.vertices.cpu().numpy()   # (F, Nv, 3)
faces = m.faces.astype(np.int64)
print(f'[col] V {V.shape}, faces {faces.shape}  ({time.perf_counter()-t0:.1f}s)')

# ---- 부위 vertex 집합 (스키닝 argmax) ----
W = np.load(SX, allow_pickle=True)['weights']   # (Nv, 55)
part = W.argmax(1)
Lhand = np.where(np.isin(part, [20] + list(range(25, 40))))[0]
Rhand = np.where(np.isin(part, [21] + list(range(40, 55))))[0]
Lfore = np.where(part == 18)[0]; Rfore = np.where(part == 19)[0]
Larm = np.concatenate([Lfore, Lhand]); Rarm = np.concatenate([Rfore, Rhand])
torso = np.where(np.isin(part, [0, 3, 6, 9, 13, 14]))[0]
print(f'[col] verts L팔={len(Larm)} R팔={len(Rarm)} 몸통={len(torso)} Lh={len(Lhand)} Rh={len(Rhand)}')

PAIRS = [('L팔-R팔', Larm, Rarm), ('L팔-몸통', Larm, torso), ('R팔-몸통', Rarm, torso),
         ('L손-R손', Lhand, Rhand)]

def vnormals(verts):
    tri = verts[faces]                                   # (Nf,3,3)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    n = np.linalg.norm(vn, axis=1, keepdims=True); n[n < 1e-9] = 1
    return vn / n

# per-frame 관통 (A의 정점이 B표면 안쪽인 깊이)
res = {nm: {'count': np.zeros(F, int), 'maxdepth': np.zeros(F)} for nm, _, _ in PAIRS}
for f in range(F):
    vn = vnormals(V[f])
    for nm, A, B in PAIRS:
        tree = cKDTree(V[f, B]); d, idx = tree.query(V[f, A])
        signed = np.einsum('ij,ij->i', V[f, A] - V[f, B][idx], vn[B][idx])  # <0 = 안쪽
        pen = np.clip(-signed, 0, None)
        pen[(pen <= EPS) | (d > THICK)] = 0   # 거리 게이트: nearest>40mm = false positive 제거
        res[nm]['count'][f] = int((pen > 0).sum())
        res[nm]['maxdepth'][f] = float(pen.max())
    if f % 300 == 0: print(f'  ...{f}/{F}')

print(f'\n=== 충돌 리포트 ({pt}, F={F}) — 관통깊이 EPS>{EPS*1000:.0f}mm ===')
out_json = {}
for nm, _, _ in PAIRS:
    c = res[nm]['count']; md = res[nm]['maxdepth']
    bad = np.where(c > 3)[0]   # 3정점 이상 관통 = 의미있는 충돌
    worst = np.argsort(md)[::-1][:8]
    print(f'[{nm}] 충돌프레임(>3정점)={len(bad)}/{F} ({100*len(bad)/F:.1f}%)  '
          f'max관통={md.max()*1000:.0f}mm')
    if len(bad):
        print(f'   worst frames(깊이mm): ' + ', '.join(f'f{int(w)}:{md[w]*1000:.0f}' for w in worst if md[w] > EPS))
    out_json[nm] = {'bad_frames': bad.tolist(), 'maxdepth_mm': (md*1000).round(1).tolist()}
json.dump(out_json, open(f'{clip}/collision_{pt.replace(".pt","")}.json', 'w'))
print(f'[col] saved {clip}/collision_{pt.replace(".pt","")}.json  ({time.perf_counter()-t0:.1f}s)')
