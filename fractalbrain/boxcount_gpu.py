# Vectorized 3D box-counting compatible with numpy OR cupy via backends.xp()
from .backends import xp, asnumpy

def _pad_to_multiple(vol, s):
    X = xp()
    z,y,x = vol.shape
    padz = (s - (z % s)) % s
    pady = (s - (y % s)) % s
    padx = (s - (x % s)) % s
    if padz or pady or padx:
        vol = X.pad(vol, ((0,padz),(0,pady),(0,padx)), mode='constant')
    return vol

def boxcount_3d_counts(binary_vol, scales):
    """
    binary_vol: 3D array (0/1)
    scales: iterable of box sizes (int)
    returns: dict {s: n_boxes_nonempty}
    """
    X = xp()
    out = {}
    v = binary_vol.astype(X.uint8, copy=False)
    for s in scales:
        vv = _pad_to_multiple(v, s)
        Z,Y,Xdim = vv.shape
        # reshape into blocks of size s x s x s
        vv = vv.reshape(Z//s, s, Y//s, s, Xdim//s, s)
        # reduce: any voxel >0 in each block
        nonempty = vv.max(axis=(1,3,5))  # boolean map of non-empty blocks
        out[int(s)] = int(nonempty.sum())
    return out
