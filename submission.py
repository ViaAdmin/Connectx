# =============================================================================
# ConnectX 提交 Agent（自包含单文件，无 numba 依赖）
#
# Kaggle 对弈环境为 NumPy 2.x，与其 numba 版本不兼容（import numba 即报错），
# 因此本文件只依赖 NumPy：NNUE 价值网络（权重以 base64 npz 内嵌）做叶评估，
# 49 位位棋盘 + 纯 Python alpha-beta 搜索：置换表 + killer/history 动态排序
# + LMR + 对称局面着法剪枝 + 迭代加深 + 墙钟中止 + 证明值/评估值分离。
#
# 网络结构: 84 -> 256 (relu) -> 64 (relu) -> 32 (relu) -> 1 (tanh)
# 特征: 42 个己方棋子格 ++ 42 个对方棋子格（observation.board 第 0 行为顶部）。
#
# 注意: Kaggle 以文件中最后定义的函数作为 agent，因此 `act` 必须定义在最后。
# =============================================================================
import base64
import io
import math
import os
import time

# 小矩阵运算必须单线程 BLAS：多线程 OpenBLAS/MKL 对 256x64 级矩阵乘的
# 线程同步开销可达 10 倍（本文件评估函数因此被拖慢 ~12 倍）
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

# 若 numpy 已被宿主框架提前 import（环境变量来不及生效），用 threadpoolctl 兜底
try:
    from threadpoolctl import threadpool_limits

    threadpool_limits(1, "blas")
except Exception:
    pass

# =============================================================================
# 【权重未包含】本仓库不提供训练好的模型权重。
# 请用 selfplay_training.ipynb 训练出自己的 nnue_model.pth 后，按以下布局打包嵌入：
#   npz 包含 9 个数组: w1(84,256) b1(256,) w2(256,64) b2(64,)
#                      w3(64,32) b3(32,) w4(32,1) b4(1,)
#   （即 state_dict 中 fc1~fc4 的 weight 转置 + bias）
#   然后: base64.b64encode(npz_bytes) 的结果替换下方字符串。
# =============================================================================
_WEIGHTS_B64 = ""  # <-- 在此粘贴你自己的权重（npz 的 base64 编码）


def _load_params():
    if not _WEIGHTS_B64:
        raise RuntimeError("_WEIGHTS_B64 为空：请先训练模型并嵌入自己的权重（见文件头部说明）")
    z = np.load(io.BytesIO(base64.b64decode(_WEIGHTS_B64)))
    return {k: np.ascontiguousarray(z[k], dtype=np.float32) for k in z.files}


_P = _load_params()
W1, B1 = _P["w1"], _P["b1"]       # (84, 256), (256,)
W2, B2 = _P["w2"], _P["b2"]       # (256, 64), (64,)
W3, B3 = _P["w3"], _P["b3"]       # (64, 32), (32,)
W4 = _P["w4"][:, 0]               # (32,)
B4 = float(_P["b4"][0])

# 位棋盘编码: bit = col*7 + row (row 0 = 底部)，每列 7 bit（6 格 + 1 哨兵位）
# 特征索引与训练一致: feat = (5-row)*7 + col
ORDER = (3, 2, 4, 1, 5, 0, 6)
# 左右对称局面下的着法顺序：列 c 与 6-c 等价，只需搜索 0~3 列（分支 7 -> 4）
SYM_ORDER = (3, 2, 1, 0)

# 证明值编码：|score| >= WIN_SCORE 表示已证明的终局胜/负（100 + 剩余depth，
# 剩余深度越大 = 离根越近 = 越快兑现，argmax 自动偏好速胜/最长抵抗）。
# NNUE 评估经 tanh 在 float32 下会饱和到恰好 ±1.0，必须与证明值严格分离，
# 否则饱和评估值会被误判为已证明结果、导致迭代加深提前停止。
WIN_SCORE = 100.0

FEAT_OF_BIT = [0] * 49
for _c in range(7):
    for _r in range(6):
        FEAT_OF_BIT[_c * 7 + _r] = (5 - _r) * 7 + _c

# 搜索全局状态（用列表避免 global 重绑定开销）
_nodes = [0]
_deadline = [0.0]

# ---------------- 搜索记忆（跨迭代/跨着手复用） ----------------
# 置换表: key = (bb_me << 49) | bb_opp -> [depth, flag, value, best_col]
#   flag: 0=EXACT 1=LOWER(fail-high 下界) 2=UPPER(fail-low 上界)
#   已证明胜/负的条目 depth 记为 99（证明结论与搜索深度无关，永久可用）；
#   存入 TT 的证明值统一钳到 ±WIN_SCORE（丢弃速度奖励，避免与剩余深度耦合失真）。
_TT = {}
_TT_MAX = 1_200_000  # 条目上限，超出整体清空（清空只损失速度，不影响正确性）
# killer: 每个 ply（用 npieces 索引，同 ply 恒为同一方走棋）记两个最近引发剪枝的列
_KILL = [[-1, -1] for _ in range(43)]
# history: 剪枝着法 += depth*depth，用于给非杀手着法动态排序
_HIST = [0.0] * 7
_last_state = [0, 99]  # [my_mark, npieces] 用于检测新对局并清空记忆

# 预分配缓冲，避免搜索过程中反复分配内存
_ACC_BUF = np.empty((43, 256), dtype=np.float32)  # 每层递归用独立一行
_ABUF = np.empty(256, dtype=np.float32)
_HBUF1 = np.empty(64, dtype=np.float32)
_HBUF2 = np.empty(32, dtype=np.float32)


def _has_won(bb):
    # 竖 / 横 / 两种斜线
    m = bb & (bb >> 1)
    if m & (m >> 2):
        return True
    m = bb & (bb >> 7)
    if m & (m >> 14):
        return True
    m = bb & (bb >> 6)
    if m & (m >> 12):
        return True
    m = bb & (bb >> 8)
    if m & (m >> 16):
        return True
    return False


def _mirror(bb):
    # 左右镜像：交换 7 bit 一组的列（col c <-> col 6-c）
    return (((bb & 0x7F) << 42) | ((bb & 0x3F80) << 28) | ((bb & 0x1FC000) << 14)
            | (bb & 0xFE00000) | ((bb >> 14) & 0x1FC000) | ((bb >> 28) & 0x3F80)
            | ((bb >> 42) & 0x7F))


def _nnue_eval(acc):
    # relu(acc)@W2+B2 -> relu -> @W3+B3 -> relu -> @W4 -> tanh（全部原地写预分配缓冲）
    np.maximum(acc, 0, out=_ABUF)
    np.dot(_ABUF, W2, out=_HBUF1)
    np.add(_HBUF1, B2, out=_HBUF1)
    np.maximum(_HBUF1, 0, out=_HBUF1)
    np.dot(_HBUF1, W3, out=_HBUF2)
    np.add(_HBUF2, B3, out=_HBUF2)
    np.maximum(_HBUF2, 0, out=_HBUF2)
    return math.tanh(float(np.dot(_HBUF2, W4)) + B4)


class _Abort(Exception):
    """搜索超时中止（抛异常一次性展开递归栈）。"""


def _tt_store(key, old, depth, flag, best, best_col):
    # 证明值钳到 ±WIN_SCORE 存入；证明性结论（下界≥胜 / 上界≤负 / 精确）深度记 99 永久有效
    if best >= WIN_SCORE:
        sval = WIN_SCORE
        sdepth = 99 if flag != 2 else depth
    elif best <= -WIN_SCORE:
        sval = -WIN_SCORE
        sdepth = 99 if flag != 1 else depth
    else:
        sval = best
        sdepth = depth
    if old is None or sdepth >= old[0]:
        _TT[key] = (sdepth, flag, sval, best_col)


def _alphabeta(bb_me, bb_opp, heights, npieces, depth, alpha, beta, maximizing, acc, sym):
    _nodes[0] += 1
    if (_nodes[0] & 255) == 0 and time.perf_counter() > _deadline[0]:
        raise _Abort()

    # ---- 置换表探测：足够深的条目直接返回或收窄窗口，并取出排序用的最优列 ----
    key = (bb_me << 49) | bb_opp
    e = _TT.get(key)
    tt_col = -1
    if e is not None:
        e_depth, e_flag, e_val, tt_col = e
        if e_depth >= depth:
            if e_flag == 0:
                return e_val
            if e_flag == 1:
                if e_val >= beta:
                    return e_val
                if e_val > alpha:
                    alpha = e_val
            else:
                if e_val <= alpha:
                    return e_val
                if e_val < beta:
                    beta = e_val
            if alpha >= beta:
                return e_val
    alpha0, beta0 = alpha, beta
    order = SYM_ORDER if sym else ORDER

    if maximizing:
        # 即胜预检：行棋方有立即获胜着法 -> 证明值 100+depth（越浅越大 = 偏好速胜）
        for c in order:
            h = heights[c]
            if h < 7 * c + 6 and _has_won(bb_me | (1 << h)):
                _TT[key] = (99, 0, WIN_SCORE, c)
                return WIN_SCORE + depth
        avail = [c for c in order if heights[c] < 7 * c + 6]
        # 动态排序（仅深节点，浅节点排序收益 < 排序开销）：TT 着法 > killer > history
        if depth > 2 and len(avail) > 1:
            k = _KILL[npieces]
            k0, k1 = k[0], k[1]
            avail.sort(key=lambda c: -(
                (1e9 if c == tt_col else 0.0) + (2e6 if c == k0 else 0.0)
                + (1e6 if c == k1 else 0.0) + _HIST[c]))
        best = -1e30
        best_col = avail[0]
        mi = 0
        for c in avail:
            h = heights[c]
            bb2 = bb_me | (1 << h)
            heights[c] = h + 1
            csym = sym and c == 3  # 走中列保持对称，走其他列必破坏对称
            if npieces + 1 == 42:
                score = 0.0
            else:
                nacc = _ACC_BUF[npieces + 1]
                np.add(acc, W1[FEAT_OF_BIT[h]], out=nacc)
                if depth == 1:
                    score = _nnue_eval(nacc)
                elif depth >= 3 and mi >= 3:
                    # LMR：排序靠后的着法先用 depth-2 浅搜；
                    # 超出 alpha（且非已证明结果）才用全深度重搜验证
                    score = _alphabeta(bb2, bb_opp, heights, npieces + 1, depth - 2,
                                       alpha, beta, False, nacc, csym)
                    if alpha < score < WIN_SCORE:
                        score = _alphabeta(bb2, bb_opp, heights, npieces + 1, depth - 1,
                                           alpha, beta, False, nacc, csym)
                else:
                    score = _alphabeta(bb2, bb_opp, heights, npieces + 1, depth - 1,
                                       alpha, beta, False, nacc, csym)
            heights[c] = h
            mi += 1
            if score > best:
                best = score
                best_col = c
                if best > alpha:
                    alpha = best
                    if alpha >= beta:
                        # 剪枝成功：记入 killer / history 供后续节点排序
                        k = _KILL[npieces]
                        if k[0] != c:
                            k[1] = k[0]
                            k[0] = c
                        _HIST[c] += depth * depth
                        break
        flag = 1 if best >= beta0 else (2 if best <= alpha0 else 0)
        _tt_store(key, e, depth, flag, best, best_col)
        return best
    else:
        for c in order:
            h = heights[c]
            if h < 7 * c + 6 and _has_won(bb_opp | (1 << h)):
                _TT[key] = (99, 0, -WIN_SCORE, c)
                return -(WIN_SCORE + depth)
        avail = [c for c in order if heights[c] < 7 * c + 6]
        if depth > 2 and len(avail) > 1:
            k = _KILL[npieces]
            k0, k1 = k[0], k[1]
            avail.sort(key=lambda c: -(
                (1e9 if c == tt_col else 0.0) + (2e6 if c == k0 else 0.0)
                + (1e6 if c == k1 else 0.0) + _HIST[c]))
        best = 1e30
        best_col = avail[0]
        mi = 0
        for c in avail:
            h = heights[c]
            bb2 = bb_opp | (1 << h)
            heights[c] = h + 1
            csym = sym and c == 3
            if npieces + 1 == 42:
                score = 0.0
            else:
                nacc = _ACC_BUF[npieces + 1]
                np.add(acc, W1[FEAT_OF_BIT[h] + 42], out=nacc)
                if depth == 1:
                    score = _nnue_eval(nacc)
                elif depth >= 3 and mi >= 3:
                    score = _alphabeta(bb_me, bb2, heights, npieces + 1, depth - 2,
                                       alpha, beta, True, nacc, csym)
                    if -WIN_SCORE < score < beta:
                        score = _alphabeta(bb_me, bb2, heights, npieces + 1, depth - 1,
                                           alpha, beta, True, nacc, csym)
                else:
                    score = _alphabeta(bb_me, bb2, heights, npieces + 1, depth - 1,
                                       alpha, beta, True, nacc, csym)
            heights[c] = h
            mi += 1
            if score < best:
                best = score
                best_col = c
                if best < beta:
                    beta = best
                    if alpha >= beta:
                        k = _KILL[npieces]
                        if k[0] != c:
                            k[1] = k[0]
                            k[0] = c
                        _HIST[c] += depth * depth
                        break
        flag = 1 if best >= beta0 else (2 if best <= alpha0 else 0)
        _tt_store(key, e, depth, flag, best, best_col)
        return best


def _search_root(board, my_piece, depth, prev_best):
    # 构建位棋盘与首层累加器
    bb_me = bb_opp = 0
    npieces = 0
    cnt = [0] * 7
    for f in range(42):
        v = board[f]
        if v:
            rt, c = divmod(f, 7)
            bpos = c * 7 + (5 - rt)
            cnt[c] += 1
            npieces += 1
            if v == my_piece:
                bb_me |= 1 << bpos
            else:
                bb_opp |= 1 << bpos
    heights = [7 * c + cnt[c] for c in range(7)]
    acc = B1.copy()
    for bpos in range(49):
        if (bb_me >> bpos) & 1:
            acc += W1[FEAT_OF_BIT[bpos]]
        elif (bb_opp >> bpos) & 1:
            acc += W1[FEAT_OF_BIT[bpos] + 42]

    # 对称检测：左右镜像局面分值相同，对称时只搜 0~3 列（分支 7 -> 4）
    sym = bb_me == _mirror(bb_me) and bb_opp == _mirror(bb_opp)

    # 根节点着法顺序：上一层最佳优先，其次 TT 记录的最优列，其余按 history 动态排序
    cols = [c for c in (SYM_ORDER if sym else ORDER) if heights[c] < 7 * c + 6]
    if not cols:
        return -1, 0.0
    key = (bb_me << 49) | bb_opp
    e = _TT.get(key)
    tt_col = e[3] if e is not None else -1
    cols.sort(key=lambda c: -(
        (1e12 if c == prev_best else 0.0) + (1e9 if c == tt_col else 0.0) + _HIST[c]))

    best_col = cols[0]
    best = -1e30
    alpha = -1e30
    for c in cols:
        h = heights[c]
        bb2 = bb_me | (1 << h)
        heights[c] = h + 1
        if _has_won(bb2):
            score = WIN_SCORE + depth
        elif npieces + 1 == 42:
            score = 0.0
        else:
            nacc = _ACC_BUF[npieces + 1]
            np.add(acc, W1[FEAT_OF_BIT[h]], out=nacc)
            if depth == 1:
                score = _nnue_eval(nacc)
            else:
                score = _alphabeta(bb2, bb_opp, heights, npieces + 1, depth - 1,
                                   alpha, 1e30, False, nacc, sym and c == 3)
        heights[c] = h
        if score > best:
            best = score
            best_col = c
        if best > alpha:
            alpha = best
    _tt_store(key, e, depth, 0, best, best_col)  # 根为全窗口搜索，值精确
    return best_col, best


def _search_iterative(board, my_piece, deadline):
    """迭代加深：从 depth 1 逐层加深，超时则返回上一完整层结果。"""
    _deadline[0] = deadline
    best_col = -1
    max_depth = 42 - sum(1 for v in board if v != 0)  # 剩余空格即最大深度
    for depth in range(1, max_depth + 1):
        if time.perf_counter() > deadline:
            break
        _nodes[0] = 0
        try:
            col, val = _search_root(board, my_piece, depth, best_col)
        except _Abort:
            break
        if col == -1:
            break
        best_col = col
        if abs(val) >= WIN_SCORE:
            break  # 已证明胜/负（非 tanh 饱和的 ±1.0 评估值），加深不改变结果
    if best_col == -1:
        for c in ORDER:
            if board[c] == 0:
                return c
        return 0
    return best_col


def act(observation, configuration):
    try:
        board = [int(v) for v in observation.board]
        my_piece = int(observation.mark)
        npieces = sum(1 for v in board if v)
        # 新对局检测（先后手变化或棋子数回退）：清空跨着手复用的搜索记忆
        if my_piece != _last_state[0] or npieces < _last_state[1]:
            _TT.clear()
            for k in _KILL:
                k[0] = k[1] = -1
            for i in range(7):
                _HIST[i] = 0.0
        _last_state[0] = my_piece
        _last_state[1] = npieces
        if len(_TT) > _TT_MAX:
            _TT.clear()
        for i in range(7):
            _HIST[i] *= 0.5  # 衰减：让排序偏向近期有效的着法
        timeout = float(getattr(configuration, "actTimeout", 2.0) or 2.0)
        deadline = time.perf_counter() + min(timeout * 0.85, 1.7)  # 留 15% 余量防超时
        return int(_search_iterative(board, my_piece, deadline))
    except Exception:
        # 任何异常都不能抛出（否则判负）：退回第一个合法列
        try:
            for c in (3, 2, 4, 1, 5, 0, 6):
                if observation.board[c] == 0:
                    return int(c)
        except Exception:
            pass
        return 0
