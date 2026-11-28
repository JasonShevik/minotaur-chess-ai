/*
 * fast_reach.c: the compiled search behind piece_reach.py (legal threats and piece reach, Chess960 rules).
 *
 * piece_reach.py's own searches are the reference: this file must give exactly the same arrays, and
 * tests/test_fast_reach.py checks that it does. Build it with
 *
 *     python build_fast_reach.py
 *
 * which makes fast_reach.dll (Windows) or fast_reach.so next to this file; piece_reach.py uses it when it is
 * there. By hand:  gcc -O3 -march=native -funroll-loops -shared -static-libgcc fast_reach.c -o fast_reach.dll
 * (add -fPIC for a .so). -march=native: the library is for the machine that builds it.
 *
 * What is counted is described at the top of piece_reach.py. How it is counted here:
 *
 *   * The search walks every sequence of legal moves depth first. Each node knows, for every square, which
 *     piece of the root position stands there now, so what it finds goes straight into that piece's row of
 *     the answer; nothing is handed back up the tree.
 *   * A position's threats are not listed move by move. For both sides at once, one pass works out every
 *     slider's attacks (used for that piece's own moves and for the other king's safety), then each piece's
 *     legal squares as one 64-bit mask.
 *   * The positions one ply above the last (whose children are only looked at, not searched) do most of the
 *     work, and a child is not worked out whole there: one move leaves most pieces threatening exactly what
 *     they threatened before it, so only the pieces the move can reach are worked out again (the piece that
 *     moved, those that look at a square that changed hands, those whose pin, king's safety, castling or en
 *     passant the move touches), and the rest are counted as unchanged. A side in check is worked out whole.
 *   * From the fourth ply on, many of those positions are reached by more than one move order, so what such a
 *     position adds (its own threats and the sum of its children's) is kept in a table under the position
 *     itself, numbered by its own pieces, and added again from there when the position comes back.
 *   * The whole answer for the last few root positions is kept too. A labelled sample is a position and the
 *     positions after it in a line of play, searched last one first, so each search finds the next position's
 *     answer ready and skips that branch.
 *
 * A position this file does not handle (it cannot have arisen in a game: no king, the side not to move in
 * check, pawns on the back ranks, an en passant square with no pawn to take, more than 16 pieces a side) is
 * refused with a non-zero result, and piece_reach.py falls back to its own search.
 *
 * Squares are python-chess's: a1 = 0, b1 = 1, ..., h8 = 63.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

enum { WHITE = 0, BLACK = 1 };
enum { PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5 };

#define F_CASTLE 1                  /* a castling move is written king's square -> its rook's square */
#define F_EP 2
#define MAX_MOVES 512
#define MAX_PLIES 12
#define CAPTURED 64                 /* reach: the column for "no longer on the board" */

#define NOT_A 0xFEFEFEFEFEFEFEFEULL
#define NOT_H 0x7F7F7F7F7F7F7F7FULL
#define RANK_2 0x000000000000FF00ULL
#define RANK_7 0x00FF000000000000ULL
#define BACK_RANKS 0xFF000000000000FFULL

#define BIT(sq) (1ULL << (sq))
#define POP(b) ((b) &= (b) - 1)

static inline int lsb(uint64_t x) { return __builtin_ctzll(x); }

typedef struct {
    uint64_t bb[2][6];              /* [colour][piece type] */
    uint64_t occ[2];
    int8_t king[2];                 /* the kings' squares */
    int8_t turn;
    int8_t ep_sq;                   /* the square a pawn just jumped over, or -1 */
    int8_t castle_rook[4];          /* [colour * 2 + side]: the rook's square or -1; side 0 = h-side (king to g), 1 = a-side (king to c) */
    int8_t piece_at[64];            /* piece type, or -1. Last: a position that is only looked at goes without it (make_leaf) */
} Pos;

#define POS_HEAD offsetof(Pos, piece_at)

typedef struct {
    uint8_t from, to, promo, flags; /* promo: 0 or KNIGHT..QUEEN */
} Move;

/* One side's legal moves in a position, as a mask of squares for each piece that has any. */
typedef struct {
    int n;
    uint8_t from[16];
    uint64_t to[16];                /* squares the piece can legally move to (castling aside) */
    uint64_t extra[16];             /* squares it would castle to: a threat, not a move of its own */
    int8_t castle[2];               /* legal castling, h-side then a-side: the rook's square, or -1 */
} Side;

/* ---------- ---------- ----------
 *   Tables
 */

static uint64_t KNIGHT_ATK[64], KING_ATK[64], PAWN_ATK[2][64];
static uint64_t FILE_MASK[64], DIAG_MASK[64], ANTI_MASK[64];    /* the whole line through a square, the square included */
static uint64_t SNIPE_B[64], SNIPE_R[64];                       /* a bishop's / rook's squares on an empty board */
static uint64_t BETWEEN_BB[64][64], LINE_BB[64][64];
static uint64_t RANK_ATK[64][64];                               /* [square][the 6 inner squares of its rank] */
static int TABLES_READY = 0;

static void init_tables(void) {
    if (TABLES_READY) return;
    for (int sq = 0; sq < 64; sq++) {
        int r = sq >> 3, f = sq & 7;
        uint64_t d = 0, a = 0, kn = 0, kg = 0;
        FILE_MASK[sq] = 0x0101010101010101ULL << f;
        for (int s = 0; s < 64; s++) {
            int sr = s >> 3, sf = s & 7, dr = sr - r, df = sf - f;
            if (dr == df) d |= BIT(s);
            if (dr == -df) a |= BIT(s);
            if ((abs(dr) == 2 && abs(df) == 1) || (abs(dr) == 1 && abs(df) == 2)) kn |= BIT(s);
            if (s != sq && abs(dr) <= 1 && abs(df) <= 1) kg |= BIT(s);
        }
        DIAG_MASK[sq] = d;
        ANTI_MASK[sq] = a;
        KNIGHT_ATK[sq] = kn;
        KING_ATK[sq] = kg;
        SNIPE_B[sq] = (d | a) & ~BIT(sq);
        SNIPE_R[sq] = ((0xFFULL << (r * 8)) | FILE_MASK[sq]) & ~BIT(sq);
        PAWN_ATK[WHITE][sq] = r < 7 ? ((f > 0 ? BIT(sq + 7) : 0) | (f < 7 ? BIT(sq + 9) : 0)) : 0;
        PAWN_ATK[BLACK][sq] = r > 0 ? ((f > 0 ? BIT(sq - 9) : 0) | (f < 7 ? BIT(sq - 7) : 0)) : 0;
        for (int inner = 0; inner < 64; inner++) {
            uint64_t atk = 0;
            for (int ff = f + 1; ff < 8; ff++) {
                atk |= BIT(r * 8 + ff);
                if (ff >= 1 && ff <= 6 && ((inner >> (ff - 1)) & 1)) break;
            }
            for (int ff = f - 1; ff >= 0; ff--) {
                atk |= BIT(r * 8 + ff);
                if (ff >= 1 && ff <= 6 && ((inner >> (ff - 1)) & 1)) break;
            }
            RANK_ATK[sq][inner] = atk;
        }
    }
    for (int s1 = 0; s1 < 64; s1++) {
        int r1 = s1 >> 3, f1 = s1 & 7;
        for (int s2 = 0; s2 < 64; s2++) {
            int r2 = s2 >> 3, f2 = s2 & 7;
            int dr = (r2 > r1) - (r2 < r1), df = (f2 > f1) - (f2 < f1);
            uint64_t bet = 0, lin = 0;
            if (s1 != s2 && (r1 == r2 || f1 == f2 || abs(r1 - r2) == abs(f1 - f2))) {
                for (int rr = r1 + dr, ff = f1 + df; rr != r2 || ff != f2; rr += dr, ff += df) bet |= BIT(rr * 8 + ff);
                for (int k = -7; k <= 7; k++) {
                    int lr = r1 + k * dr, lf = f1 + k * df;
                    if (lr >= 0 && lr < 8 && lf >= 0 && lf < 8) lin |= BIT(lr * 8 + lf);
                }
            }
            BETWEEN_BB[s1][s2] = bet;       /* the squares strictly between, when the two share a line */
            LINE_BB[s1][s2] = lin;          /* the whole line through both */
        }
    }
    TABLES_READY = 1;
}

/* A slider's squares along one file or diagonal ("hyperbola quintessence"). Right whether or not the slider's
   own square is in occ. Ranks need the table: reversing the bytes does not reverse a rank. */
static inline uint64_t line_attacks(int sq, uint64_t occ, uint64_t mask) {
    uint64_t o = occ & mask, bit = BIT(sq);
    uint64_t forward = o - 2 * bit;
    uint64_t reverse = __builtin_bswap64(__builtin_bswap64(o) - 2 * __builtin_bswap64(bit));
    return (forward ^ reverse) & mask;
}

static inline uint64_t rook_attacks(int sq, uint64_t occ) {
    return RANK_ATK[sq][(occ >> ((sq & 56) + 1)) & 63] | line_attacks(sq, occ, FILE_MASK[sq]);
}

static inline uint64_t bishop_attacks(int sq, uint64_t occ) {
    return line_attacks(sq, occ, DIAG_MASK[sq]) | line_attacks(sq, occ, ANTI_MASK[sq]);
}

/* Is the square attacked by that colour, with these squares occupied? */
static int attacked(const Pos *p, int sq, int by, uint64_t occ) {
    if (PAWN_ATK[by ^ 1][sq] & p->bb[by][PAWN]) return 1;
    if (KNIGHT_ATK[sq] & p->bb[by][KNIGHT]) return 1;
    if (KING_ATK[sq] & p->bb[by][KING]) return 1;
    uint64_t diag = p->bb[by][BISHOP] | p->bb[by][QUEEN];
    if (diag && (bishop_attacks(sq, occ) & diag)) return 1;
    uint64_t orth = p->bb[by][ROOK] | p->bb[by][QUEEN];
    return orth && (rook_attacks(sq, occ) & orth);
}

/* ---------- ---------- ----------
 *   Positions
 */

static inline void put(Pos *p, int sq, int c, int pt) {
    p->bb[c][pt] |= BIT(sq);
    p->occ[c] |= BIT(sq);
    p->piece_at[sq] = (int8_t)pt;
}

static inline void lift(Pos *p, int sq, int c, int pt) {
    p->bb[c][pt] &= ~BIT(sq);
    p->occ[c] &= ~BIT(sq);
    p->piece_at[sq] = -1;
}

static inline void castle_squares(int king_from, int rook_from, int *king_to, int *rook_to) {
    int rank = king_from & 56;
    *king_to = rank | (rook_from < king_from ? 2 : 6);
    *rook_to = rank | (rook_from < king_from ? 3 : 5);
}

static void make_move(const Pos *src, Move m, Pos *dst) {
    *dst = *src;
    int c = src->turn, e = c ^ 1, f = m.from, t = m.to;
    dst->turn = (int8_t)e;
    dst->ep_sq = -1;
    if (m.flags & F_CASTLE) {
        int kt, rt;
        castle_squares(f, t, &kt, &rt);
        lift(dst, f, c, KING);
        lift(dst, t, c, ROOK);
        put(dst, kt, c, KING);
        put(dst, rt, c, ROOK);
        dst->king[c] = (int8_t)kt;
        dst->castle_rook[c * 2] = dst->castle_rook[c * 2 + 1] = -1;
        return;
    }
    int pt = src->piece_at[f];
    if (m.flags & F_EP) {
        lift(dst, t ^ 8, e, PAWN);
    } else if (src->piece_at[t] >= 0) {
        lift(dst, t, e, src->piece_at[t]);
        if (t == dst->castle_rook[e * 2]) dst->castle_rook[e * 2] = -1;
        if (t == dst->castle_rook[e * 2 + 1]) dst->castle_rook[e * 2 + 1] = -1;
    }
    lift(dst, f, c, pt);
    put(dst, t, c, m.promo ? m.promo : pt);
    if (pt == KING) {
        dst->king[c] = (int8_t)t;
        dst->castle_rook[c * 2] = dst->castle_rook[c * 2 + 1] = -1;
    } else if (pt == ROOK) {
        if (f == dst->castle_rook[c * 2]) dst->castle_rook[c * 2] = -1;
        if (f == dst->castle_rook[c * 2 + 1]) dst->castle_rook[c * 2 + 1] = -1;
    } else if (pt == PAWN && abs(t - f) == 16) {
        dst->ep_sq = (int8_t)((f + t) >> 1);
    }
}

/* The position after the move, for looking at only: everything but piece_at, which is left unset. (What works out
   a position's moves as masks reads the bitboards alone; list_moves and make_move need piece_at.) */
static inline void make_leaf(const Pos *src, Move m, Pos *dst) {
    memcpy(dst, src, POS_HEAD);
    const int c = src->turn, e = c ^ 1, f = m.from, t = m.to;
    const uint64_t fb = BIT(f), tb = BIT(t);
    dst->turn = (int8_t)e;
    dst->ep_sq = -1;
    if (m.flags & F_CASTLE) {
        int kt, rt;
        castle_squares(f, t, &kt, &rt);
        dst->bb[c][KING] = BIT(kt);
        dst->bb[c][ROOK] = (dst->bb[c][ROOK] & ~tb) | BIT(rt);
        dst->occ[c] = (dst->occ[c] & ~(fb | tb)) | BIT(kt) | BIT(rt);
        dst->king[c] = (int8_t)kt;
        dst->castle_rook[c * 2] = dst->castle_rook[c * 2 + 1] = -1;
        return;
    }
    const int pt = src->piece_at[f];
    if (m.flags & F_EP) {
        const uint64_t taken = BIT(t ^ 8);
        dst->bb[e][PAWN] &= ~taken;
        dst->occ[e] &= ~taken;
    } else if (src->piece_at[t] >= 0) {
        dst->bb[e][src->piece_at[t]] &= ~tb;
        dst->occ[e] &= ~tb;
        if (t == dst->castle_rook[e * 2]) dst->castle_rook[e * 2] = -1;
        if (t == dst->castle_rook[e * 2 + 1]) dst->castle_rook[e * 2 + 1] = -1;
    }
    dst->bb[c][pt] &= ~fb;
    dst->bb[c][m.promo ? m.promo : pt] |= tb;
    dst->occ[c] = (dst->occ[c] & ~fb) | tb;
    if (pt == KING) {
        dst->king[c] = (int8_t)t;
        dst->castle_rook[c * 2] = dst->castle_rook[c * 2 + 1] = -1;
    } else if (pt == ROOK) {
        if (f == dst->castle_rook[c * 2]) dst->castle_rook[c * 2] = -1;
        if (f == dst->castle_rook[c * 2 + 1]) dst->castle_rook[c * 2 + 1] = -1;
    } else if (pt == PAWN && abs(t - f) == 16) {
        dst->ep_sq = (int8_t)((f + t) >> 1);
    }
}

/* A FEN (standard, X-FEN or Shredder castling letters; move counters optional) -> Pos. 0 when the position is
   one this file handles, otherwise non-zero (see the top of the file). */
static int parse_fen(const char *fen, Pos *pos) {
    memset(pos, 0, sizeof(*pos));
    memset(pos->piece_at, -1, sizeof(pos->piece_at));
    pos->ep_sq = -1;
    memset(pos->castle_rook, -1, sizeof(pos->castle_rook));

    const char *p = fen;
    while (*p == ' ') p++;
    int rank = 7, file = 0;
    for (; *p && *p != ' '; p++) {
        char ch = *p;
        if (ch == '/') {
            if (file != 8 || rank == 0) return -1;
            rank--;
            file = 0;
        } else if (ch >= '1' && ch <= '8') {
            file += ch - '0';
            if (file > 8) return -1;
        } else {
            int c = (ch >= 'A' && ch <= 'Z') ? WHITE : BLACK, pt;
            switch (ch | 32) {
                case 'p': pt = PAWN; break;
                case 'n': pt = KNIGHT; break;
                case 'b': pt = BISHOP; break;
                case 'r': pt = ROOK; break;
                case 'q': pt = QUEEN; break;
                case 'k': pt = KING; break;
                default: return -1;
            }
            if (file > 7) return -1;
            put(pos, rank * 8 + file, c, pt);
            file++;
        }
    }
    if (rank != 0 || file != 8) return -1;
    for (int c = 0; c < 2; c++) {
        if (__builtin_popcountll(pos->bb[c][KING]) != 1 || __builtin_popcountll(pos->occ[c]) > 16) return -1;
        if (pos->bb[c][PAWN] & BACK_RANKS) return -1;
        pos->king[c] = (int8_t)lsb(pos->bb[c][KING]);
    }

    while (*p == ' ') p++;
    if (*p == 'w') pos->turn = WHITE;
    else if (*p == 'b') pos->turn = BLACK;
    else return -1;
    p++;
    while (*p == ' ') p++;

    for (; *p && *p != ' '; p++) {              /* castling rights, read as python-chess reads them */
        char ch = *p;
        if (ch == '-') continue;
        int c = (ch >= 'A' && ch <= 'Z') ? WHITE : BLACK, base = c == WHITE ? 0 : 56;
        int k = pos->king[c], kf = k & 7;
        char up = (char)(ch & ~32);
        if ((k & 56) != base) continue;         /* no castling with the king off its back rank */
        uint64_t rooks = pos->bb[c][ROOK] & (0xFFULL << base);
        if (up == 'K') {                        /* the outermost rook on the h-side */
            for (int f = 7; f > kf; f--)
                if (rooks >> (base + f) & 1) { pos->castle_rook[c * 2] = (int8_t)(base + f); break; }
        } else if (up == 'Q') {                 /* the outermost rook on the a-side */
            for (int f = 0; f < kf; f++)
                if (rooks >> (base + f) & 1) { pos->castle_rook[c * 2 + 1] = (int8_t)(base + f); break; }
        } else if (up >= 'A' && up <= 'H') {
            int f = up - 'A';
            if (f != kf && (rooks >> (base + f) & 1)) pos->castle_rook[c * 2 + (f > kf ? 0 : 1)] = (int8_t)(base + f);
        } else {
            return -1;
        }
    }
    while (*p == ' ') p++;

    if (*p && *p != '-') {                      /* en passant: only a square a pawn really just jumped over */
        int f = p[0] - 'a', r = p[1] - '1';
        if (f < 0 || f > 7 || r != (pos->turn == WHITE ? 5 : 2)) return -1;
        int ep = r * 8 + f, pawn = ep ^ 8, origin = pos->turn == WHITE ? ep + 8 : ep - 8;
        uint64_t occ = pos->occ[0] | pos->occ[1];
        if ((occ >> ep & 1) || (occ >> origin & 1) || !(pos->bb[pos->turn ^ 1][PAWN] >> pawn & 1)) return -1;
        pos->ep_sq = (int8_t)ep;
    }
    /* The side that is not to move must not be in check: its king could be taken, which no legal play allows */
    if (attacked(pos, pos->king[pos->turn ^ 1], pos->turn, pos->occ[0] | pos->occ[1])) return -1;
    return 0;
}

/* ---------- ---------- ----------
 *   Legal moves, as masks
 */

static inline int side_entry(Side *s, int n, int sq) {
    for (int i = 0; i < n; i++)
        if (s->from[i] == sq) return i;
    s->from[n] = (uint8_t)sq;
    s->to[n] = 0;
    s->extra[n] = 0;
    return n;
}

/* The legal moves of colour c as if it were its turn. ep_active: c really is to move, so the en passant
   square is its to use. slider[sq]: every slider's attacks in this position; att_e: every square the other
   colour attacks (its own pieces' squares included: they are defended). */
static void gen_side(const Pos *p, int c, int ep_active, uint64_t occ, const uint64_t *slider, uint64_t att_e, Side *s) {
    const int e = c ^ 1, K = p->king[c];
    const uint64_t own = p->occ[c], opp = p->occ[e];
    const uint64_t e_diag = p->bb[e][BISHOP] | p->bb[e][QUEEN], e_orth = p->bb[e][ROOK] | p->bb[e][QUEEN];
    uint64_t danger = att_e, checkers = 0, check_mask = ~0ULL, t, b;
    int n = 0;
    s->castle[0] = s->castle[1] = -1;

    if (danger >> K & 1) {
        uint64_t sl = (bishop_attacks(K, occ) & e_diag) | (rook_attacks(K, occ) & e_orth);
        checkers = sl | (PAWN_ATK[c][K] & p->bb[e][PAWN]) | (KNIGHT_ATK[K] & p->bb[e][KNIGHT]);
        /* A checking slider also covers the square behind the king, which the king hides from it now */
        for (; sl; POP(sl)) danger |= LINE_BB[K][lsb(sl)] & KING_ATK[K] & ~BIT(lsb(sl));
        if (checkers)       /* one check: take the checker or stand in its way; two: only the king moves */
            check_mask = (checkers & (checkers - 1)) ? 0 : (BETWEEN_BB[K][lsb(checkers)] | checkers);
    }

    t = KING_ATK[K] & ~own & ~danger;
    if (t) { s->from[n] = (uint8_t)K; s->to[n] = t; s->extra[n] = 0; n++; }

    if (check_mask) {
        uint64_t pinned = 0, ray[64];
        for (b = (SNIPE_B[K] & e_diag) | (SNIPE_R[K] & e_orth); b; POP(b)) {
            int sn = lsb(b);
            uint64_t between = BETWEEN_BB[K][sn] & occ;
            if (between && !(between & (between - 1)) && (between & own)) {
                pinned |= between;
                ray[lsb(between)] = LINE_BB[K][sn];
            }
        }
        const uint64_t targets = ~own & check_mask;

        for (b = p->bb[c][KNIGHT] & ~pinned; b; POP(b)) {       /* a pinned knight has no move */
            int sq = lsb(b);
            t = KNIGHT_ATK[sq] & targets;
            s->from[n] = (uint8_t)sq; s->to[n] = t; s->extra[n] = 0; n += t != 0;
        }
        for (b = p->bb[c][BISHOP] | p->bb[c][ROOK] | p->bb[c][QUEEN]; b; POP(b)) {
            int sq = lsb(b);
            t = slider[sq] & targets;
            if (pinned >> sq & 1) t &= ray[sq];
            s->from[n] = (uint8_t)sq; s->to[n] = t; s->extra[n] = 0; n += t != 0;
        }

        const int up = c == WHITE ? 8 : -8;
        const uint64_t jump_rank = c == WHITE ? 0x00000000FF000000ULL : 0x000000FF00000000ULL;   /* where a double step ends */
        const int ep = ep_active ? p->ep_sq : -1;
        for (b = p->bb[c][PAWN]; b; POP(b)) {
            int sq = lsb(b), one = sq + up;
            const uint64_t step = BIT(one) & ~occ;
            t = (PAWN_ATK[c][sq] & opp) | step | ((c == WHITE ? step << 8 : step >> 8) & ~occ & jump_rank);
            t &= targets;
            if (pinned >> sq & 1) t &= ray[sq];
            if (ep >= 0 && (PAWN_ATK[c][sq] >> ep & 1)) {
                int taken = ep ^ 8;
                if (!checkers || checkers == BIT(taken) || (check_mask >> ep & 1)) {
                    /* Both pawns leave their squares: nothing may then look through at the king */
                    uint64_t after = (occ ^ BIT(sq) ^ BIT(taken)) | BIT(ep);
                    if (!(rook_attacks(K, after) & e_orth) && !(bishop_attacks(K, after) & e_diag)) t |= BIT(ep);
                }
            }
            s->from[n] = (uint8_t)sq; s->to[n] = t; s->extra[n] = 0; n += t != 0;
        }

        if (!checkers) {
            for (int side = 0; side < 2; side++) {
                int r = p->castle_rook[c * 2 + side], kt, rt;
                if (r < 0) continue;
                castle_squares(K, r, &kt, &rt);
                uint64_t king_path = BETWEEN_BB[K][kt];
                uint64_t empty = king_path | BETWEEN_BB[r][rt] | BIT(kt) | BIT(rt);
                if ((occ ^ BIT(K) ^ BIT(r)) & empty) continue;          /* something in the way */
                if (danger & king_path) continue;                       /* the king would pass through check */
                if (attacked(p, kt, e, occ ^ BIT(K) ^ BIT(r) ^ BIT(rt))) continue;      /* or end in it, the rook moved */
                s->castle[side] = (int8_t)r;
                if (kt != K) { int i = side_entry(s, n, K); n += i == n; s->extra[i] |= BIT(kt); }
                if (rt != r) { int i = side_entry(s, n, r); n += i == n; s->extra[i] |= BIT(rt); }
            }
        }
    }
    s->n = n;
}

/* The legal moves of the side to move (out[0]) and, when both is set, of the other side as if it were its
   turn, without the en passant square (out[1]). Every slider's attacks are worked out once and serve both
   that piece's own moves and the other king's safety.
   slider: filled with every slider's attacks in p, by its square (the caller's 64 words). */
static void gen_both(const Pos *p, int both, Side out[2], uint64_t *slider) {
    const uint64_t occ = p->occ[0] | p->occ[1];
    uint64_t att[2], b;
    for (int c = 0; c < 2; c++) {
        uint64_t pawns = p->bb[c][PAWN], a = KING_ATK[p->king[c]];
        a |= c == WHITE ? ((pawns & NOT_A) << 7) | ((pawns & NOT_H) << 9) : ((pawns & NOT_A) >> 9) | ((pawns & NOT_H) >> 7);
        for (b = p->bb[c][KNIGHT]; b; POP(b)) a |= KNIGHT_ATK[lsb(b)];
        for (b = p->bb[c][BISHOP]; b; POP(b)) { int sq = lsb(b); a |= slider[sq] = bishop_attacks(sq, occ); }
        for (b = p->bb[c][ROOK]; b; POP(b)) { int sq = lsb(b); a |= slider[sq] = rook_attacks(sq, occ); }
        for (b = p->bb[c][QUEEN]; b; POP(b)) { int sq = lsb(b); a |= slider[sq] = bishop_attacks(sq, occ) | rook_attacks(sq, occ); }
        att[c] = a;
    }
    gen_side(p, p->turn, 1, occ, slider, att[p->turn ^ 1], &out[0]);
    if (both) gen_side(p, p->turn ^ 1, 0, occ, slider, att[p->turn], &out[1]);
}

/* The side to move's legal moves one by one (a promotion is four moves). Returns how many. */
static int list_moves(const Pos *p, const Side *s, Move *mv) {
    int n = 0;
    for (int i = 0; i < s->n; i++) {
        uint8_t f = s->from[i];
        uint64_t t = s->to[i];
        if (p->piece_at[f] == PAWN) {
            for (; t; POP(t)) {
                uint8_t to = (uint8_t)lsb(t);
                if (BACK_RANKS >> to & 1) {
                    for (uint8_t promo = KNIGHT; promo <= QUEEN; promo++) mv[n++] = (Move){f, to, promo, 0};
                } else {
                    /* only a capture puts a pawn on the en passant square: the square before it holds the pawn to take */
                    mv[n++] = (Move){f, to, 0, (uint8_t)(to == p->ep_sq ? F_EP : 0)};
                }
            }
        } else {
            for (; t; POP(t)) mv[n++] = (Move){f, (uint8_t)lsb(t), 0, 0};
        }
    }
    for (int side = 0; side < 2; side++)
        if (s->castle[side] >= 0) mv[n++] = (Move){(uint8_t)p->king[p->turn], (uint8_t)s->castle[side], 0, F_CASTLE};
    return n;
}

/* ---------- ---------- ----------
 *   What is remembered between positions
 */

typedef struct {
    uint64_t hash;
    uint32_t offset;                /* into the arena, in 8-byte words */
    uint16_t gen;                   /* the slot is in use when this is the table's generation */
    uint8_t n;                      /* pieces on the board */
    uint8_t pad;
} Slot;

typedef struct {
    uint64_t key[8];
    int plies, n;
    size_t cap;                     /* words allocated */
    uint64_t *data;                 /* [plies + 1][n][64], pieces in the order of their squares */
} Root;

#define ROOTS 4
#define PROBES 8

typedef struct {
    /* positions one ply above the last, from the fourth ply on: their own threats and their children's */
    Slot *slots;
    size_t slot_mask, slots_used;
    uint16_t gen;
    uint64_t *arena;
    size_t arena_cap, arena_used;   /* in 8-byte words */
    size_t arena_mb;
    int slots_log2;
    Root roots[ROOTS];              /* the last few whole answers (threats) */
    int next_root;
    uint8_t scratch[32 * 64];       /* a position's children's threats, by its own pieces; all 0 between uses */
    uint64_t *out;                  /* the answer being built */
    uint64_t stats[8];              /* 0 positions generated, 1 table hits, 2 table stores, 3 root hits, 4 table clears, 5 searches */
} Ctx;

/* Everything about a position that decides its moves and the moves after them. */
static void make_key(const Pos *p, uint64_t key[8]) {
    for (int pt = 0; pt < 6; pt++) key[pt] = p->bb[0][pt] | p->bb[1][pt];
    key[6] = p->occ[WHITE];
    int ep = -1;
    if (p->ep_sq >= 0) {
        /* The en passant square counts only if a pawn can take there without baring its king to a slider.
           (A knight's check can still forbid it; then the square is kept, which only costs a shared entry.) */
        const int c = p->turn, e = c ^ 1, K = p->king[c], taken = p->ep_sq ^ 8;
        const uint64_t occ = p->occ[0] | p->occ[1];
        const uint64_t e_diag = p->bb[e][BISHOP] | p->bb[e][QUEEN], e_orth = p->bb[e][ROOK] | p->bb[e][QUEEN];
        for (uint64_t b = PAWN_ATK[e][p->ep_sq] & p->bb[c][PAWN]; b; POP(b)) {
            uint64_t after = (occ ^ BIT(lsb(b)) ^ BIT(taken)) | BIT(p->ep_sq);
            if (!(rook_attacks(K, after) & e_orth) && !(bishop_attacks(K, after) & e_diag)) { ep = p->ep_sq; break; }
        }
    }
    key[7] = (uint64_t)p->turn | (uint64_t)(ep + 1) << 8
           | (uint64_t)(p->castle_rook[0] + 1) << 16 | (uint64_t)(p->castle_rook[1] + 1) << 24
           | (uint64_t)(p->castle_rook[2] + 1) << 32 | (uint64_t)(p->castle_rook[3] + 1) << 40;
}

static inline uint64_t hash_key(const uint64_t key[8]) {
    uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 8; i++) {
        h = (h ^ key[i]) * 0xD6E8FEB86659FD93ULL;
        h ^= h >> 32;
    }
    return h;
}

static void table_clear(Ctx *C) {
    if (!C->slots) return;
    if (++C->gen == 0) {            /* the counter came round: really empty the slots */
        memset(C->slots, 0, (C->slot_mask + 1) * sizeof(Slot));
        C->gen = 1;
    }
    C->arena_used = 0;
    C->slots_used = 0;
    C->stats[4]++;
}

static int table_ready(Ctx *C) {
    if (C->slots) return 1;
    if (!C->arena_mb) return 0;
    size_t slots = (size_t)1 << C->slots_log2;
    C->slots = (Slot *)calloc(slots, sizeof(Slot));
    C->arena_cap = C->arena_mb * 1024 * 1024 / 8;
    if (C->arena_cap > 0xFFFFFFFFULL) C->arena_cap = 0xFFFFFFFFULL;
    C->arena = (uint64_t *)malloc(C->arena_cap * 8);
    if (!C->slots || !C->arena) {
        free(C->slots);
        free(C->arena);
        C->slots = NULL;
        C->arena = NULL;
        C->arena_mb = 0;            /* no memory for it: search without */
        return 0;
    }
    C->slot_mask = slots - 1;
    C->gen = 1;
    return 1;
}

/* A stored position: key[8], own[n] (its pieces' threats), cells[n] (which of its children's counts are not 0),
   then those counts, a byte each, in the order of the bits. */
static const uint64_t *table_find(const Ctx *C, uint64_t hash, const uint64_t key[8], int n) {
    if (!C->slots) return NULL;
    for (size_t i = 0; i < PROBES; i++) {
        const Slot *s = &C->slots[(hash + i) & C->slot_mask];
        if (s->gen != C->gen) return NULL;
        if (s->hash == hash && s->n == n && !memcmp(C->arena + s->offset, key, 64)) return C->arena + s->offset;
    }
    return NULL;
}

/* Room for one more position with n pieces, or NULL (then it is simply not stored). */
static uint64_t *table_reserve(Ctx *C, uint64_t hash, int n) {
    if (!table_ready(C)) return NULL;
    size_t need = 8 + 2 * (size_t)n + (size_t)n * 8;        /* the counts take n * 64 bytes at most */
    if (C->arena_used + need > C->arena_cap || C->slots_used * 2 > C->slot_mask) table_clear(C);
    if (need > C->arena_cap) return NULL;
    Slot *slot = NULL;
    for (size_t i = 0; i < PROBES && !slot; i++) {
        Slot *s = &C->slots[(hash + i) & C->slot_mask];
        if (s->gen != C->gen) slot = s;
    }
    if (!slot) return NULL;                                 /* a crowded corner of the table */
    slot->hash = hash;
    slot->offset = (uint32_t)C->arena_used;
    slot->gen = C->gen;
    slot->n = (uint8_t)n;
    C->slots_used++;
    return C->arena + C->arena_used;                        /* the caller says how much it used */
}

/* ---------- ---------- ----------
 *   What one move can change
 */

/* The squares a knight on any square of d attacks (so: where a knight must stand to attack a square of d). */
static inline uint64_t knight_reach(uint64_t d) {
    const uint64_t h1 = ((d >> 1) & NOT_H) | ((d << 1) & NOT_A);
    const uint64_t h2 = ((d >> 2) & 0x3F3F3F3F3F3F3F3FULL) | ((d << 2) & 0xFCFCFCFCFCFCFCFCULL);
    return (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8);
}

/* The squares from which a pawn of colour c looks at a square of d: it would step or jump there, or take there. */
static inline uint64_t pawn_reach(int c, uint64_t d) {
    return c == WHITE ? (d >> 8) | ((d >> 16) & RANK_2) | ((d & NOT_H) >> 7) | ((d & NOT_A) >> 9)
                      : (d << 8) | ((d << 16) & RANK_7) | ((d & NOT_A) << 7) | ((d & NOT_H) << 9);
}

/* Colour c's pinned pieces (as gen_side finds them). A pinned piece moves only along the line through its king. */
static inline uint64_t pinned_of(const Pos *p, int c, uint64_t occ) {
    const int e = c ^ 1, K = p->king[c];
    const uint64_t e_diag = p->bb[e][BISHOP] | p->bb[e][QUEEN], e_orth = p->bb[e][ROOK] | p->bb[e][QUEEN];
    uint64_t pinned = 0;
    for (uint64_t b = (SNIPE_B[K] & e_diag) | (SNIPE_R[K] & e_orth); b; POP(b)) {
        const uint64_t between = BETWEEN_BB[K][lsb(b)] & occ;
        if (between && !(between & (between - 1)) && (between & p->occ[c])) pinned |= between;
    }
    return pinned;
}

/* Colour c's castling when it is not in check (as gen_side decides it): the square its king would go to (kx)
   and, for the h-side and the a-side, the rook (rs) and the square that rook would go to (rx). */
static inline void castle_extras(const Pos *p, int c, uint64_t occ, uint64_t danger, uint64_t *kx, int rs[2], uint64_t rx[2]) {
    const int K = p->king[c], e = c ^ 1;
    for (int side = 0; side < 2; side++) {
        int r = p->castle_rook[c * 2 + side], kt, rt;
        if (r < 0) continue;
        castle_squares(K, r, &kt, &rt);
        const uint64_t king_path = BETWEEN_BB[K][kt];
        const uint64_t empty = king_path | BETWEEN_BB[r][rt] | BIT(kt) | BIT(rt);
        if ((occ ^ BIT(K) ^ BIT(r)) & empty) continue;
        if (danger & king_path) continue;
        if (attacked(p, kt, e, occ ^ BIT(K) ^ BIT(r) ^ BIT(rt))) continue;
        if (kt != K) *kx |= BIT(kt);
        if (rt != r) { rs[side] = r; rx[side] = BIT(rt); }
    }
}

/* ---------- ---------- ----------
 *   Legal threats
 *
 *   out is [plies + 1][64][64]: [k][p][s] counts the sequences of k plies after which the piece that stood on
 *   square p of the root position could legally move to s. who[sq]: the root square of the piece on sq now.
 */

static inline void add_threats(const Side *s, const uint8_t *who, uint64_t *layer) {
    for (int i = 0; i < s->n; i++) {
        uint64_t *row = layer + (size_t)who[s->from[i]] * 64;
        for (uint64_t m = s->to[i] | s->extra[i]; m; POP(m)) row[lsb(m)]++;
    }
}

/* Which piece is where after the move: w[sq] for every occupied square. (Squares left empty keep a stale value
   that nothing reads.) */
static inline void follow(uint8_t *w, Move m) {
    if (m.flags & F_CASTLE) {
        int kt, rt;
        castle_squares(m.from, m.to, &kt, &rt);
        uint8_t king = w[m.from], rook = w[m.to];
        w[kt] = king;
        w[rt] = rook;
    } else {
        w[m.to] = w[m.from];
    }
}

static int root_find(Ctx *C, const uint64_t key[8], int need) {
    for (int i = 0; i < ROOTS; i++)
        if (C->roots[i].data && C->roots[i].plies >= need && !memcmp(C->roots[i].key, key, 64)) return i;
    return -1;
}

static void threats_walk(Ctx *C, const Pos *pos, const uint8_t *who, int left, int ply) {
    uint64_t *layer = C->out + (size_t)ply * 4096;
    uint64_t key[8], hash = 0;
    const uint64_t occ = pos->occ[0] | pos->occ[1];
    const int n = __builtin_popcountll(occ);
    const int stored = left == 1 && ply >= 3;       /* the kind of position the table keeps */

    if (left >= 1 && ply <= 1) {                    /* a whole answer from an earlier search? */
        make_key(pos, key);
        int found = root_find(C, key, left);
        if (found >= 0) {
            const uint64_t *src = C->roots[found].data;
            for (int d = 0; d <= left; d++) {
                uint64_t *dst = C->out + (size_t)(ply + d) * 4096;
                for (uint64_t b = occ; b; POP(b), src += 64) {
                    uint64_t *row = dst + (size_t)who[lsb(b)] * 64;
                    for (int sq = 0; sq < 64; sq++) row[sq] += src[sq];
                }
            }
            C->stats[3]++;
            return;
        }
    }
    if (stored) {
        make_key(pos, key);
        hash = hash_key(key);
        const uint64_t *e = table_find(C, hash, key, n);
        if (e) {
            const uint64_t *own = e + 8, *cells = own + n;
            const uint8_t *count = (const uint8_t *)(cells + n);
            uint64_t *next = layer + 4096;
            int i = 0;
            for (uint64_t b = occ; b; POP(b), i++) {
                size_t r = (size_t)who[lsb(b)] * 64;
                for (uint64_t m = own[i]; m; POP(m)) layer[r + lsb(m)]++;
                for (uint64_t m = cells[i]; m; POP(m)) next[r + lsb(m)] += *count++;
            }
            C->stats[1]++;
            return;
        }
    }

    Side sd[2];
    uint64_t slider[64];
    gen_both(pos, 1, sd, slider);
    C->stats[0]++;
    add_threats(&sd[0], who, layer);
    add_threats(&sd[1], who, layer);
    if (left == 0) return;

    Move mv[MAX_MOVES];
    const int nm = list_moves(pos, &sd[0], mv);

    if (left == 1 && nm <= 255) {
        /* The children are only looked at. Their threats are gathered by this position's own pieces, a byte
           a count (at most one per child), then added to the answer in one go and, from the fourth ply on,
           kept for the next time this position turns up.
           One move changes little: in most children a piece threatens exactly what it threatens here (base).
           Those children are only counted (same); the squares are walked one by one just for the rest. */
        uint8_t local[64] = {0}, same[32], differs[32], *scratch = C->scratch;
        uint8_t mine[2][16];                            /* each colour's pieces here, by their numbers */
        int n_mine[2] = {0, 0};
        uint64_t cells[32], base[32];
        int i = 0;
        for (uint64_t b = occ; b; POP(b), i++) {
            const int sq = lsb(b), colour = (int)(pos->occ[BLACK] >> sq & 1);
            local[sq] = (uint8_t)i; cells[i] = 0; base[i] = 0; differs[i] = 0;
            mine[colour][n_mine[colour]++] = (uint8_t)i;
        }
        for (int side = 0; side < 2; side++)
            for (int j = 0; j < sd[side].n; j++) base[local[sd[side].from[j]]] = sd[side].to[j] | sd[side].extra[j];

        /* A child is not worked out whole. A piece's threats there can differ from its threats here only if the
           move reaches it: it moved; a square it looks at changed hands (D); its pin, its king's safety or its
           right to castle or to take en passant changed. Those pieces are worked out again; every other piece
           threatens what it threatens here, and is not looked at. A side that is in check (there, or here when
           it is the side to move) or whose king moved is worked out whole, as before. differs[i]: the children
           in which piece i does not threaten exactly what it threatens here. */
        const int M = pos->turn, O = M ^ 1, KM = pos->king[M];
        const uint64_t pinned_here[2] = {pinned_of(pos, WHITE, occ), pinned_of(pos, BLACK, occ)};
        const uint64_t king_lines[2] = {SNIPE_B[pos->king[WHITE]] | SNIPE_R[pos->king[WHITE]],
                                        SNIPE_B[pos->king[BLACK]] | SNIPE_R[pos->king[BLACK]]};
        const int checked_here = attacked(pos, KM, O, occ);
        const uint64_t ep_pawns = pos->ep_sq >= 0 ? PAWN_ATK[O][pos->ep_sq] & pos->bb[M][PAWN] : 0;
        uint64_t castle_rooks[2] = {0, 0};
        for (int c = 0; c < 2; c++)
            for (int side = 0; side < 2; side++)
                if (pos->castle_rook[c * 2 + side] >= 0) castle_rooks[c] |= BIT(pos->castle_rook[c * 2 + side]);

#define TALLY(sq_, t_) do { \
            const int piece_ = local[sq_]; \
            const uint64_t mask_ = (t_); \
            if (mask_ != base[piece_]) { \
                uint8_t *row_ = scratch + piece_ * 64; \
                differs[piece_]++; \
                cells[piece_] |= mask_; \
                for (uint64_t x_ = mask_; x_; POP(x_)) row_[lsb(x_)]++; \
            } \
        } while (0)

        for (int k = 0; k < nm; k++) {
            Pos child;
            const Move m = mv[k];
            make_leaf(pos, m, &child);
            int kt = m.to, rt = m.to;
            if (m.flags & F_CASTLE) castle_squares(m.from, m.to, &kt, &rt);
            const uint8_t keep_k = local[kt], keep_r = local[rt];
            const int taken = (m.flags & F_EP) ? local[m.to ^ 8]
                            : (!(m.flags & F_CASTLE) && pos->piece_at[m.to] >= 0) ? keep_k : -1;
            follow(local, m);

            const uint64_t occ2 = child.occ[0] | child.occ[1];
            const uint64_t changed = occ ^ occ2;                                            /* empty <-> occupied */
            const uint64_t D = (pos->occ[0] ^ child.occ[0]) | (pos->occ[1] ^ child.occ[1]);  /* ... or changed hands */
            const uint64_t fresh = BIT(kt) | BIT(rt);                                       /* where a piece arrived */
            const uint64_t moved = fresh | BIT(m.from) | BIT(m.to);                         /* ... or left */
            /* Every slider's attacks (kept from here when no square it attacked changed), each colour's attacks,
               and the sliders that look at a square of D */
            uint64_t child_slider[64], att[2], touched = 0, b;
#define SLIDE(expr) { \
                const int sq = lsb(b); \
                uint64_t v; \
                if (fresh >> sq & 1) { v = (expr); touched |= BIT(sq); } \
                else { v = slider[sq]; if (D & v) { touched |= BIT(sq); if (changed & v) v = (expr); } } \
                child_slider[sq] = v; \
                a |= v; }
            for (int c = 0; c < 2; c++) {
                uint64_t pawns = child.bb[c][PAWN], a = KING_ATK[child.king[c]];
                a |= c == WHITE ? ((pawns & NOT_A) << 7) | ((pawns & NOT_H) << 9) : ((pawns & NOT_A) >> 9) | ((pawns & NOT_H) >> 7);
                for (b = child.bb[c][KNIGHT]; b; POP(b)) a |= KNIGHT_ATK[lsb(b)];
                for (b = child.bb[c][BISHOP]; b; POP(b)) SLIDE(bishop_attacks(sq, occ2))
                for (b = child.bb[c][ROOK]; b; POP(b)) SLIDE(rook_attacks(sq, occ2))
                for (b = child.bb[c][QUEEN]; b; POP(b)) SLIDE(bishop_attacks(sq, occ2) | rook_attacks(sq, occ2))
                att[c] = a;
            }
#undef SLIDE

            for (int c = 0; c < 2; c++) {
                const int e = c ^ 1, K = child.king[c], to_move = c == O;       /* in the child the other side moves */
                if (c == O ? (int)(att[M] >> K & 1) : (checked_here || K != KM)) {
                    /* the whole side: every piece is taken to differ, then those that do not are given back */
                    Side all;
                    gen_side(&child, c, to_move, occ2, child_slider, att[e], &all);
                    for (int j = 0; j < n_mine[c]; j++) differs[mine[c][j]]++;
                    for (int j = 0; j < all.n; j++) {
                        const int piece = local[all.from[j]];
                        const uint64_t t = all.to[j] | all.extra[j];
                        if (t == base[piece]) { differs[piece]--; continue; }
                        uint8_t *row = scratch + piece * 64;
                        cells[piece] |= t;
                        for (uint64_t x = t; x; POP(x)) row[lsb(x)]++;
                    }
                    continue;
                }
                const uint64_t own = child.occ[c], opp = child.occ[e];
                if (to_move && taken >= 0) differs[taken]++;                    /* a piece that is gone threatens nothing */
                /* (a pin can come or go wherever a piece left or arrived: castling can swap a king and a rook
                   without any square changing hands) */
                const uint64_t pinned = ((D | moved) & king_lines[c]) ? pinned_of(&child, c, occ2) : pinned_here[c];
                const int ep = to_move ? child.ep_sq : -1;
                uint64_t reached = (fresh & own)
                                 | (child.bb[c][KNIGHT] & knight_reach(D))
                                 | (touched & own)
                                 | (child.bb[c][PAWN] & pawn_reach(c, D))
                                 | ((pinned ^ pinned_here[c]) & own)
                                 | (castle_rooks[c] & child.bb[c][ROOK]);
                if (ep >= 0) reached |= PAWN_ATK[e][ep] & child.bb[c][PAWN];
                if (!to_move) reached |= ep_pawns & child.bb[c][PAWN];
                uint64_t kx = 0, rx[2] = {0, 0};
                int rs[2] = {-1, -1};
                if (child.castle_rook[c * 2] >= 0 || child.castle_rook[c * 2 + 1] >= 0)
                    castle_extras(&child, c, occ2, att[e], &kx, rs, rx);

                TALLY(K, (KING_ATK[K] & ~own & ~att[e]) | kx);
                reached &= ~BIT(K);
                for (b = reached & child.bb[c][KNIGHT]; b; POP(b)) {
                    const int sq = lsb(b);
                    TALLY(sq, (pinned >> sq & 1) ? 0 : KNIGHT_ATK[sq] & ~own);
                }
                for (b = reached & (child.bb[c][BISHOP] | child.bb[c][ROOK] | child.bb[c][QUEEN]); b; POP(b)) {
                    const int sq = lsb(b);
                    uint64_t t = child_slider[sq] & ~own;
                    if (pinned >> sq & 1) t &= LINE_BB[K][sq];
                    t |= (sq == rs[0] ? rx[0] : 0) | (sq == rs[1] ? rx[1] : 0);
                    TALLY(sq, t);
                }
                if (reached & child.bb[c][PAWN]) {
                    const int up = c == WHITE ? 8 : -8;
                    const uint64_t jump_rank = c == WHITE ? 0x00000000FF000000ULL : 0x000000FF00000000ULL;
                    const uint64_t e_diag = child.bb[e][BISHOP] | child.bb[e][QUEEN], e_orth = child.bb[e][ROOK] | child.bb[e][QUEEN];
                    for (b = reached & child.bb[c][PAWN]; b; POP(b)) {
                        const int sq = lsb(b), one = sq + up;
                        const uint64_t step = BIT(one) & ~occ2;
                        uint64_t t = (PAWN_ATK[c][sq] & opp) | step | ((c == WHITE ? step << 8 : step >> 8) & ~occ2 & jump_rank);
                        if (pinned >> sq & 1) t &= LINE_BB[K][sq];
                        if (ep >= 0 && (PAWN_ATK[c][sq] >> ep & 1)) {
                            const uint64_t after = (occ2 ^ BIT(sq) ^ BIT(ep ^ 8)) | BIT(ep);
                            if (!(rook_attacks(K, after) & e_orth) && !(bishop_attacks(K, after) & e_diag)) t |= BIT(ep);
                        }
                        TALLY(sq, t);
                    }
                }
            }
            local[kt] = keep_k;
            local[rt] = keep_r;
        }
#undef TALLY
        for (i = 0; i < n; i++) same[i] = (uint8_t)(nm - differs[i]);
        C->stats[0] += (uint64_t)nm;

        uint64_t *entry = stored ? table_reserve(C, hash, n) : NULL;
        uint8_t *count = NULL;
        if (entry) {
            memcpy(entry, key, 64);
            memcpy(entry + 8, base, (size_t)n * 8);
            count = (uint8_t *)(entry + 8 + 2 * n);
        }
        uint64_t *next = layer + 4096;
        i = 0;
        for (uint64_t b = occ; b; POP(b), i++) {
            uint64_t *row = next + (size_t)who[lsb(b)] * 64;
            uint8_t *from = scratch + i * 64;
            const uint64_t kept = same[i] ? base[i] : 0, all = cells[i] | kept;
            for (uint64_t m = all; m; POP(m)) {
                const int sq = lsb(m);
                const uint8_t c = (uint8_t)(from[sq] + ((kept >> sq & 1) ? same[i] : 0));
                row[sq] += c;
                if (count) *count++ = c;
                from[sq] = 0;
            }
            if (entry) entry[8 + n + i] = all;
        }
        if (entry) {
            C->arena_used += ((size_t)(count - (uint8_t *)entry) + 7) / 8;
            C->stats[2]++;
        }
        return;
    }

    for (int k = 0; k < nm; k++) {
        Pos child;
        uint8_t w[64];
        make_move(pos, mv[k], &child);
        memcpy(w, who, 64);
        follow(w, mv[k]);
        threats_walk(C, &child, w, left - 1, ply + 1);
    }
}

/* ---------- ---------- ----------
 *   Reach
 *
 *   out is [depth][64][65]: [d - 1][p][s] counts the sequences of d plies after which the piece that stood on
 *   square p of the root position stands on s (65th column: is off the board). where[p]: where that piece is
 *   now. The last ply is not played: a move leaves every piece where it was, except the one or two it moves or
 *   takes, so a position adds its number of moves for every piece and then corrects those.
 */

static void reach_walk(Ctx *C, const Pos *pos, const uint8_t *who, const uint8_t *where,
                       const uint8_t *pieces, int n, int ply, int depth) {
    Side sd[2];
    Move mv[MAX_MOVES];
    uint64_t slider[64];
    gen_both(pos, 0, sd, slider);
    C->stats[0]++;
    const int nm = list_moves(pos, &sd[0], mv);
    if (!nm) return;
    uint64_t *layer = C->out + (size_t)ply * 64 * 65;
    for (int i = 0; i < n; i++) layer[(size_t)pieces[i] * 65 + where[pieces[i]]] += (uint64_t)nm;
    for (int k = 0; k < nm; k++) {
        const Move m = mv[k];
        if (m.flags & F_CASTLE) {
            int kt, rt;
            castle_squares(m.from, m.to, &kt, &rt);
            uint64_t *king = layer + (size_t)who[m.from] * 65, *rook = layer + (size_t)who[m.to] * 65;
            king[m.from]--; king[kt]++;
            rook[m.to]--; rook[rt]++;
        } else {
            const int taken = (m.flags & F_EP) ? (m.to ^ 8) : (pos->piece_at[m.to] >= 0 ? m.to : -1);
            if (taken >= 0) {
                uint64_t *row = layer + (size_t)who[taken] * 65;
                row[taken]--; row[CAPTURED]++;
            }
            uint64_t *row = layer + (size_t)who[m.from] * 65;
            row[m.from]--; row[m.to]++;
        }
    }
    if (ply + 1 == depth) return;
    for (int k = 0; k < nm; k++) {
        const Move m = mv[k];
        Pos child;
        uint8_t w[64], wh[64];
        make_move(pos, m, &child);
        memcpy(w, who, 64);
        memcpy(wh, where, 64);
        if (m.flags & F_CASTLE) {
            int kt, rt;
            castle_squares(m.from, m.to, &kt, &rt);
            wh[who[m.from]] = (uint8_t)kt;
            wh[who[m.to]] = (uint8_t)rt;
        } else {
            const int taken = (m.flags & F_EP) ? (m.to ^ 8) : (pos->piece_at[m.to] >= 0 ? m.to : -1);
            if (taken >= 0) wh[who[taken]] = CAPTURED;
            wh[who[m.from]] = m.to;
        }
        follow(w, m);
        reach_walk(C, &child, w, wh, pieces, n, ply + 1, depth);
    }
}

/* ---------- ---------- ----------
 *   What piece_reach.py calls
 */

/* Raised whenever the functions below or what they return change: piece_reach.py refuses a library built from
   another version of this file, so a stale build is never used by mistake. */
API int fr_version(void) { return 1; }

/* A search context: one per process. table_mb: memory for remembered positions (0: remember none);
   slots_log2: the table holds up to 2^slots_log2 / 2 of them. Both are only allocated when first needed
   (threats to 4 plies or more). */
API void *fr_new(size_t table_mb, int slots_log2) {
    init_tables();
    Ctx *C = (Ctx *)calloc(1, sizeof(Ctx));
    if (!C) return NULL;
    C->arena_mb = table_mb;
    C->slots_log2 = slots_log2 < 10 ? 10 : (slots_log2 > 28 ? 28 : slots_log2);
    return C;
}

API void fr_free(void *ctx) {
    Ctx *C = (Ctx *)ctx;
    if (!C) return;
    for (int i = 0; i < ROOTS; i++) free(C->roots[i].data);
    free(C->slots);
    free(C->arena);
    free(C);
}

/* Forget everything remembered from earlier searches. Never needed for a right answer. */
API void fr_clear(void *ctx) {
    Ctx *C = (Ctx *)ctx;
    table_clear(C);
    for (int i = 0; i < ROOTS; i++) C->roots[i].plies = -1;
}

/* 0 positions generated, 1 table hits, 2 table stores, 3 whole answers reused, 4 table clears, 5 searches */
API void fr_stats(void *ctx, uint64_t out[8]) {
    memcpy(out, ((Ctx *)ctx)->stats, sizeof(((Ctx *)ctx)->stats));
}

/* out: [plies + 1][64][64] uint64. 0, or non-zero when the position is not one this file handles. */
API int fr_threat_counts(void *ctx, const char *fen, int plies, uint64_t *out) {
    Ctx *C = (Ctx *)ctx;
    Pos pos;
    if (!C || plies < 0 || plies > MAX_PLIES || parse_fen(fen, &pos)) return -1;
    uint8_t who[64];
    for (int sq = 0; sq < 64; sq++) who[sq] = (uint8_t)sq;
    memset(out, 0, (size_t)(plies + 1) * 4096 * 8);
    C->out = out;
    C->stats[5]++;
    threats_walk(C, &pos, who, plies, 0);

    if (plies >= 1) {               /* keep the answer for the searches of the positions before this one */
        const uint64_t occ = pos.occ[0] | pos.occ[1];
        const int n = __builtin_popcountll(occ);
        const size_t need = (size_t)(plies + 1) * (size_t)n * 64;
        uint64_t key[8];
        make_key(&pos, key);
        if (root_find(C, key, plies) >= 0) return 0;        /* already kept, at least this deep */
        int at = root_find(C, key, 0);
        if (at < 0) { at = C->next_root; C->next_root = (C->next_root + 1) % ROOTS; }
        Root *r = &C->roots[at];
        if (r->cap < need) {
            free(r->data);
            r->data = (uint64_t *)malloc(need * 8);
            r->cap = r->data ? need : 0;
        }
        if (r->data) {
            uint64_t *dst = r->data;
            memcpy(r->key, key, 64);
            r->plies = plies;
            r->n = n;
            for (int d = 0; d <= plies; d++)
                for (uint64_t b = occ; b; POP(b), dst += 64) memcpy(dst, out + (size_t)d * 4096 + (size_t)lsb(b) * 64, 512);
        }
    }
    return 0;
}

/* out: [depth][64][65] uint64. 0, or non-zero when the position is not one this file handles. */
API int fr_reach_counts(void *ctx, const char *fen, int depth, uint64_t *out) {
    Ctx *C = (Ctx *)ctx;
    Pos pos;
    if (!C || depth < 1 || depth > MAX_PLIES || parse_fen(fen, &pos)) return -1;
    uint8_t who[64], where[64], pieces[32];
    int n = 0;
    for (int sq = 0; sq < 64; sq++) who[sq] = where[sq] = (uint8_t)sq;
    for (uint64_t b = pos.occ[0] | pos.occ[1]; b; POP(b)) pieces[n++] = (uint8_t)lsb(b);
    memset(out, 0, (size_t)depth * 64 * 65 * 8);
    C->out = out;
    C->stats[5]++;
    reach_walk(C, &pos, who, where, pieces, n, 0, depth);
    return 0;
}
