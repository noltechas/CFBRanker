# professional_bracket.py
# Componentized, fast bracket renderer with correct seed placement (left of logo)
# and smaller logos that don't touch container edges.

import matplotlib
matplotlib.use("Agg")  # headless + fast

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import requests


# =========================
# THEME & SIZING
# =========================
THEME = {
    "bg_top":  "#0e172a",
    "bg_bot":  "#111c35",
    "panel":   "#1a2746",
    "card":    "#223358",
    "card_hi": "#2b4170",
    "ink":     "#eef3fb",
    "ink_sub": "#c7d5ea",
    "ink_dim": "#9db0cf",
    "line":    "#7da0d8",
    "accent_at":  "#2196f3",  # At-large
    "accent_auto":"#26c281",  # Auto-bid
    "gold":    "#ffd166",
    "shadow":  "#000000",
    "title":   "#f06c64",
}

SIZES = {
    # Canvas
    "fig_w": 24.0,
    "fig_h": 16.0,

    # MatchPod
    "pod_w":  4.8,
    "row_h":  0.98,
    "row_gap":0.22,

    # TeamCard internals (relative to inset 0..1 space)
    "pad":         0.045,
    "accent_w":    0.018,
    "seed_r":      0.075,  # circle radius (kept prominent)
    "logo_w":      0.15,   # smaller to avoid edge collisions (was 0.18)
    "logo_gap":    0.045,  # extra space after logo
    "vs_r":        0.15,

    # Typography baselines
    "fs_name":     15,
    "fs_record":   12,
    "fs_conf":     11,
    "fs_seed":     14,
    "fs_vs":       12,
}

def _stroke():
    return [pe.withStroke(linewidth=3, foreground="black", alpha=0.45)]


# =========================
# CANVAS
# =========================
class Canvas:
    def __init__(self, w=SIZES["fig_w"], h=SIZES["fig_h"]):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(32, 20), dpi=120)
        self.fig.patch.set_facecolor(THEME["bg_top"])
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.axis("off")
        self._gradient()

    def _gradient(self):
        grad = np.linspace(0, 1, 256).reshape(-1, 1)
        self.ax.imshow(
            grad, extent=[0, self.w, 0, self.h],
            cmap=plt.cm.colors.LinearSegmentedColormap.from_list(
                "", [THEME["bg_top"], THEME["bg_bot"]]
            ),
            aspect="auto", alpha=1.0, zorder=0,
        )

    def save(self, path):
        plt.tight_layout()
        plt.subplots_adjust(left=0.02, right=0.985, top=0.965, bottom=0.04)
        self.fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=THEME["bg_top"])
        return path


# =========================
# ASSETS (short timeouts + cache)
# =========================
_LOGO_CACHE = {}

def _placeholder_logo(size=(84,84)):
    w, h = size
    ph = Image.new("RGBA", size, (140, 145, 155, 255))
    d = ImageDraw.Draw(ph)
    d.rounded_rectangle([8,8,w-8,h-8], radius=14, outline=(230,230,235,255), width=6)
    d.ellipse([20,20,w-20,h-20], fill=(210,215,220,255))
    return ph

def _fetch_logo(url, size=(84,84)):
    if not url or not isinstance(url, str) or not url.startswith("http"):
        return None
    key = (url, size)
    if key in _LOGO_CACHE:
        return _LOGO_CACHE[key]
    try:
        r = requests.get(url, timeout=(0.6, 0.8), allow_redirects=True)
        r.raise_for_status()
        im = Image.open(BytesIO(r.content)).convert("RGBA").resize(size, Image.Resampling.LANCZOS)
    except Exception:
        im = None
    _LOGO_CACHE[key] = im
    return im

def _logo_for(team_info, team, size=(84,84), allow_placeholder=True):
    if team is None:
        return None  # no logo for TBD/BYE
    info = team_info.get(team.id, {})
    logos = info.get("logos") or []
    img = _fetch_logo(logos[0], size=size) if logos else None
    if img is None and allow_placeholder:
        return _placeholder_logo(size=size)
    return img


# =========================
# FAST TEXT FITTING (no redraw loops)
# =========================
def _measure_text_px(ax, renderer, text, fs, **fontkw):
    t = ax.text(0, 0, text, fontsize=fs, alpha=0, **fontkw)
    t.draw(renderer)
    bb = t.get_window_extent(renderer=renderer)
    t.remove()
    return bb.width

def _fit_text_once(ax, renderer, text, max_w_rel, fs_base, min_fs=8, **fontkw):
    max_w_px = max_w_rel * ax.bbox.width
    w0 = _measure_text_px(ax, renderer, text, fs_base, **fontkw)
    if w0 <= max_w_px:
        return text, fs_base

    fs_scaled = max(min_fs, int(fs_base * (max_w_px / max(w0, 1))))
    w1 = _measure_text_px(ax, renderer, text, fs_scaled, **fontkw)
    if w1 <= max_w_px:
        return text, fs_scaled

    avg_char = w1 / max(len(text), 1)
    n = max(1, int((max_w_px / max(avg_char, 1))) - 1)
    txt = (text if n >= len(text) else (text[:n] + "…"))
    w2 = _measure_text_px(ax, renderer, txt, fs_scaled, **fontkw)
    while w2 > max_w_px and len(txt) > 1:
        txt = txt[:-2] + "…"
        w2 = _measure_text_px(ax, renderer, txt, fs_scaled, **fontkw)
    return txt, fs_scaled


# =========================
# CARD COMPONENTS
# =========================
class AccentBar:
    def __init__(self, kind):
        self.kind = kind

    def draw(self, ax, pad, accent_w):
        color = THEME["accent_auto"] if self.kind == "Auto-bid" else THEME["accent_at"]
        ax.add_patch(patches.Rectangle((pad, pad), accent_w, 1 - 2*pad,
                                       facecolor=color, zorder=3))

class SeedBadge:
    def __init__(self, seed):
        self.seed = seed

    def draw(self, ax, x, y, r):
        if self.seed is None:
            return
        ax.add_patch(patches.Circle((x, y), r, facecolor=THEME["gold"],
                                    edgecolor="white", linewidth=3, zorder=6))
        ax.text(x, y, str(self.seed), ha="center", va="center",
                fontsize=SIZES["fs_seed"], fontweight="bold",
                color="black", zorder=7)

class LogoAvatar:
    def __init__(self, logo_img):
        self.logo_img = logo_img

    def draw(self, ax, x, y, w, h):
        if self.logo_img is None:
            return
        cx = x + w/2
        cy = y + h/2
        # Slightly smaller zoom so image never touches edges
        scale = 0.75 * (w / 0.18)  # shrink vs previous 0.95*(w/0.18)
        ab = AnnotationBbox(OffsetImage(self.logo_img, zoom=scale),
                            (cx, cy), frameon=False, zorder=5, box_alignment=(0.5,0.5))
        ax.add_artist(ab)

class TextBlock:
    def __init__(self, team, is_bye=False):
        self.team = team
        self.is_bye = is_bye

    def draw(self, ax, renderer, x, y, w, h):
        name_y = y + h*0.66
        rec_y  = y + h*0.45
        conf_y = y + h*0.24

        if self.team is None and self.is_bye:
            name, rec, conf = "BYE", "—", ""
        elif self.team is None:
            name, rec, conf = "TBD", "", ""
        else:
            name = self.team.name
            rec  = f"({self.team.wins_count}-{self.team.losses_count})"
            conf = self.team.conference.name

        nm, fsn = _fit_text_once(ax, renderer, name, w, SIZES["fs_name"])
        ax.text(x, name_y, nm, fontsize=fsn, color=THEME["ink"],
                ha="left", va="center", fontweight="bold", path_effects=_stroke(), zorder=8)

        if rec:
            rc, fsr = _fit_text_once(ax, renderer, rec, w, SIZES["fs_record"])
            ax.text(x, rec_y, rc, fontsize=fsr, color=THEME["ink_sub"],
                    ha="left", va="center", path_effects=_stroke(), zorder=8)

        if conf:
            cf, fsc = _fit_text_once(ax, renderer, conf, w, SIZES["fs_conf"])
            ax.text(x, conf_y, cf, fontsize=fsc, color=THEME["ink_dim"],
                    ha="left", va="center", style="italic", path_effects=_stroke(), zorder=8)

class TeamCard:
    """
    Draws inside an inset axis (0..1 space) so internals stay proportionate.
    Seed badge is circular and placed to the LEFT of the logo, vertically centered.
    """
    def __init__(self, team, seed=None, kind="At-large", logo=None, is_bye=False):
        self.team = team
        self.seed = seed
        self.kind = kind
        self.logo = logo
        self.is_bye = is_bye

    def draw(self, root_ax, renderer, x, y, w, h):
        # Inset axis in DATA units (correct placement)
        iax = root_ax.inset_axes([x, y, w, h], transform=root_ax.transData)
        iax.set_xlim(0, 1); iax.set_ylim(0, 1); iax.axis("off")

        pad      = SIZES["pad"]
        accent_w = SIZES["accent_w"]
        seed_r   = SIZES["seed_r"]
        logo_w   = SIZES["logo_w"]
        gap      = SIZES["logo_gap"]

        # Card bg + shadow
        iax.add_patch(patches.FancyBboxPatch(
            (0 + 0.01, 0 - 0.01), 1, 1, boxstyle="round,pad=0.02",
            facecolor=THEME["shadow"], alpha=0.28, zorder=1))
        iax.add_patch(patches.FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0.02",
            facecolor=THEME["card"], edgecolor=THEME["card_hi"], linewidth=2, zorder=2))

        # Accent
        AccentBar(self.kind).draw(iax, pad, accent_w)

        # Seed (LEFT of logo, vertically centered)
        seed_cx = pad + accent_w + seed_r + 0.02
        seed_cy = 0.5
        SeedBadge(self.seed).draw(iax, seed_cx, seed_cy, seed_r)

        # Logo box (to the RIGHT of seed)
        logo_x = pad + accent_w + 2*seed_r + 0.06
        LogoAvatar(self.logo).draw(iax, logo_x, pad, logo_w, 1 - 2*pad)

        # Text block (after logo)
        text_x = logo_x + logo_w + gap
        text_w = max(0.05, 1 - text_x - pad)
        TextBlock(self.team, is_bye=self.is_bye).draw(iax, renderer, text_x, pad, text_w, 1 - 2*pad)

        return (x, y + h/2), (x + w, y + h/2)

class VsChip:
    def __init__(self): pass
    def draw(self, ax, x, cy):
        r = SIZES["vs_r"]
        ax.add_patch(patches.Circle((x, cy), r, facecolor=THEME["card_hi"],
                                    edgecolor="white", linewidth=2, zorder=7))
        ax.text(x, cy, "VS", ha="center", va="center",
                fontsize=SIZES["fs_vs"], fontweight="bold",
                color=THEME["ink"], zorder=8)

class MatchPod:
    def __init__(self, top_card: TeamCard, bot_card: TeamCard):
        self.top = top_card
        self.bot = bot_card

    @property
    def height(self):
        return SIZES["row_h"]*2 + SIZES["row_gap"]

    def draw(self, ax, renderer, x, cy, w=SIZES["pod_w"]):
        rh = SIZES["row_h"]
        gap = SIZES["row_gap"]
        top_y = cy + gap/2
        bot_y = cy - gap/2 - rh

        left_port_top, right_port_top = self.top.draw(ax, renderer, x, top_y, w, rh)
        left_port_bot, right_port_bot = self.bot.draw(ax, renderer, x, bot_y, w, rh)

        ax.add_line(plt.Line2D([x + 0.16, x + w - 0.58], [cy, cy],
                               color=THEME["line"], linewidth=1.4, alpha=0.85, zorder=6))
        VsChip().draw(ax, x + w - 0.36, cy)

        return (x, cy), (x + w, cy)


# =========================
# CONNECTORS
# =========================
def straight(ax, start, end):
    x1, y1 = start; x2, y2 = end
    ax.plot([x1, x2], [y1, y2], color=THEME["line"], linewidth=3.2, alpha=0.95, zorder=1)

def elbow(ax, start, end, bend_x=None):
    x1, y1 = start; x2, y2 = end
    if bend_x is None:
        bend_x = x1 + 0.55*(x2 - x1)
    ax.plot([x1, bend_x], [y1, y1], color=THEME["line"], linewidth=3.2, alpha=0.95, zorder=1)
    ax.plot([bend_x, bend_x], [y1, y2], color=THEME["line"], linewidth=3.2, alpha=0.95, zorder=1)
    ax.plot([bend_x, x2], [y2, y2], color=THEME["line"], linewidth=3.2, alpha=0.95, zorder=1)
    ax.add_patch(patches.Circle((bend_x, y2), 0.035, facecolor=THEME["gold"],
                                edgecolor="white", linewidth=1.4, zorder=2))


# =========================
# LAYOUT + RENDER
# =========================
def _title(ax, year):
    ax.add_patch(patches.FancyBboxPatch((1.0, 14.6), 22.0, 1.2, boxstyle="round,pad=0.12",
                                        facecolor=THEME["title"], edgecolor=THEME["gold"], linewidth=3, zorder=2))
    ax.text(12.0, 15.2, f"{year} COLLEGE FOOTBALL PLAYOFF",
            ha="center", va="center", fontsize=34, fontweight="bold",
            color=THEME["ink"], path_effects=_stroke(), zorder=3)
    ax.text(12.0, 14.85, "12-Team Championship Bracket",
            ha="center", va="center", fontsize=18, color=THEME["ink_sub"], zorder=3)

def _round_label(ax, x, text):
    ax.text(x, 13.9, text, ha="center", va="center", fontsize=14,
            fontweight="bold", color=THEME["ink_sub"], zorder=3)


def create_professional_playoff_bracket(playoff_teams, year, team_info):
    cvs = Canvas()
    ax = cvs.ax
    _title(ax, year)

    cvs.fig.canvas.draw()
    renderer = cvs.fig.canvas.get_renderer()

    # Columns (left edges) & row centers
    X_R1, X_QF, X_SF, X_CH = 1.0, 7.4, 13.8, 20.2
    W_POD = SIZES["pod_w"]
    Y_ROWS = [12.8, 10.4, 8.0, 5.6]
    Y_SF   = [11.6, 7.2]
    Y_CH   = 9.4

    _round_label(ax, X_R1 + W_POD/2, "FIRST ROUND")
    _round_label(ax, X_QF + W_POD/2, "QUARTERFINALS")
    _round_label(ax, X_SF + W_POD/2, "SEMIFINALS")
    _round_label(ax, X_CH + W_POD/2, "CHAMPIONSHIP")

    # maps
    seeds = {team: i + 1 for i, (team, _, _) in enumerate(playoff_teams)}
    kinds = {team: kind for team, _, kind in playoff_teams}

    top4 = [t for t,_,_ in playoff_teams[:4]]
    fr   = [t for t,_,_ in playoff_teams[4:12]]

    # Round 1 pairs: (5v12, 6v11, 7v10, 8v9)
    r1_pairs = [
        (fr[0] if len(fr)>0 else None, fr[7] if len(fr)>7 else None),
        (fr[1] if len(fr)>1 else None, fr[6] if len(fr)>6 else None),
        (fr[2] if len(fr)>2 else None, fr[5] if len(fr)>5 else None),
        (fr[3] if len(fr)>3 else None, fr[4] if len(fr)>4 else None),
    ]

    # --- Draw Round 1 pods
    r1_ports = []
    for i, (t1, t2) in enumerate(r1_pairs):
        pod = MatchPod(
            TeamCard(t1, seed=seeds.get(t1), kind=kinds.get(t1,"At-large"),
                     logo=_logo_for(team_info, t1, allow_placeholder=True)),
            TeamCard(t2, seed=seeds.get(t2), kind=kinds.get(t2,"At-large"),
                     logo=_logo_for(team_info, t2, allow_placeholder=True)),
        )
        r1_ports.append(pod.draw(ax, renderer, X_R1, Y_ROWS[i], W_POD))

    # --- Quarterfinals pods: Top-4 vs BYE (no placeholder logos for BYE)
    qf_ports = []
    for i, t in enumerate(top4):
        pod = MatchPod(
            TeamCard(t, seed=seeds.get(t), kind=kinds.get(t,"At-large"),
                     logo=_logo_for(team_info, t, allow_placeholder=True)),
            TeamCard(None, seed=None, kind="At-large", logo=None, is_bye=True),
        )
        qf_ports.append(pod.draw(ax, renderer, X_QF, Y_ROWS[i], W_POD))

    # --- Semifinals pods (TBD vs TBD, no logos)
    sf_ports = []
    for i in range(2):
        pod = MatchPod(
            TeamCard(None, seed=None, kind="At-large", logo=None),
            TeamCard(None, seed=None, kind="At-large", logo=None),
        )
        sf_ports.append(pod.draw(ax, renderer, X_SF, Y_SF[i], W_POD))

    # --- Championship panel
    w_ch, h_ch = 4.8, 1.15
    ax.add_patch(patches.FancyBboxPatch((X_CH + 0.02, Y_CH - h_ch/2 - 0.02), w_ch, h_ch,
                                        boxstyle="round,pad=0.06", facecolor=THEME["shadow"], alpha=0.35, zorder=1))
    ax.add_patch(patches.FancyBboxPatch((X_CH, Y_CH - h_ch/2), w_ch, h_ch,
                                        boxstyle="round,pad=0.06", facecolor=THEME["gold"],
                                        edgecolor="white", linewidth=3, zorder=2))
    ax.text(X_CH + w_ch/2, Y_CH, "NATIONAL CHAMPION",
            ha="center", va="center", fontsize=18, fontweight="bold", color="black", zorder=3)

    # --- Connectors
    for i in range(4):
        straight(ax, r1_ports[i][1], qf_ports[i][0])

    elbow(ax, qf_ports[0][1], (X_SF, Y_SF[0]), bend_x=X_SF - 0.6)
    elbow(ax, qf_ports[1][1], (X_SF, Y_SF[0]), bend_x=X_SF - 0.6)
    elbow(ax, qf_ports[2][1], (X_SF, Y_SF[1]), bend_x=X_SF - 0.6)
    elbow(ax, qf_ports[3][1], (X_SF, Y_SF[1]), bend_x=X_SF - 0.6)

    elbow(ax, sf_ports[0][1], (X_CH, Y_CH), bend_x=X_CH - 0.6)
    elbow(ax, sf_ports[1][1], (X_CH, Y_CH), bend_x=X_CH - 0.6)

    # watermark
    ax.text(12, 0.75, f"Advanced CFB Ranking System • {year} Season",
            ha="center", va="center", fontsize=11, color=THEME["ink_dim"], style="italic")

    return cvs.fig


# =========================
# PUBLIC API
# =========================
def create_espn_style_bracket(final_rankings, conferences, auto_bids, year, team_info):
    if not final_rankings:
        print("Cannot create playoff bracket: no teams ranked")
        return

    all_teams = {t.id: t for t, _ in final_rankings}
    champions = detect_conference_championships(year, all_teams)

    auto_candidates, seen = [], set()
    for conf, champ in champions.items():
        score = next((s for t, s in final_rankings if t.id == champ.id), 0)
        auto_candidates.append((champ, score, "Auto-bid"))
        seen.add(conf)

    for team, score in final_rankings:
        conf = team.conference.name
        if conf not in ["FBS Independents", "Independent"] and conf not in seen:
            auto_candidates.append((team, score, "Auto-bid"))
            seen.add(conf)
        if len(seen) >= 5:
            break

    auto_candidates.sort(key=lambda x: x[1], reverse=True)
    auto_bids_list = auto_candidates[:5]
    auto_ids = {t.id for t, _, _ in auto_bids_list}

    at_large = [(t, s, "At-large") for t, s in final_rankings if t.id not in auto_ids][:7]
    field = auto_bids_list + at_large
    field.sort(key=lambda x: x[1], reverse=True)

    fig = create_professional_playoff_bracket(field, year, team_info)
    plt.savefig(f"espn_style_playoff_bracket_{year}.png", dpi=300, bbox_inches="tight",
                facecolor=THEME['bg_top'], edgecolor="none")
    print(f"Professional playoff bracket saved as 'espn_style_playoff_bracket_{year}.png'")
    return fig


def detect_conference_championships(year, teams):
    from main import fetch_data
    winners = {}
    for week in range(14, 17):
        try:
            data = fetch_data(year, week)
            for g in data:
                if not g.get("completed", False):
                    continue
                if not g.get("conferenceGame", False):
                    continue
                hc, ac = g.get("homeConference"), g.get("awayConference")
                if not hc or hc != ac:
                    continue
                hp, ap = g.get("homePoints", 0), g.get("awayPoints", 0)
                if hp == ap:
                    continue

                home_name, away_name = g.get("homeTeam"), g.get("awayTeam")
                home_t = away_t = None
                for t in teams.values():
                    if t.name == home_name and t.conference.name == hc:
                        home_t = t
                    if t.name == away_name and t.conference.name == ac:
                        away_t = t
                if not home_t or not away_t:
                    continue

                winners[hc] = home_t if hp > ap else away_t
        except Exception:
            continue
    return winners
