"""Pygame GUI renderer for CombatEnv (render_mode="human").

Decoupled from combat_env.py's core logic: this module only ever *reads*
env state, never influences step() behavior. SDL is forced onto a software
rasterizer before pygame.init() so this render path never competes with
DQNAgent for the GPU (which trains on CUDA) — at this workload size (a
handful of primitives, modest framerate) that has no perceptible
performance cost.

All layout/color/timing constants come from a render_config dict (see
the `render:` section of config/sim_default.yaml for the schema) rather than being
hardcoded here. The config is captured once, on the first render() call of a
process, since it determines the fixed window size.
"""
import os

os.environ["SDL_RENDER_DRIVER"] = "software"

import math

import pygame

from envs import geometry

_state = {"screen": None, "clock": None, "config": None, "font": None}


def _ensure_init(env, config):
    if _state["screen"] is not None:
        return
    pygame.init()
    _state["config"] = config
    margin = config["window"]["margin_px"]
    scale = config["window"]["scale_px_per_unit"]
    width_px = int(env.map_width * scale) + 2 * margin
    height_px = int(env.map_height * scale) + 2 * margin
    _state["screen"] = pygame.display.set_mode((width_px, height_px))
    pygame.display.set_caption("dqn-combat-sim")
    _state["clock"] = pygame.time.Clock()
    _state["font"] = pygame.font.Font(None, config["stats_text"]["font_size_px"])


def _world_to_screen(pos, env):
    config = _state["config"]
    margin = config["window"]["margin_px"]
    scale = config["window"]["scale_px_per_unit"]
    x = margin + pos[0] * scale
    y = margin + (env.map_height - pos[1]) * scale  # flip so +y is "up" on screen
    return int(x), int(y)


def _point_at_angle(center_px, angle_deg, length_px):
    angle_rad = math.radians(angle_deg)
    # screen y is flipped relative to world y (see _world_to_screen)
    return (center_px[0] + length_px * math.cos(angle_rad), center_px[1] - length_px * math.sin(angle_rad))


def _draw_hp_bar(screen, center_px, hp, max_hp, config):
    width = config["hp_bar"]["width_px"]
    height = config["hp_bar"]["height_px"]
    x = center_px[0] - width // 2
    y = center_px[1] - 24
    pygame.draw.rect(screen, config["colors"]["hp_bar_background"], (x, y, width, height))
    frac = max(0.0, min(1.0, hp / max_hp)) if max_hp > 0 else 0.0
    pygame.draw.rect(screen, config["colors"]["hp_bar_foreground"], (x, y, int(width * frac), height))


def _draw_orientation(screen, center_px, orientation_bin, bin_size_degrees, color, config):
    length = config["orientation_indicator"]["length_px"]
    angle_deg = geometry.bin_to_angle_center(orientation_bin, bin_size_degrees)
    end = _point_at_angle(center_px, angle_deg, length)
    pygame.draw.line(screen, color, center_px, end, 2)


def _draw_range_wedge(screen, center_px, orientation_bin, bin_size_degrees, radius_px, color):
    """Outline of the pie slice the entity's current orientation bin covers, out to gun
    range — replaces a full range circle with just the bin-width slice actually being aimed."""
    angle_center = geometry.bin_to_angle_center(orientation_bin, bin_size_degrees)
    lo = angle_center - bin_size_degrees / 2.0
    hi = angle_center + bin_size_degrees / 2.0
    n_steps = max(2, int(bin_size_degrees / 6))  # ~6 degrees per arc segment
    arc_points = [
        _point_at_angle(center_px, lo + (hi - lo) * i / n_steps, radius_px)
        for i in range(n_steps + 1)
    ]
    pygame.draw.line(screen, color, center_px, arc_points[0], 1)
    pygame.draw.line(screen, color, center_px, arc_points[-1], 1)
    pygame.draw.lines(screen, color, False, arc_points, 1)


def _draw_fire_events(screen, env, config, radius_px):
    """Every fire attempt this tick. The soldier's are always drawn — colored by whether it
    was aimed at the enemy's true bearing, independent of whether the accuracy roll landed —
    so a miss from bad aim reads differently from a miss from bad luck. The enemy (which
    always aims correctly by design) still only shows its tracer on an actual hit."""
    colors = config["colors"]
    for fire_event in env.last_fire_events:
        start_px = _world_to_screen(fire_event["position"], env)
        if fire_event["shooter"] == "soldier":
            color = colors["fire_correct"] if fire_event["facing_target"] else colors["fire_incorrect"]
            if fire_event["hit"]:
                end_px = _world_to_screen(fire_event["target_position"], env)
            else:
                angle_deg = geometry.bin_to_angle_center(fire_event["orientation_bin"], env.bin_size_degrees)
                end_px = _point_at_angle(start_px, angle_deg, radius_px)
            pygame.draw.line(screen, color, start_px, end_px, 2)
        elif fire_event["hit"]:
            end_px = _world_to_screen(fire_event["target_position"], env)
            pygame.draw.line(screen, colors["tracer"], start_px, end_px, 2)


def _draw_stats_text(screen, center_px, hp, max_hp, ammo, config):
    text = f"HP {int(round(hp))}/{int(max_hp)}  AMMO {ammo}"
    surface = _state["font"].render(text, True, config["colors"]["stats_text"])
    rect = surface.get_rect(center=(center_px[0], center_px[1] - 36))
    screen.blit(surface, rect)


def render(env, config):
    _ensure_init(env, config)
    screen = _state["screen"]
    config = _state["config"]  # config is locked in at init time since it determines window size
    colors = config["colors"]
    scale = config["window"]["scale_px_per_unit"]
    range_px = env.max_range * scale

    screen.fill(colors["background"])

    top_left = _world_to_screen((0.0, env.map_height), env)
    bottom_right = _world_to_screen((env.map_width, 0.0), env)
    border_rect = pygame.Rect(
        top_left[0], top_left[1],
        bottom_right[0] - top_left[0], bottom_right[1] - top_left[1],
    )
    pygame.draw.rect(screen, colors["border"], border_rect, 2)

    soldier_px = _world_to_screen(env.soldier["position"], env)
    enemy_px = _world_to_screen(env.enemy["position"], env)

    _draw_range_wedge(screen, soldier_px, env.soldier["orientation_bin"], env.bin_size_degrees, range_px, colors["range_wedge_soldier"])
    _draw_range_wedge(screen, enemy_px, env.enemy["orientation_bin"], env.bin_size_degrees, range_px, colors["range_wedge_enemy"])

    _draw_fire_events(screen, env, config, range_px)

    pygame.draw.circle(screen, colors["soldier"], soldier_px, 10)
    pygame.draw.circle(screen, colors["enemy"], enemy_px, 10)

    _draw_orientation(screen, soldier_px, env.soldier["orientation_bin"], env.bin_size_degrees, colors["soldier"], config)
    _draw_orientation(screen, enemy_px, env.enemy["orientation_bin"], env.bin_size_degrees, colors["enemy"], config)

    _draw_hp_bar(screen, soldier_px, env.soldier["hp"], env.soldier_max_hp, config)
    _draw_hp_bar(screen, enemy_px, env.enemy["hp"], env.enemy_max_hp, config)

    _draw_stats_text(screen, soldier_px, env.soldier["hp"], env.soldier_max_hp, env.soldier["ammo"], config)
    _draw_stats_text(screen, enemy_px, env.enemy["hp"], env.enemy_max_hp, env.enemy["ammo"], config)

    pygame.display.flip()
    _state["clock"].tick(config["window"]["fps"])

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            close()


def get_frame():
    """Last-rendered frame as an (H, W, 3) uint8 RGB array, or None if render() hasn't
    been called yet (or close() was already called). For video/GIF recording."""
    if _state["screen"] is None:
        return None
    return pygame.surfarray.array3d(_state["screen"]).transpose(1, 0, 2)


def close():
    if _state["screen"] is not None:
        pygame.quit()
        _state["screen"] = None
        _state["clock"] = None
        _state["config"] = None
        _state["font"] = None
