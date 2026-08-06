"""Pygame GUI renderer for CombatEnv (render_mode="human").

Decoupled from combat_env.py's core logic: this module only ever *reads*
env state, never influences step() behavior. SDL is forced onto a software
rasterizer before pygame.init() so this render path never touches CUDA or
either GPU's compute pipeline (action plan Section 2a) — at this workload
size (a handful of primitives, modest framerate) that has no perceptible
performance cost.
"""
import math
import os

os.environ["SDL_RENDER_DRIVER"] = "software"

import pygame

from envs import geometry

_MARGIN_PX = 40
_SCALE_PX_PER_UNIT = 8

_BG_COLOR = (20, 20, 25)
_BORDER_COLOR = (200, 200, 200)
_RANGE_COLOR = (90, 90, 100)
_SOLDIER_COLOR = (70, 160, 255)
_ENEMY_COLOR = (255, 90, 90)
_HP_BG_COLOR = (60, 60, 60)
_HP_FG_COLOR = (60, 220, 100)
_TRACER_COLOR = (255, 240, 120)

_state = {"screen": None, "clock": None}


def _ensure_init(env):
    if _state["screen"] is not None:
        return
    pygame.init()
    width_px = int(env.map_width * _SCALE_PX_PER_UNIT) + 2 * _MARGIN_PX
    height_px = int(env.map_height * _SCALE_PX_PER_UNIT) + 2 * _MARGIN_PX
    _state["screen"] = pygame.display.set_mode((width_px, height_px))
    pygame.display.set_caption("dqn-combat-sim")
    _state["clock"] = pygame.time.Clock()


def _world_to_screen(pos, env):
    x = _MARGIN_PX + pos[0] * _SCALE_PX_PER_UNIT
    y = _MARGIN_PX + (env.map_height - pos[1]) * _SCALE_PX_PER_UNIT  # flip so +y is "up" on screen
    return int(x), int(y)


def _draw_hp_bar(screen, center_px, hp, max_hp, width=40, height=6):
    x = center_px[0] - width // 2
    y = center_px[1] - 24
    pygame.draw.rect(screen, _HP_BG_COLOR, (x, y, width, height))
    frac = max(0.0, min(1.0, hp / max_hp)) if max_hp > 0 else 0.0
    pygame.draw.rect(screen, _HP_FG_COLOR, (x, y, int(width * frac), height))


def _draw_orientation(screen, center_px, orientation_bin, bin_size_degrees, color, length=18):
    angle_deg = geometry.bin_to_angle_center(orientation_bin, bin_size_degrees)
    angle_rad = math.radians(angle_deg)
    # screen y is flipped relative to world y (see _world_to_screen)
    end = (center_px[0] + length * math.cos(angle_rad), center_px[1] - length * math.sin(angle_rad))
    pygame.draw.line(screen, color, center_px, end, 2)


def render(env):
    _ensure_init(env)
    screen = _state["screen"]

    screen.fill(_BG_COLOR)

    top_left = _world_to_screen((0.0, env.map_height), env)
    bottom_right = _world_to_screen((env.map_width, 0.0), env)
    border_rect = pygame.Rect(
        top_left[0], top_left[1],
        bottom_right[0] - top_left[0], bottom_right[1] - top_left[1],
    )
    pygame.draw.rect(screen, _BORDER_COLOR, border_rect, 2)

    soldier_px = _world_to_screen(env.soldier["position"], env)
    enemy_px = _world_to_screen(env.enemy["position"], env)

    pygame.draw.circle(screen, _RANGE_COLOR, soldier_px, int(env.max_range * _SCALE_PX_PER_UNIT), 1)

    for shooter_pos, target_pos in env.last_hit_events:
        pygame.draw.line(
            screen, _TRACER_COLOR,
            _world_to_screen(shooter_pos, env), _world_to_screen(target_pos, env), 2,
        )

    pygame.draw.circle(screen, _SOLDIER_COLOR, soldier_px, 10)
    pygame.draw.circle(screen, _ENEMY_COLOR, enemy_px, 10)

    _draw_orientation(screen, soldier_px, env.soldier["orientation_bin"], env.bin_size_degrees, _SOLDIER_COLOR)
    _draw_orientation(screen, enemy_px, env.enemy["orientation_bin"], env.bin_size_degrees, _ENEMY_COLOR)

    _draw_hp_bar(screen, soldier_px, env.soldier["hp"], env.soldier_max_hp)
    _draw_hp_bar(screen, enemy_px, env.enemy["hp"], env.enemy_max_hp)

    pygame.display.flip()
    _state["clock"].tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            close()


def close():
    if _state["screen"] is not None:
        pygame.quit()
        _state["screen"] = None
        _state["clock"] = None
