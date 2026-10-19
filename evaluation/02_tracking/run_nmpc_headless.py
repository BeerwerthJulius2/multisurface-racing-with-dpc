"""Run NMPC.py main() unchanged with rendering disabled (batch runs; from the repository root)."""
import os, sys
sys.path.insert(0, os.getcwd())
import f110_gym.envs.f110_env as fe
fe.F110Env.render = lambda self, *a, **k: None
fe.F110Env.add_render_callback = lambda self, *a, **k: None
import NMPC
NMPC.main()
