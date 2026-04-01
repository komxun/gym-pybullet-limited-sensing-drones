"""Test script to verify all drone actions (accelerate, decelerate, constant) work correctly.

Usage:
    python -m training.test_actions
    python -m training.test_actions --gui        # with PyBullet GUI
    python -m training.test_actions --steps 50   # custom step count per action
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_pybullet_drones.envs.AutoroutingSARLAviary import AutoroutingSARLAviary


def run_action_test(gui=False, steps_per_action=30):
    """Run each action for N steps and report speed/position behavior."""

    env = AutoroutingSARLAviary(num_drones=3, gui=gui)
    results = {}

    ACTION_NAMES = {0: "ACCELERATE", 1: "DECELERATE", 2: "CONSTANT"}

    # ----------------------------------------------------------------
    # Test 1: ACCELERATE (action 0)
    # ----------------------------------------------------------------
    print("=" * 60)
    print("TEST 1: ACCELERATE (action=0)")
    print("=" * 60)
    obs, info = env.reset()
    speeds = []
    positions = []
    for step in range(steps_per_action):
        state = env._getDroneStateVector(0)
        speed = np.linalg.norm(state[10:13])
        target_speed = np.linalg.norm(env.routing[0].TARGET_VEL)
        pos = state[0:3].copy()
        speeds.append(speed)
        positions.append(pos)
        if step % 5 == 0:
            print(f"  step {step:3d} | pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}) "
                  f"| actual_speed={speed:.3f} | target_speed={target_speed:.3f}")
        obs, r, term, trunc, info = env.step(0)
        if term or trunc:
            print(f"  Episode ended at step {step} (term={term}, trunc={trunc})")
            break

    accel_ok = speeds[-1] > speeds[0]
    print(f"\n  RESULT: speed went from {speeds[0]:.3f} -> {speeds[-1]:.3f}")
    print(f"  {'PASS' if accel_ok else 'FAIL'}: Speed should increase\n")
    results["ACCELERATE"] = accel_ok

    # ----------------------------------------------------------------
    # Test 2: DECELERATE (action 1) — first accelerate, then decelerate
    # ----------------------------------------------------------------
    print("=" * 60)
    print("TEST 2: DECELERATE (action=1) after acceleration")
    print("=" * 60)
    obs, info = env.reset()

    # Accelerate first
    print("  Phase 1: Accelerating for 15 steps...")
    for _ in range(15):
        obs, r, term, trunc, info = env.step(0)
        if term or trunc:
            obs, info = env.reset()

    state = env._getDroneStateVector(0)
    speed_before_decel = np.linalg.norm(state[10:13])
    print(f"  Speed before deceleration: {speed_before_decel:.3f}")

    # Now decelerate
    print("  Phase 2: Decelerating...")
    speeds = []
    target_speeds = []
    for step in range(steps_per_action):
        state = env._getDroneStateVector(0)
        speed = np.linalg.norm(state[10:13])
        target_speed = np.linalg.norm(env.routing[0].TARGET_VEL)
        speeds.append(speed)
        target_speeds.append(target_speed)
        if step % 5 == 0:
            print(f"  step {step:3d} | actual_speed={speed:.3f} | target_speed={target_speed:.3f}")
        obs, r, term, trunc, info = env.step(1)
        if term or trunc:
            print(f"  Episode ended at step {step} (term={term}, trunc={trunc})")
            break

    decel_ok = speeds[-1] < speed_before_decel
    no_negative = all(ts >= 0 for ts in target_speeds)
    print(f"\n  RESULT: speed went from {speed_before_decel:.3f} -> {speeds[-1]:.3f}")
    print(f"  {'PASS' if decel_ok else 'FAIL'}: Speed should decrease")
    print(f"  {'PASS' if no_negative else 'FAIL'}: Target speed should never be negative (no reversal)\n")
    results["DECELERATE"] = decel_ok and no_negative

    # ----------------------------------------------------------------
    # Test 3: CONSTANT (action 2) — accelerate, then hold constant speed
    # ----------------------------------------------------------------
    print("=" * 60)
    print("TEST 3: CONSTANT (action=2) — maintain speed")
    print("=" * 60)
    obs, info = env.reset()

    # Accelerate first
    print("  Phase 1: Accelerating for 15 steps...")
    for _ in range(15):
        obs, r, term, trunc, info = env.step(0)
        if term or trunc:
            obs, info = env.reset()

    state = env._getDroneStateVector(0)
    speed_before_const = np.linalg.norm(state[10:13])
    print(f"  Speed before constant: {speed_before_const:.3f}")

    # Now hold constant
    print("  Phase 2: Constant velocity...")
    speeds = []
    target_speeds = []
    for step in range(steps_per_action):
        state = env._getDroneStateVector(0)
        speed = np.linalg.norm(state[10:13])
        target_speed = np.linalg.norm(env.routing[0].TARGET_VEL)
        speeds.append(speed)
        target_speeds.append(target_speed)
        if step % 5 == 0:
            print(f"  step {step:3d} | actual_speed={speed:.3f} | target_speed={target_speed:.3f}")
        obs, r, term, trunc, info = env.step(2)
        if term or trunc:
            print(f"  Episode ended at step {step} (term={term}, trunc={trunc})")
            break

    # Target speed should stay roughly constant (not accelerating or decelerating)
    # Skip the first few readings — the constant command takes effect after 1 step
    settled_speeds = target_speeds[2:] if len(target_speeds) > 2 else target_speeds
    if len(settled_speeds) >= 2:
        speed_variation = max(settled_speeds) - min(settled_speeds)
    else:
        speed_variation = 0.0
    const_ok = speed_variation < 0.5  # settled target speed shouldn't vary much
    print(f"\n  RESULT: settled target speed range = [{min(settled_speeds):.3f}, {max(settled_speeds):.3f}], variation={speed_variation:.3f}")
    print(f"  {'PASS' if const_ok else 'FAIL'}: Target speed should remain roughly constant (after settling)\n")
    results["CONSTANT"] = const_ok

    # ----------------------------------------------------------------
    # Test 4: Action sequence — accel -> decel -> constant -> accel
    # ----------------------------------------------------------------
    print("=" * 60)
    print("TEST 4: Mixed action sequence (accel->decel->constant->accel)")
    print("=" * 60)
    obs, info = env.reset()
    sequence = [0]*10 + [1]*10 + [2]*10 + [0]*10
    seq_ok = True
    for step, action in enumerate(sequence):
        state = env._getDroneStateVector(0)
        speed = np.linalg.norm(state[10:13])
        target_speed = np.linalg.norm(env.routing[0].TARGET_VEL)
        if step % 10 == 0:
            print(f"  step {step:3d} | action={ACTION_NAMES[action]:12s} "
                  f"| speed={speed:.3f} | target_speed={target_speed:.3f}")
        try:
            obs, r, term, trunc, info = env.step(action)
            if term or trunc:
                print(f"  Episode ended at step {step}")
                obs, info = env.reset()
        except Exception as e:
            print(f"  FAIL at step {step}, action={action}: {e}")
            seq_ok = False
            break

    print(f"\n  {'PASS' if seq_ok else 'FAIL'}: Mixed sequence completed without errors\n")
    results["MIXED_SEQUENCE"] = seq_ok

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    env.close()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s}: {status}")
        if not passed:
            all_pass = False

    print(f"\n  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test drone GNC actions")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI")
    parser.add_argument("--steps", type=int, default=30, help="Steps per action test")
    args = parser.parse_args()

    success = run_action_test(gui=args.gui, steps_per_action=args.steps)
    sys.exit(0 if success else 1)
