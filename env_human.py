import gym
import gym_sokoban 
import numpy as np

# Create the environment
import cv2

# map keys to Sokoban “push” actions:
# 1 = Push Up, 2 = Push Down, 3 = Push Left, 4 = Push Right, 5 = Move Up, 6 = Move Down, 7 = Move Left, 8 = Move Right
ACTION_MAP = {
    ord('w'): 1, 
    ord('s'): 2,
    ord('a'): 3,
    ord('d'): 4,
    ord('j'): 5,
    ord('m'): 6,
    ord('n'): 7,
    ord(','): 8,
}

def main():
    # choose any registered Sokoban size, e.g. Sokoban-v0, Sokoban-small-v0, etc.
    env = gym.make('Sokoban-large-v0')

    obs = env.reset()
    done = False

    while True:
        # render an RGB array (shape H×W×3)
        frame = env.render(mode='rgb_array')
        # ensure frame is a NumPy array of type uint8
        frame = np.array(frame, dtype=np.uint8)
        # show it in a window
        cv2.imshow('gym-sokoban (WASD to move, Q to quit)', frame)

        # wait indefinitely for a key press
        key = cv2.waitKey(0) & 0xFF
        # quit on 'q'
        if key == ord('q'):
            break

        # look up the action; ignore any other keys
        action = ACTION_MAP.get(key)
        if action is None:
            continue

        # step the environment
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result

        # if the level is finished or failed, reset
        if done:
            obs = env.reset()

    # clean up
    env.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()