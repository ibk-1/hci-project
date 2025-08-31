import numpy as np
from dataclasses import dataclass

GRID = 5
EMPTY, MOUSE, CHEESE, TRAP, WALL, ORGANIC = 0, 1, 2, 3, 4, 5

@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: dict

class MouseGrid:
    """
    5x5 grid. Rewards (per brief):
      +10 entering CHEESE or ORGANIC
      -50 entering TRAP
      -0.2 entering EMPTY or bumping WALL
    One element per cell.
    Observation: 6x5x5 one-hot planes [EMPTY,MOUSE,CHEESE,TRAP,WALL,ORGANIC].
    Actions: 0=up,1=down,2=left,3=right.
    """
    def __init__(self, n_traps=2, n_walls=2, n_cheese=1, n_org=1, rng=None, max_steps=40):
        self.n_traps, self.n_walls, self.n_cheese, self.n_org = n_traps, n_walls, n_cheese, n_org
        self.rng = np.random.RandomState(None if rng is None else rng)
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.grid = np.zeros((GRID, GRID), dtype=np.int32)
        # place mouse
        self.mouse = self._place(MOUSE)
        # place walls, traps, cheeses
        for _ in range(self.n_walls): self._place(WALL)
        for _ in range(self.n_traps): self._place(TRAP)
        self.cheeses = [self._place(CHEESE) for _ in range(self.n_cheese)]
        self.orgs    = [self._place(ORGANIC) for _ in range(self.n_org)]
        self.t       = 0
        return self.obs()

    def _place(self, val):
        while True:
            r, c = self.rng.randint(0, GRID), self.rng.randint(0, GRID)
            if self.grid[r, c] == EMPTY:
                self.grid[r, c] = val
                return (r, c)

    def obs(self):
        planes = np.zeros((6, GRID, GRID), dtype=np.float32)
        for r in range(GRID):
            for c in range(GRID):
                planes[self.grid[r,c], r, c] = 1.0
        return planes

    def step(self, a: int) -> StepResult:
        self.t += 1
        drc = [(-1,0),(1,0),(0,-1),(0,1)][a]
        r, c = self.mouse
        nr, nc = r + drc[0], c + drc[1]

        bumped = False
        if not (0 <= nr < GRID and 0 <= nc < GRID) or self.grid[nr, nc] == WALL:
            nr, nc = r, c
            bumped = True

        # move mouse
        self.grid[r, c] = EMPTY if (r, c) not in self.cheeses and (r, c) not in self.orgs else self.grid[r, c]
        self.mouse = (nr, nc)
        # reward
        cell = self.grid[nr, nc]
        if cell in (CHEESE, ORGANIC):
            rew = 10.0
        elif cell == TRAP:
            rew = -50.0
        else:
            rew = -0.2 if bumped or cell == EMPTY else -0.2

        done = (cell in (CHEESE, ORGANIC, TRAP)) or (self.t >= self.max_steps)
        # once eaten / hit trap, clear cell
        if cell in (CHEESE, ORGANIC, TRAP):
            self.grid[nr, nc] = MOUSE
        else:
            self.grid[nr, nc] = MOUSE

        # inside step(), at the end
        return StepResult(self.obs(), rew, done, {"cell": int(cell), "t": self.t, "bumped": bool(bumped)})

    def render_ascii(self):
        char = {EMPTY:".", MOUSE:"M", CHEESE:"C", TRAP:"X", WALL:"#", ORGANIC:"O"}
        lines=[]
        for r in range(GRID):
            lines.append(" ".join(char[self.grid[r,c]] for c in range(GRID)))
        return "\n".join(lines)
