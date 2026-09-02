/**
 * RRT* over the cost valley.
 *
 * Port of `src/planners/RRTSCV/RRTStarCV.py` and `TreeNode.py`. The tree is
 * grown from the AUV's current location toward the minimum of the cost valley;
 * branch cost is the normalised travel distance plus the cost-valley integral
 * along the branch, so the tree prefers routes through informative water rather
 * than the shortest line. The first step of the resulting trajectory is the
 * next waypoint.
 *
 * The Python version draws its samples from a precomputed
 * `RRT_Random_Locations.npy`; here they are drawn on the fly from the seeded
 * generator, rejecting anything outside the operational area.
 */
import type { Field } from "./field.js";
import type { CostValley } from "./costValley.js";
import type { Random } from "./random.js";

export interface TreeNode {
  x: number;
  y: number;
  cost: number;
  parent: number; // index into the node array, -1 for the root
}

export interface RrtParameters {
  /** Distance between consecutive AUV waypoints, in metres. */
  stepsize: number;
  /** Probability of sampling the target instead of a random point. */
  goalSamplingRate: number;
  /** Number of tree-expansion iterations per planning call. */
  maxIterations: number;
  /** Radius within which a new node counts as having reached the target. */
  homeRadius: number;
  /** Rewiring neighbourhood radius. */
  neighbourRadius: number;
}

export function defaultRrtParameters(waypointDistance: number): RrtParameters {
  return {
    stepsize: waypointDistance,
    goalSamplingRate: 0.01,
    maxIterations: 1200,
    homeRadius: waypointDistance * 0.8,
    neighbourRadius: waypointDistance * 1.12,
  };
}

export class RRTStarCV {
  nodes: TreeNode[] = [];
  trajectory: number[][] = [];
  costTrajectory = 0;

  constructor(
    private readonly field: Field,
    readonly costValley: CostValley,
    readonly params: RrtParameters,
    private readonly random: Random,
  ) {}

  /** True if a point is inside the operational area and clear of the obstacle. */
  isLocationLegal(x: number, y: number): boolean {
    return !this.field.obstacleContains(x, y);
  }

  /** True if the straight segment crosses neither the border nor the obstacle. */
  isPathLegal(ax: number, ay: number, bx: number, by: number): boolean {
    return (
      !this.field.isBorderInTheWay(ax, ay, bx, by) &&
      !this.field.isObstacleInTheWay(ax, ay, bx, by)
    );
  }

  /**
   * Plan from `loc_start` toward `loc_target` and return the next waypoint,
   * one step size along the trajectory the tree found.
   */
  getNextWaypoint(
    startX: number, startY: number,
    targetX: number, targetY: number,
  ): [number, number] {
    this.expandTrees(startX, startY, targetX, targetY);
    this.buildShortestTrajectory(startX, startY);

    const path = this.trajectory;
    let nextX: number;
    let nextY: number;
    if (path.length <= 2) {
      nextX = targetX;
      nextY = targetY;
    } else {
      nextX = path[path.length - 2][0];
      nextY = path[path.length - 2][1];
    }

    const angle = Math.atan2(nextX - startX, nextY - startY);
    let wpX = startX + this.params.stepsize * Math.sin(angle);
    let wpY = startY + this.params.stepsize * Math.cos(angle);

    // Fall back to any legal heading if the chosen step is not viable.
    if (!this.isLocationLegal(wpX, wpY) || !this.isPathLegal(startX, startY, wpX, wpY)) {
      for (let i = 0; i < 60; i++) {
        const a = (2 * Math.PI * i) / 59;
        const cx = startX + this.params.stepsize * Math.cos(a);
        const cy = startY + this.params.stepsize * Math.sin(a);
        if (this.isLocationLegal(cx, cy) && this.isPathLegal(startX, startY, cx, cy)) {
          wpX = cx;
          wpY = cy;
          break;
        }
      }
    }
    return [wpX, wpY];
  }

  private targetNode: TreeNode = { x: 0, y: 0, cost: 0, parent: -1 };

  private expandTrees(startX: number, startY: number, targetX: number, targetY: number): void {
    const { stepsize, goalSamplingRate, maxIterations, homeRadius, neighbourRadius } = this.params;
    const nodes: TreeNode[] = [{ x: startX, y: startY, cost: 0, parent: -1 }];
    this.nodes = nodes;
    this.targetNode = { x: targetX, y: targetY, cost: 0, parent: -1 };

    const [xmin, xmax] = this.field.xlim;
    const [ymin, ymax] = this.field.ylim;

    for (let iter = 0; iter < maxIterations; iter++) {
      // s1: sample a location, occasionally the goal itself.
      let sx: number;
      let sy: number;
      if (this.random.uniform() <= goalSamplingRate) {
        sx = targetX;
        sy = targetY;
      } else {
        let tries = 0;
        do {
          sx = this.random.range(xmin, xmax);
          sy = this.random.range(ymin, ymax);
          tries++;
        } while (
          tries < 30 &&
          (!this.field.borderContains(sx, sy) || this.field.obstacleContains(sx, sy))
        );
        if (!this.field.borderContains(sx, sy) || this.field.obstacleContains(sx, sy)) continue;
      }

      // s2: nearest existing node.
      let nearest = 0;
      let nearestDist = Infinity;
      for (let i = 0; i < nodes.length; i++) {
        const dx = nodes[i].x - sx;
        const dy = nodes[i].y - sy;
        const d = dx * dx + dy * dy;
        if (d < nearestDist) {
          nearestDist = d;
          nearest = i;
        }
      }

      // s3: steer at most one step size toward the sample.
      let nx = sx;
      let ny = sy;
      if (Math.sqrt(nearestDist) > stepsize) {
        const base = nodes[nearest];
        const angle = Math.atan2(sx - base.x, sy - base.y);
        nx = base.x + stepsize * Math.sin(angle);
        ny = base.y + stepsize * Math.cos(angle);
      }

      // s4: reject a colliding node.
      if (!this.isLocationLegal(nx, ny)) continue;

      // s5: choose the cheapest parent among the neighbourhood, then rewire it.
      let parent = nearest;
      let bestCost = this.costBetween(nodes[nearest], nx, ny);
      const neighbours: number[] = [];
      for (let i = 0; i < nodes.length; i++) {
        const dx = nodes[i].x - nx;
        const dy = nodes[i].y - ny;
        if (dx * dx + dy * dy <= neighbourRadius * neighbourRadius) {
          neighbours.push(i);
          const c = this.costBetween(nodes[i], nx, ny);
          if (c < bestCost) {
            bestCost = c;
            parent = i;
          }
        }
      }
      const newNode: TreeNode = { x: nx, y: ny, cost: bestCost, parent };

      // s6: reject the branch if the segment is not traversable.
      if (!this.isPathLegal(nodes[parent].x, nodes[parent].y, nx, ny)) continue;

      // s7: connect to the target if we are close enough, otherwise keep growing.
      const dxg = nx - targetX;
      const dyg = ny - targetY;
      if (Math.sqrt(dxg * dxg + dyg * dyg) < homeRadius) {
        const index = nodes.length;
        nodes.push(newNode);
        this.targetNode.parent = index;
        this.targetNode.cost = this.costBetween(newNode, targetX, targetY);
        continue;
      }
      const index = nodes.length;
      nodes.push(newNode);

      // Rewire the neighbourhood through the new node where that is cheaper.
      for (const i of neighbours) {
        const c = this.costBetween(newNode, nodes[i].x, nodes[i].y);
        if (c < nodes[i].cost && this.isPathLegal(nx, ny, nodes[i].x, nodes[i].y)) {
          nodes[i].cost = c;
          nodes[i].parent = index;
        }
      }
    }
  }

  /** Cost of reaching (x, y) through `from`: parent cost + travel + valley. */
  private costBetween(from: TreeNode, x: number, y: number): number {
    const dx = from.x - x;
    const dy = from.y - y;
    const costDistance = Math.sqrt(dx * dx + dy * dy) / this.params.stepsize;
    const costValley = this.costValley.costAlongPath(from.x, from.y, x, y);
    return from.cost + costDistance + costValley;
  }

  /**
   * Walk back from the target to the root, as `__get_shortest_trajectory`, so
   * the result reads target-first and start-last:
   * `[target, node_k, ..., node_1, start]`.
   *
   * The root itself is deliberately not pushed by the loop - `startX, startY`
   * closes the path - which is what keeps `trajectory[length - 2]` the first
   * real step away from the start rather than the start over again.
   */
  private buildShortestTrajectory(startX: number, startY: number): void {
    const trajectory: number[][] = [[this.targetNode.x, this.targetNode.y]];
    this.costTrajectory = this.targetNode.cost;

    let pointer = this.targetNode.parent;
    let guard = 0;
    while (pointer >= 0 && this.nodes[pointer].parent >= 0 && guard++ <= this.params.maxIterations) {
      const node = this.nodes[pointer];
      trajectory.push([node.x, node.y]);
      pointer = node.parent;
    }
    trajectory.push([startX, startY]);
    this.trajectory = trajectory;
  }
}
