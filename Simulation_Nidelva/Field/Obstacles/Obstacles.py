



class Obstacle:

    def __init__(self):
        pass

    def iscollided(self, node):
        point = Point(node.location.x, node.location.y)
        line = LineString([(node.parent.location.x, node.parent.location.y),
                           (node.location.x, node.location.y)])
        collision = False
        for i in range(len(self.polygon_obstacles)):
            if self.polygon_obstacles[i].contains(point) or self.polygon_obstacles[i].intersects(line):
                # print("Collision detected")
                collision = True
        return collision

if __name__ == "__main__"
    obs = Obstacle()
