import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20


class TreePlotter:
    __nodes = None

    def update_trees(self, nodes):
        self.__nodes = nodes

    def plot_tree(self):
        for node in self.__nodes:
            if node.get_parent() is not None:
                loc = node.get_location()
                loc_p = node.get_parent().get_location()
                plt.plot([loc[0], loc_p[0]],
                         [loc[1], loc_p[1]], "-g")

