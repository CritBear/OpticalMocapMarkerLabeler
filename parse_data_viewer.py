import numpy as np

class Viewer:
    def __init__(self):
        print('Enter file path : ', end='')
        self.file_path = input()

        self.markers = []
        self.edges = []

        # To recognize hand easily, render edge connecting markers
        self.edge_configs = [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 6], [7, 8], [8, 9], [10, 11], [11, 12], [13, 14], [14, 15], [16, 17], [17, 18]]

    def run(self):
        import vpython.no_notebook
        import vpython

        window = vpython.canvas(x=0, y=0, width=1200, height=1200, title='Parsing data Viewer',
                                center=vpython.vector(0, 0, 0), background=vpython.vector(0, 0, 0))

        axis_x = vpython.curve(pos=[(0, 0, 0), (100, 0, 0)], color=vpython.color.red)
        axis_y = vpython.curve(pos=[(0, 0, 0), (0, 100, 0)], color=vpython.color.green)
        axis_z = vpython.curve(pos=[(0, 0, 0), (0, 0, 100)], color=vpython.color.blue)

        frame_data = np.load(self.file_path)

        self.init_objects(vpython)

        print(frame_data.shape)
        for frame in frame_data:
            self.update(vpython, frame)

            vpython.sleep(0.01)

    def init_objects(self, vpython):
        for _ in range(19):
            self.markers.append(vpython.sphere(radius=5, color=vpython.color.white))

        # paint backhand markers
        self.markers[16].color = vpython.color.red
        self.markers[17].color = vpython.color.green
        self.markers[18].color = vpython.color.blue

        for _ in self.edge_configs:
            self.edges.append(vpython.cylinder(radius=0.2, color=vpython.color.yellow))

    def update(self, vpython, frame):
        for m_idx, m_pos in enumerate(frame):
            self.markers[m_idx].pos = vpython.vector(m_pos[0], m_pos[1], m_pos[2])

        for e_idx, e_config in enumerate(self.edge_configs):
            self.edges[e_idx].pos = self.markers[e_config[0]].pos
            self.edges[e_idx].axis = self.markers[e_config[1]].pos - self.markers[e_config[0]].pos
