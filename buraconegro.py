import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Markers, Mesh, Line
from vispy.geometry import create_sphere
from vispy import app


class BlackHoleSimulator(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Buraco Negro Cinemático Épico")
        self.resize(900, 700)

        layout = QtWidgets.QHBoxLayout(self)
        self.canvas = SceneCanvas(keys='interactive', bgcolor='black', parent=self, size=(700, 700))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        layout.addWidget(self.canvas.native)

        # Painel de controles
        control_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(control_panel)
        control_panel.addWidget(QtWidgets.QLabel("Velocidade da animação"))
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(20)
        control_panel.addWidget(self.speed_slider)
        self.reset_btn = QtWidgets.QPushButton("Resetar Câmera")
        control_panel.addWidget(self.reset_btn)
        control_panel.addStretch()

        # Buraco negro
        sphere_data = create_sphere(radius=0.5, cols=40, rows=40)
        self.blackhole = Mesh(meshdata=sphere_data, color=(0, 0, 0, 1), parent=self.view.scene)

        # Horizonte de eventos
        self.event_horizon_radius = 0.8
        theta = np.linspace(0, 2 * np.pi, 300)
        x = self.event_horizon_radius * np.cos(theta)
        y = self.event_horizon_radius * np.sin(theta)
        z = np.zeros_like(theta)
        ring_points = np.c_[x, y, z]
        self.event_horizon = Line(pos=ring_points, color=(1, 0.5, 0, 1), width=4, method="gl", parent=self.view.scene)

        # Partículas
        self.num_particles = 400
        self.base_radius = 2.5
        self.min_radius = 0.6
        self.history_length = 25

        self.particle_angles = np.random.rand(self.num_particles) * 2 * np.pi
        self.particle_speeds = 0.5 + 1.0 * np.random.rand(self.num_particles)
        self.particle_radii = self.base_radius * (0.7 + 0.3 * np.random.rand(self.num_particles))

        self.particle_history = np.zeros((self.num_particles, self.history_length, 3), dtype=np.float32)
        self.particle_markers = Markers(parent=self.view.scene)

        # Partículas capturadas
        self.captured_particles = np.zeros(self.num_particles, dtype=bool)
        self.capture_progress = np.zeros(self.num_particles)

        # Variáveis
        self.time = 0
        self.speed = 0.02

        # Primeira atualização
        self.update_particles(0)

        # Eventos
        self.speed_slider.valueChanged.connect(self.on_speed_change)
        self.reset_btn.clicked.connect(self.reset_camera)

        # Timer
        self.timer = app.Timer('auto', connect=self.on_timer, start=True)

    def update_particles(self, dt):
        self.time += dt * self.speed

        # Atualiza ângulo e raio
        self.particle_angles = (self.particle_angles + self.particle_speeds * dt * 2) % (2 * np.pi)
        self.particle_radii -= dt * self.speed * 0.25

        # Determina partículas capturadas
        near_horizon = self.particle_radii < self.event_horizon_radius * 1.5
        self.captured_particles = near_horizon

        # Incrementa progresso da captura
        self.capture_progress[self.captured_particles] += dt * 0.8
        self.capture_progress[~self.captured_particles] *= 0.9
        self.capture_progress = np.clip(self.capture_progress, 0, 1)

        # Efeito de distorção gravitacional (atração para o centro)
        grav_pull = 0.05 + 0.15 * self.capture_progress
        self.particle_radii -= grav_pull * dt

        # Coordenadas cartesianas com efeito de flare e alongamento
        flare = 1 + 2 * self.capture_progress
        x = self.particle_radii * np.cos(self.particle_angles) * flare
        y = self.particle_radii * np.sin(self.particle_angles) * flare
        z = 0.1 * np.sin(5 * self.particle_angles + self.time * 5) * (1 + self.capture_progress)

        # Atualiza histórico
        self.particle_history = np.roll(self.particle_history, shift=-1, axis=1)
        self.particle_history[:, -1, 0] = x
        self.particle_history[:, -1, 1] = y
        self.particle_history[:, -1, 2] = z

        # Reinicia partículas absorvidas
        reset_mask = self.particle_radii < self.min_radius
        self.particle_radii[reset_mask & (self.capture_progress < 1)] = self.base_radius
        self.particle_angles[reset_mask & (self.capture_progress < 1)] = np.random.rand(np.sum(reset_mask & (self.capture_progress < 1))) * 2 * np.pi
        self.capture_progress[reset_mask] = 0

        # Renderiza partículas
        positions = self.particle_history.reshape(-1, 3)

        # Cores do rastro
        t = np.linspace(0, 1, self.history_length)
        trail_colors = np.zeros((self.history_length, 4))
        trail_colors[:, 0] = 1.0
        trail_colors[:, 1] = 1.0 - t
        trail_colors[:, 2] = 0.0
        trail_colors[:, 3] = t
        colors = np.tile(trail_colors, (self.num_particles, 1))

        # Intensidade faíscas e flare (corrigido)
        capture_vals = np.repeat(self.capture_progress, self.history_length)[np.repeat(self.captured_particles, self.history_length)]
        intensity_mask = np.repeat(self.captured_particles, self.history_length)
        colors[intensity_mask, 0] = 1.0
        colors[intensity_mask, 1] = 0.2 + 0.8 * capture_vals
        colors[intensity_mask, 2] = 0.0

        # Desaparecimento completo
        disappeared_mask = np.repeat(self.capture_progress >= 1, self.history_length)
        colors[disappeared_mask, 3] *= 0.0

        # Tamanho das partículas aumenta com flare
        sizes = 5 + 4 * np.repeat(self.capture_progress, self.history_length)

        self.particle_markers.set_data(positions, face_color=colors, size=sizes)

        # Horizonte pulsante
        pulse = 0.5 + 0.5 * np.sin(self.time * 5)
        self.event_horizon.set_data(color=(1, 0.5 * pulse, 0, 1))

    def on_timer(self, event):
        self.update_particles(event.dt)
        self.canvas.update()

    def on_speed_change(self, value):
        self.speed = value / 1000.0

    def reset_camera(self):
        self.view.camera.set_range()


if __name__ == '__main__':
    appQt = QtWidgets.QApplication(sys.argv)
    sim = BlackHoleSimulator()
    sim.show()
    sys.exit(appQt.exec_())
