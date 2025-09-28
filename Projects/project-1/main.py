import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass

@dataclass
class RigidBody:
    initial_position: float
    initial_velocity: float
    initial_acceleration: float
    mass: float
    color: str
    name: str

    stationary: bool = False 
    
    time:         np.array = None
    position:     np.array = None
    velocity:     np.array = None
    acceleration: np.array = None
    energy:       np.array = None
    momentum:     np.array = None
    
    def init(self, n):
        self.position     = np.zeros(n)
        self.velocity     = np.zeros(n)
        self.acceleration = np.zeros(n)
        self.energy       = np.zeros(n)
        self.momentum     = np.zeros(n)
        
        self.position[0]     = self.initial_position
        self.velocity[0]     = self.initial_velocity
        self.acceleration[0] = self.initial_acceleration
        
    def __hash__(self):
        return hash(self.name)
        
    def distance_to(self, other_body, i):
        return np.sqrt(np.sum((self.position[i] - other_body.position[i])**2))

    def collided_with(self, other_body, collision_radius, i):
        return self.distance_to(other_body, i) <= collision_radius
    
    def near_barrier(self, bottom_barrier, top_barrier, collision_radius, i):
        return \
            abs(self.position[i] - bottom_barrier) <= collision_radius or \
            abs(self.position[i] - top_barrier) <= collision_radius

def apply_gravitational_field(i, body, objects):
    if body.stationary:
        return
    
    other_objects = [x for x in objects if x is not body]


def simulate(objects, time_range, dt, 
             collision_radius=0.1, 
             bottom_barrier=-10, 
             top_barrier=10):
    time = np.arange(time_range[0], time_range[1], dt)
    total_frames = np.size(time)
    collisions = set()
    
    for body in objects:
        body.init(total_frames)
    
    for i in range(1, total_frames):
        # update positions
        for body in objects:
            if body.stationary = True:
                continue
            
            body.position[i] = body.position[i-1] + body.velocity[i-1]     * dt
            body.velocity[i] = body.velocity[i-1] + body.acceleration[i-1] * dt
            body.acceleration[i] = apply_gravitational_field(i, body, objects)
        
        # detect collisions
        to_invert = set()
        
        for first_body_idx, body in enumerate(objects):
            for other_body in objects[first_body_idx+1:]:
                if body.collided_with(other_body, collision_radius, i):
                    collisions.add((time[i], body.position[i]))
                    to_invert.add(body)
                    to_invert.add(other_body)
                    
        for body in objects:
            if body.near_barrier(bottom_barrier, top_barrier, collision_radius, i):
                to_invert.add(body)
        
        for body in to_invert:
            body.velocity[i] *= -1
            
    # ending calculations
    for body in objects:
        body.momentum = body.mass * abs(body.velocity)
        body.energy = (1/2) * body.mass * body.velocity ** 2
        
    # transpose collisions
    collisions = np.array(list(zip(*collisions)))
        
    return time, collisions

def plot_results(objects, time, collisions):
    fig, axs = plt.subplots(4,1, sharex=True, figsize=(8,8))

    # position
    for body in objects:
        axs[0].plot(time, body.position, 
                    color=body.color, 
                    label=body.name,
                    linestyle='dashed')
    axs[0].scatter(collisions[0], collisions[1], label='Collisions')
    axs[0].axhline(y=10, color='black', label='Top Barrier')
    axs[0].axhline(y=-10, color='black', label='Bottom Barrier')
    axs[0].set_ylabel('Position\n(m)')
    axs[0].set_title('Position vs. Time')
    axs[0].legend()

    # velocity
    for body in objects:
        axs[1].plot(time, body.velocity, 
                    color=body.color, 
                    label=body.name)
    axs[1].set_ylabel('Velocity\n(m/s)')
    axs[1].set_title('Velocity vs. Time')
    axs[1].legend()

    # kinetic energy
    total_energy = np.zeros_like(time)
    for body in objects:
        axs[2].plot(time, body.energy * 1e3, 
                    color=body.color, 
                    label=body.name)
        total_energy = total_energy + body.energy
    axs[2].plot(time, total_energy * 1e3, color='green', label='Total Kinetic Energy')
    axs[2].set_ylabel('Kinetic Energy\n(mJ)')
    axs[2].set_title('Kinetic Energy vs. Time')
    axs[2].legend()

    # momentum
    total_momentum = np.zeros_like(time)
    for body in objects:
        axs[3].plot(time, body.momentum * 1e2, 
                    color=body.color, 
                    label=body.name)
        total_momentum = total_momentum + body.momentum
    axs[3].plot(time, total_momentum * 1e2, color='green', label='Total Momentum')
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Momentum\n(kg*m/s*10^2)')
    axs[3].set_title('Momentum vs. Time')
    axs[3].legend()

    plt.show()

def simulate_and_plot(objects, time_range, dt):
    time, collisions = simulate(objects, time_range, dt)
    plot_results(objects, time, collisions)


"""
simulate_and_plot([
    RigidBody(
        initial_position = -5,
        initial_velocity =  2,
        initial_acceleration = 0,
        mass = 0.01,
        color = 'lightblue',
        name = 'Object 1'
    ),
    RigidBody(
        initial_position =  7,
        initial_velocity = -1,
        initial_acceleration = 0,
        mass = 0.001,
        color = 'orange',
        name = 'Object 2'
    )], time_range=(0, 100), dt=0.01)

simulate_and_plot([
    RigidBody(
        initial_position = -5,
        initial_velocity =  2,
        initial_acceleration = 0,
        mass = 0.01,
        color = 'lightblue',
        name = 'Object 1'
    ),
    RigidBody(
        initial_position =  7,
        initial_velocity = -1,
        initial_acceleration = -1,
        mass = 0.001,
        color = 'orange',
        name = 'Object 2'
    )], time_range=(0, 100), dt=0.01)

simulate_and_plot([
    RigidBody(
        initial_position = -5,
        initial_velocity =  2,
        initial_acceleration = 2,
        mass = 0.01,
        color = 'lightblue',
        name = 'Object 1'
    ),
    RigidBody(
        initial_position =  7,
        initial_velocity = -1,
        initial_acceleration = -1,
        mass = 0.1,
        color = 'orange',
        name = 'Object 2'
    )], time_range=(0, 100), dt=0.01)
"""
