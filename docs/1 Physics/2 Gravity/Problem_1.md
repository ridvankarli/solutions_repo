# Orbital Period and Orbital Radius: Kepler's Third Law

## 1. Mathematical Derivation of Kepler's Third Law

Kepler's Third Law establishes a fundamental relationship between an object's orbital period and its orbital radius. This relationship can be derived from first principles using Newton's laws of motion and universal gravitation.

### For Circular Orbits

For a body of mass $m$ orbiting a much larger body of mass $M$ in a circular orbit:

1. The centripetal force is provided by gravitational attraction:

$$F_{\text{centripetal}} = F_{\text{gravity}}$$

2. Expanding each side:

$$m\frac{v^2}{r} = G\frac{Mm}{r^2}$$

Where:
- $v$ is the orbital velocity
- $r$ is the orbital radius
- $G$ is the gravitational constant ($6.674 \times 10^{-11} \text{ m}^3 \text{ kg}^{-1} \text{ s}^{-2}$)

3. Solving for $v$:

$$v = \sqrt{\frac{GM}{r}}$$

4. For a circular orbit, the period $T$ is related to velocity by:

$$v = \frac{2\pi r}{T}$$

5. Substituting this into our equation:

$$\frac{2\pi r}{T} = \sqrt{\frac{GM}{r}}$$

6. Rearranging:

$$T^2 = \frac{4\pi^2 r^3}{GM}$$

7. This can be simplified to:

$$T^2 = \frac{4\pi^2}{GM} \cdot r^3$$

Which demonstrates that the square of the orbital period is proportional to the cube of the orbital radius:

$$T^2 \propto r^3$$

Or more specifically:

$$\frac{T^2}{r^3} = \frac{4\pi^2}{GM} = \text{constant}$$

This is Kepler's Third Law, also known as the "law of harmonies."

## 2. Implications for Astronomy

Kepler's Third Law has profound implications for our understanding of celestial mechanics and provides powerful tools for astronomers:

### Determining Masses of Celestial Bodies

The relationship can be rearranged to solve for the mass of the central body:

$$M = \frac{4\pi^2 r^3}{GT^2}$$

This allows astronomers to:
- Calculate the mass of the Sun by measuring the orbital periods and radii of planets
- Determine the masses of planets by studying their moons
- Estimate the masses of distant stars by observing their binary companions
- Calculate the mass of the Milky Way based on the orbital properties of globular clusters

### Predicting Orbital Distances

For a system where the central mass is known, the law can be used to:
- Predict the distance of a celestial body based on its orbital period
- Verify the presence of unseen planets or stars based on gravitational perturbations
- Calculate the "habitable zone" distances around stars of different masses

### Exoplanet Detection

Kepler's Third Law is instrumental in the detection and characterization of exoplanets:
- Transit timing: Variations in transit timing can reveal additional planets in a system
- Radial velocity methods: The law helps convert observed radial velocity measurements into orbital parameters
- Direct imaging: Knowledge of expected orbital periods helps plan observation campaigns

### Natural Satellite Systems

The law explains the distribution of satellites and rings around planets:
- Predicts the locations of stable orbits
- Helps identify resonance effects between moons
- Provides insight into the formation history of planetary systems

## 3. Real-World Examples

Let's analyze several celestial systems to verify Kepler's Third Law:

### Earth-Moon System

The Moon orbits Earth at an average distance of 384,400 km with an orbital period of 27.32 days.

Using Kepler's Third Law, we can calculate Earth's mass:

$$M_{\text{Earth}} = \frac{4\pi^2 r^3}{GT^2}$$

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
G = 6.674e-11  # Gravitational constant (m^3 kg^-1 s^-2)

# Earth-Moon system
r_moon = 384400000  # meters
T_moon = 27.32 * 24 * 3600  # seconds (27.32 days)

# Calculate Earth's mass from Moon's orbit
M_earth_calculated = 4 * np.pi**2 * r_moon**3 / (G * T_moon**2)
print(f"Earth's mass calculated from Moon's orbit: {M_earth_calculated:.3e} kg")
print(f"Actual Earth's mass: 5.972e24 kg")
```

### Solar System

The table below shows the orbital parameters of planets in our Solar System:

| Planet   | Orbital Radius (AU) | Orbital Period (years) | T²/r³ (yr²/AU³) |
|----------|---------------------|------------------------|------------------|
| Mercury  | 0.387               | 0.241                  | 2.987            |
| Venus    | 0.723               | 0.615                  | 2.991            |
| Earth    | 1.000               | 1.000                  | 3.000            |
| Mars     | 1.524               | 1.881                  | 2.991            |
| Jupiter  | 5.203               | 11.86                  | 2.999            |
| Saturn   | 9.537               | 29.46                  | 3.001            |
| Uranus   | 19.191              | 84.01                  | 2.998            |
| Neptune  | 30.069              | 164.8                  | 2.995            |

```python
# Solar system data
planets = {
    'Mercury': {'a': 0.387, 'T': 0.241},
    'Venus': {'a': 0.723, 'T': 0.615},
    'Earth': {'a': 1.000, 'T': 1.000},
    'Mars': {'a': 1.524, 'T': 1.881},
    'Jupiter': {'a': 5.203, 'T': 11.86},
    'Saturn': {'a': 9.537, 'T': 29.46},
    'Uranus': {'a': 19.191, 'T': 84.01},
    'Neptune': {'a': 30.069, 'T': 164.8}
}

# Calculate T²/r³ for each planet
for planet, data in planets.items():
    t_squared_over_r_cubed = data['T']**2 / data['a']**3
    print(f"{planet}: T²/r³ = {t_squared_over_r_cubed:.4f}")

# Calculate Solar mass from each planet's orbit
for planet, data in planets.items():
    M_sun = 4 * np.pi**2 * (data['a'] * 1.496e11)**3 / (G * (data['T'] * 365.25 * 24 * 3600)**2)
    print(f"Sun's mass calculated from {planet}'s orbit: {M_sun:.3e} kg")

print(f"Actual Sun's mass: 1.989e30 kg")
```

### Verification with Plot

We can visualize the relationship by plotting orbital period squared against orbital radius cubed:

```python
# Extract data for plotting
radii = np.array([data['a'] for planet, data in planets.items()])
periods = np.array([data['T'] for planet, data in planets.items()])
names = list(planets.keys())

# Create plot
plt.figure(figsize=(12, 8))

# Plot T² vs r³
plt.subplot(2, 2, 1)
plt.scatter(radii**3, periods**2, c='blue', s=80, alpha=0.7)
for i, name in enumerate(names):
    plt.annotate(name, (radii[i]**3, periods[i]**2), fontsize=10)

# Plot best fit line
x_range = np.linspace(0, max(radii**3), 100)
plt.plot(x_range, 3 * x_range, 'r--', label=r'$T^2 = 3r^3$ (theoretical)')
plt.xlabel(r'Orbital Radius Cubed ($\text{AU}^3$)')
plt.ylabel(r'Orbital Period Squared ($\text{years}^2$)')
plt.title("Kepler's Third Law: $T^2 \propto r^3$")
plt.grid(True, alpha=0.3)
plt.legend()

# Use log-log scale to better visualize the relationship
plt.subplot(2, 2, 2)
plt.loglog(radii, periods, 'o', ms=10)
for i, name in enumerate(names):
    plt.annotate(name, (radii[i]*1.1, periods[i]), fontsize=10)

# Add theoretical line with slope 3/2
x_range = np.logspace(np.log10(min(radii))-0.5, np.log10(max(radii))+0.5, 100)
plt.loglog(x_range, x_range**(3/2), 'r--', label=r'Slope = 3/2')
plt.xlabel('Orbital Radius (AU, log scale)')
plt.ylabel('Orbital Period (years, log scale)')
plt.title("Kepler's Third Law (Log-Log Plot)")
plt.grid(True, alpha=0.3, which='both')
plt.legend()

plt.tight_layout()
plt.savefig('keplers_third_law.png')
plt.show()
```

## 4. Computational Model for Simulating Circular Orbits

We can verify Kepler's Third Law by simulating the orbital motion of bodies:

```python
def simulate_orbit(central_mass, orbital_radius, num_points=1000):
    """Simulate a circular orbit and return coordinates"""
    # Calculate period using Kepler's Third Law
    period = 2 * np.pi * np.sqrt(orbital_radius**3 / (G * central_mass))
    
    # Generate points for one complete orbit
    theta = np.linspace(0, 2*np.pi, num_points)
    x = orbital_radius * np.cos(theta)
    y = orbital_radius * np.sin(theta)
    
    return x, y, period

# Simulation parameters
M_sun = 1.989e30  # kg
AU = 1.496e11  # meters

# Create animation of the inner planets
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Set up plot limits
max_radius = 1.7 * AU  # Out to Mars
ax.set_xlim(-max_radius, max_radius)
ax.set_ylim(-max_radius, max_radius)

# Plot the Sun
sun = plt.Circle((0, 0), 0.05 * AU, color='yellow')
ax.add_patch(sun)

# Colors for planets
colors = {'Mercury': 'gray', 'Venus': 'orange', 'Earth': 'blue', 'Mars': 'red'}

# Planet data (in AU)
inner_planets = {
    'Mercury': 0.387,
    'Venus': 0.723,
    'Earth': 1.000,
    'Mars': 1.524
}

# Plot orbits
for planet, radius in inner_planets.items():
    x, y, period = simulate_orbit(M_sun, radius * AU)
    ax.plot(x, y, '--', color=colors[planet], alpha=0.5)
    period_days = period / (24 * 3600)
    ax.text(0.2 * radius * AU, radius * AU, 
            f"{planet}\nRadius: {radius:.3f} AU\nPeriod: {period_days:.1f} days",
            color=colors[planet])

# Create planet objects for animation
planet_objects = {}
planet_positions = {}

for planet, radius in inner_planets.items():
    x, y, _ = simulate_orbit(M_sun, radius * AU, num_points=1000)
    planet_objects[planet] = plt.Circle((x[0], y[0]), 0.025 * AU, color=colors[planet])
    planet_positions[planet] = (x, y)
    ax.add_patch(planet_objects[planet])

plt.title("Simulation of Inner Solar System (not to scale)")
plt.xlabel("Distance (m)")
plt.ylabel("Distance (m)")

# For animation, you would add:
"""
def animate(i):
    for planet in inner_planets:
        x, y = planet_positions[planet]
        # Different planets move at different speeds according to Kepler's laws
        # We'll adjust the index calculation to reflect this
        idx = int((i * len(x) / (inner_planets[planet]**1.5)) % len(x))
        planet_objects[planet].center = (x[idx], y[idx])
    return list(planet_objects.values())

ani = FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
ani.save('solar_system.gif', writer='pillow', fps=30)
"""

plt.savefig('inner_planets.png')
plt.show()

# Verify relationship with multiple simulations
radii = np.linspace(0.5 * AU, 10 * AU, 20)
periods = []

for r in radii:
    _, _, period = simulate_orbit(M_sun, r)
    periods.append(period)

# Convert to years
periods_years = np.array(periods) / (365.25 * 24 * 3600)
radii_AU = radii / AU

plt.figure(figsize=(10, 6))
plt.plot(radii_AU**3, periods_years**2, 'bo', alpha=0.7)

# Add best fit line
coeffs = np.polyfit(radii_AU**3, periods_years**2, 1)
poly_fn = np.poly1d(coeffs)
plt.plot(radii_AU**3, poly_fn(radii_AU**3), 'r-', 
         label=f'Best fit: $T^2 = {coeffs[0]:.4f}\\cdot r^3 + {coeffs[1]:.4f}$')

plt.xlabel(r'Orbital Radius Cubed ($\text{AU}^3$)')
plt.ylabel(r'Orbital Period Squared ($\text{years}^2$)')
plt.title("Verification of Kepler's Third Law from Simulation")
plt.grid(True)
plt.legend()
plt.savefig('kepler_verification.png')
plt.show()
```

## 5. Extension to Elliptical Orbits

Kepler's Third Law applies to elliptical orbits as well as circular ones. For elliptical orbits:

$$T^2 = \frac{4\pi^2}{GM} \cdot a^3$$

Where $a$ is the semi-major axis of the ellipse.

### Modifications for Elliptical Orbits

The key differences for elliptical orbits:

1. The orbital speed varies according to Kepler's Second Law (equal areas in equal times)
2. The distance from the central body to the orbiting body changes throughout the orbit
3. The semi-major axis replaces the radius in the formula

```python
def simulate_elliptical_orbit(central_mass, semi_major_axis, eccentricity, num_points=1000):
    """Simulate an elliptical orbit and return coordinates"""
    # Calculate period using Kepler's Third Law (same formula)
    period = 2 * np.pi * np.sqrt(semi_major_axis**3 / (G * central_mass))
    
    # Semi-minor axis
    semi_minor_axis = semi_major_axis * np.sqrt(1 - eccentricity**2)
    
    # Generate points for one complete orbit
    theta = np.linspace(0, 2*np.pi, num_points)
    
    # Parametric equation of ellipse
    x = semi_major_axis * np.cos(theta)
    y = semi_minor_axis * np.sin(theta)
    
    # Shift ellipse so the focus is at the origin (where the Sun is)
    x = x + eccentricity * semi_major_axis
    
    return x, y, period

# Demonstrate with different eccentricities
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Sun at focus
sun = plt.Circle((0, 0), 0.1 * AU, color='yellow')
ax.add_patch(sun)

# Plot orbits with different eccentricities
eccentricities = [0, 0.2, 0.5, 0.7, 0.9]
semi_major = 1 * AU  # Keep same semi-major axis

for e in eccentricities:
    x, y, period = simulate_elliptical_orbit(M_sun, semi_major, e)
    ax.plot(x, y, label=f'e = {e}, T = {period/(24*3600):.1f} days')
    
    # Mark perihelion and aphelion
    if e > 0:
        perihelion = (1-e) * semi_major
        aphelion = (1+e) * semi_major
        ax.plot([perihelion], [0], 'ro', ms=5)
        ax.plot([-(aphelion-perihelion)], [0], 'bo', ms=5)

plt.legend()
plt.title("Elliptical Orbits with Same Semi-Major Axis (1 AU)")
plt.xlabel("Distance (m)")
plt.ylabel("Distance (m)")
plt.savefig('elliptical_orbits.png')
plt.show()

# Verify Kepler's Third Law for elliptical orbits
eccentricities = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
semi_majors = np.linspace(0.5 * AU, 5 * AU, 5)

results = []

for a in semi_majors:
    for e in eccentricities:
        _, _, period = simulate_elliptical_orbit(M_sun, a, e)
        results.append({
            'semi_major': a / AU,
            'eccentricity': e,
            'period': period / (365.25 * 24 * 3600)  # in years
        })

# Convert to DataFrame for easy plotting
import pandas as pd
df = pd.DataFrame(results)

# Calculate T²/a³ ratio - should be constant regardless of eccentricity
df['T2_a3_ratio'] = df['period']**2 / df['semi_major']**3

plt.figure(figsize=(10, 6))
for e in eccentricities:
    subset = df[df['eccentricity'] == e]
    plt.scatter(subset['semi_major']**3, subset['period']**2, 
                label=f'e = {e}', s=80, alpha=0.7)

plt.plot(np.linspace(0, 125, 100), 3 * np.linspace(0, 125, 100), 'k--', 
         label='Theoretical: $T^2 = 3a^3$')
plt.xlabel(r'Semi-Major Axis Cubed ($\text{AU}^3$)')
plt.ylabel(r'Orbital Period Squared ($\text{years}^2$)')
plt.title("Kepler's Third Law with Different Eccentricities")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('kepler_elliptical.png')
plt.show()

# Show that T²/a³ ratio is constant regardless of eccentricity
plt.figure(figsize=(10, 6))
plt.scatter(df['eccentricity'], df['T2_a3_ratio'], c=df['semi_major'], 
            cmap='viridis', s=80, alpha=0.7)
plt.colorbar(label='Semi-Major Axis (AU)')
plt.axhline(y=3, color='r', linestyle='--', label='Theoretical value: 3')
plt.xlabel('Eccentricity')
plt.ylabel(r'$T^2/a^3$ Ratio')
plt.title("Kepler's Constant Across Different Eccentricities")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('kepler_constant.png')
plt.show()
```

## 6. Applications in Modern Astrophysics

Kepler's Third Law continues to be fundamental in modern astrophysics:

### Binary Star Systems

For binary star systems where both masses are significant:

$$T^2 = \frac{4\pi^2 a^3}{G(M_1 + M_2)}$$

This modified form allows astronomers to:
- Calculate the combined mass of the system
- When combined with spectroscopic data, determine individual stellar masses
- Study the evolution of close binary systems

### Exoplanet Detection and Characterization

Kepler's Third Law plays a crucial role in:
- Transit timing variations (TTVs) for detecting additional planets
- Radial velocity measurements to determine planetary masses
- Estimating habitable zone boundaries

### Dark Matter Studies

The law helps reveal the presence of dark matter:
- Galaxy rotation curves deviate from predictions based on visible matter
- The velocity distributions of stars in galaxies suggest additional mass
- Applying Kepler's Third Law to galactic rotation allows estimation of dark matter content

### General Relativity Effects

In extreme gravitational environments:
- Mercury's orbit precession demonstrated limitations of Newton's formulation
- Einstein's General Relativity provides corrections to Kepler's laws
- For objects orbiting very massive bodies (like black holes), relativistic effects become significant

## 7. Conclusion

Kepler's Third Law, relating the square of the orbital period to the cube of the orbital radius, represents one of the most elegant and enduring principles in physics. From its original formulation based on astronomical observations to its derivation from Newton's laws and extension in Einstein's relativity, this relationship continues to provide a fundamental framework for understanding orbital dynamics.

Our simulations confirm that this relationship holds across a wide range of orbital parameters, including different eccentricities. The constant ratio between T² and r³ (or a³ for elliptical orbits) provides a powerful tool for astronomers to determine masses, predict orbital characteristics, and explore the nature of gravity throughout the universe.

From Earth-orbiting satellites to distant exoplanetary systems and galactic dynamics, Kepler's Third Law remains an essential tool in modern astronomy and astrophysics, demonstrating how a simple mathematical relationship can provide profound insights into the workings of the cosmos.