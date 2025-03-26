# Orbital Period and Orbital Radius

## 1. Motivation
Kepler's Third Law states that the square of the orbital period is proportional to the cube of the orbital radius. This fundamental relationship in celestial mechanics allows scientists to predict planetary motions and analyze gravitational interactions across various scales. Understanding this law is crucial for applications in astrophysics, satellite navigation, and space exploration.

## 2. Key Equations
### Orbital Motion
For a body orbiting under gravitational influence:
- Centripetal force: \[ F_c = \frac{m v^2}{r} \]
- Gravitational force: \[ F_g = \frac{G M m}{r^2} \]

Equating the two:
\[ \frac{m v^2}{r} = \frac{G M m}{r^2} \]

Since orbital velocity is \( v = \frac{2\pi r}{T} \), substituting gives:
\[ \frac{m}{r} \left( \frac{4\pi^2 r^2}{T^2} \right) = \frac{G M m}{r^2} \]

Simplifying:
\[ T^2 = \frac{4\pi^2 r^3}{G M} \]

This confirms Kepler’s Third Law: the square of the orbital period is proportional to the cube of the orbital radius.

## 3. Analysis
- The law applies to planets, moons, and artificial satellites.
- Helps in calculating planetary masses and distances.
- Used in space mission planning and astrophysical modeling.

## 4. Implementation
- Develop a Python script to simulate circular orbits.
- Verify Kepler’s Third Law through computational analysis.
- Generate plots showing the relationship between orbital period and radius.

## 5. Practical Applications
- Determining satellite orbits around Earth.
- Predicting planetary motion in exoplanetary systems.
- Understanding the mechanics of binary star systems.

## 6. Limitations and Extensions
- Kepler’s law applies ideally to two-body systems; real-world deviations occur due to perturbations.
- Can be extended to elliptical orbits using generalized forms.
- Future work can include analyzing multi-body interactions in complex gravitational systems.

