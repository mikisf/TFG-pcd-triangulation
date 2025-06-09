# Reconstrucció de Superfícies a partir de Núvols de Punts

Aquest repositori conté les implementacions desenvolupats pel projecte de reconstrucció de superfícies, que inclou tècniques com Marching Cubes, Marching Tetrahedra, reconstrucció de Poisson i simplificació de malles mitjançant mètriques d'error quàdriques (QEM).

## 📁 Estructura del repositori

### `marching_cubes/`

Implementacions relacionades amb els algorismes de Marching Squares i Marching Cubes.

- `marching_squares.py`: Algorisme Marching Squares aplicat sobre una graella 2D generada aleatòriament.
- `marching_cubes.py`: Algorisme Marching Cubes aplicat sobre un camp escalar 3D aleatori.
- `extract_isosurface.py`: Script per aplicar Marching Cubes sobre volums reals (CThead, MRbrain i Bunny) i extreure’n les isosuperfícies.

### `marching_tetrahedra/`

Implementacions de l’algorisme Marching Tetrahedra i eines de visualització.

- `marching_tetrahedra.py`: Extracció d’isosuperfícies aplicat al volum del disc (`Disk`).
- `test_all_tetra_cases.py`: Visualització de tots els casos possibles de l’algorisme Marching Tetrahedra.

### `poisson/`

Scripts per a la reconstrucció de superfícies mitjançant el mètode de Poisson.

- `poisson_2d.py`: Reconstrucció d’un cercle a partir d’un conjunt de punts en 2D.
- `poisson_3d.py`: Reconstrucció del model 3D del conill de Stanford (`Bunny`) a partir del seu núvol de punts.

### `results/`

Scripts per a la generació i comparació de resultats entre diferents mètodes.

- `compare_algorithms.py`: Generació de superfícies esfèriques utilitzant Marching Cubes, Marching Tetrahedra i Poisson.
- `metrics.py`: Càlcul de mètriques de qualitat sobre les malles generades (regularitat, distribució d’àrees, degeneració, etc.).

### `surface_simplification_QEM/`

Implementació del mètode de simplificació de superfícies mitjançant mètriques d’error quàdriques (QEM).

- `surface_simplification_QEM.py`: Reducció de la complexitat geomètrica del model del `Laptop`, preservant-ne la forma essencial.

## ⚙️ Requisits

Es recomana l’ús d’un entorn virtual per gestionar les dependències:

```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

## 🧑‍🎓 Crèdits

Aquest repositori ha estat desenvolupat com a part del treball de final de grau en Matemàtiques i Enginyeria Informàtica. El codi ha estat dissenyat amb finalitats educatives i d’investigació.
