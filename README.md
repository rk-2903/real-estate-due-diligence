## Project Overview

This project introduces an AI-powered system to automate real estate layout planning using generative models, geospatial intelligence, and physics-based simulation. The system integrates satellite imagery, elevation data, and infrastructural constraints (e.g., roads, pipelines) to generate optimized land development layouts.
We explore three approaches: (1) Generative AI for layout proposals, (2) physics-driven simulation using Pymunk and Pygame, and (3) rule-based plot generation using JavaScript. Each approach offers unique benefits and challenges. This work aims to enhance early-stage development due diligence through interactive, data-driven layout generation while paving the way for scalable AI-human collaboration in land planning.


This repository contains real estate due diligence tools with two distinct approaches:

1. **JavaScript Approach**: Web-based land subdivision planner (`Javascript-approach/real_estate.html`)
2. **Python/Pygame Approach**: Physics-based land development simulation using PyMunk and Shapely (`pygame-approach/`)

## Architecture

### JavaScript Approach
- Single-file HTML application with embedded CSS and JavaScript
- Canvas-based interactive land planning tool
- Features: polygon drawing, exclusion zones, automated subdivision generation
- Coordinate system conversion between lat/lng and pixel coordinates
- Real-time statistics calculation for land use efficiency

### Python Approach
- Multiple Python files with increasing complexity:
  - `pymunk_shapely.py`: Basic physics simulation with manual rectangle placement
  - `pymunk_shapely3.py`: Enhanced version with automated placement and AOI/obstacle handling  
  - `pymunk_shapely_user_input.py`: Interactive version with user input for dimensions
- Dependencies: `pymunk`, `pygame`, `geopandas`, `shapely`
- Uses LabelMe JSON format for polygon data (`map1.json`, `map3.json`)
- Physics simulation with attraction/repulsion forces
- Y-down coordinate system alignment between Pygame and PyMunk

## Key Technologies

### JavaScript Approach
- HTML5 Canvas for visualization
- Vanilla JavaScript (no frameworks)
- Coordinate geometry calculations
- Real-time interactive drawing

### Python Approach  
- **PyMunk**: 2D physics simulation engine
- **Pygame**: Graphics and user interaction
- **Shapely**: Geometric operations and polygon handling
- **GeoPandas**: Geospatial data management
- **JSON**: Data persistence for polygon definitions

## Data Format

The Python approach uses LabelMe JSON format with polygon shapes labeled as:
- `aoi`: Area of Interest (buildable land)
- `road`: Road exclusion zones
- `pipeline`: Pipeline exclusion zones  
- `pond`: Water body exclusion zones

## Common Development Tasks

### Running the Applications

JavaScript approach:
```bash
# Open in browser - no build required
open Javascript-approach/real_estate.html
```

Python approach:
```bash
cd pygame-approach
python pymunk_shapely_user_input.py  # Most advanced version
```

### Key Constants and Configuration

Python simulations use these configurable parameters:
- `WIDTH, HEIGHT = 1041, 887` (screen dimensions)
- `RECT_SIZE`: Dynamically set based on user input (block dimensions)
- `DIVISION_SIZE`: Internal plot subdivisions
- `BORDER_WIDTH`: Road width for visual representation

### Physics Parameters
- Gravity: `(0, 900)` (Y-down coordinate system)
- Force strengths: Attraction (~80000), Repulsion (~150000)
- Velocity limits applied to prevent excessive movement

## Important Implementation Details

### Coordinate Systems
- JavaScript: Standard web coordinates with custom lat/lng conversion
- Python: Y-down coordinate system throughout (Pygame, PyMunk, Shapely aligned)
- No coordinate flipping needed in Python approach

### File Dependencies
- Python files depend on JSON map files in same directory
- No package.json - uses Python virtual environment in `realstate/` subdirectory
- Large JSON files (>170k tokens) contain detailed polygon coordinate data

### Interactive Controls
- Left-click: Add rectangle/polygon
- Right-click + drag: Move objects
- 'D' key: Delete object under mouse (Python only)
- Mode switching for different drawing tools (JavaScript only)
