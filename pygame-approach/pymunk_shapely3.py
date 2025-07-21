import json
import pymunk
import pymunk.pygame_util
from pymunk.pygame_util import DrawOptions
import pygame
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from shapely.affinity import scale
from shapely.ops import nearest_points

# Define screen dimensions
WIDTH, HEIGHT = 1041, 887
# Define the size of the main rectangle (width, height) as requested: 40x80
RECT_SIZE = (40, 80)
# Define the size of the internal divisions (width, height) as requested: 20x20
DIVISION_SIZE = (20, 20)
# Border width for the main rectangles
BORDER_WIDTH = 10
# List to store dynamic Pymunk bodies (rectangles)
dynamic_bodies = []


def shapely_polygons_to_static_segment_edges(gdf, space, scale_factor=1.0):
    """
    Converts Shapely polygons from a GeoDataFrame into static Pymunk segment edges
    in the given Pymunk space. These segments act as static boundaries in the simulation.
    Crucially, this now assumes a Y-DOWN coordinate system for Pymunk, matching Pygame.
    """
    static_bodies = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if not geom.is_valid or geom.is_empty:
            continue

        # Handle both single Polygons and MultiPolygons
        polygons = [geom] if isinstance(geom, Polygon) else list(geom.geoms)

        for poly in polygons:
            # Scale the polygon if a scale_factor is provided
            scaled_poly = scale(poly, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
            # Get the coordinates from the exterior of the scaled polygon.
            # These coordinates are already in the Pygame/Shapely Y-down coordinate system.
            coords_y_down = list(scaled_poly.exterior.coords)
            
            # Create a static Pymunk body for these segments.
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            space.add(body)
            static_bodies.append(body)

            # Create segments for each edge of the polygon.
            # No Y-coordinate flipping is needed here as Pymunk is now treated as Y-down.
            for i in range(len(coords_y_down) - 1):
                p1 = coords_y_down[i]
                p2 = coords_y_down[i+1]
                
                seg = pymunk.Segment(body, p1, p2, radius=1)
                seg.elasticity = 0.0  # No bounce
                seg.friction = 0.5    # Friction
                space.add(seg) # Add the segment to the Pymunk space

    return static_bodies


def apply_polygon_forces(rect_bodies, aoi_polygons, obstacle_polygons, attraction_strength=80000, repulsion_strength=150000):
    """
    Applies attractive force towards AOI polygons and repulsive force from obstacle polygons.
    Forces are inversely proportional to distance squared.
    All coordinates (Pymunk, Shapely, Pygame) are now treated as Y-DOWN.
    """
    aoi_union = None
    if aoi_polygons:
        aoi_union = aoi_polygons[0]
        for i in range(1, len(aoi_polygons)):
            aoi_union = aoi_union.union(aoi_polygons[i])

    for body in rect_bodies:
        # Only apply force to dynamic bodies
        if body.body_type != pymunk.Body.DYNAMIC:
            continue

        pos_current = body.position # Current position of the rectangle's body (Pymunk, Y-down)
        rect_point_shapely = Point(pos_current[0], pos_current[1]) # Shapely Point (Y-down)

        total_force_x, total_force_y = 0, 0

        # --- Attraction to AOI edges ---
        # Attract if the rectangle is not fully contained within the AOI
        if aoi_union and not aoi_union.is_empty and not aoi_union.contains(rect_point_shapely):
            # Find the closest point on the AOI's exterior to the rectangle's center.
            closest_point_on_aoi = nearest_points(rect_point_shapely, aoi_union.exterior)[1]
            
            # Calculate vector from rectangle to closest point on AOI (Y-down)
            dx = closest_point_on_aoi.x - rect_point_shapely.x
            dy = closest_point_on_aoi.y - rect_point_shapely.y # Y is consistent

            dist_sq = dx**2 + dy**2

            if dist_sq > 0:
                mag = (dx**2 + dy**2)**0.5
                norm_x = dx / mag
                norm_y = dy / mag
                
                # Apply inverse square law attraction
                force_mag = attraction_strength / dist_sq
                total_force_x += norm_x * force_mag
                total_force_y += norm_y * force_mag

        # --- Repulsion from Obstacles ---
        for obs_poly in obstacle_polygons:
            # Check for direct intersection or close proximity
            if obs_poly.intersects(rect_point_shapely) or rect_point_shapely.distance(obs_poly.exterior) < RECT_SIZE[0] * 1.5:
                # Find the closest point on the obstacle's exterior to the rectangle's center.
                closest_point_on_obs = nearest_points(rect_point_shapely, obs_poly.exterior)[1]
                
                # Calculate vector from closest point on obstacle to rectangle (Y-down)
                dx = rect_point_shapely.x - closest_point_on_obs.x # Force away from obstacle
                dy = rect_point_shapely.y - closest_point_on_obs.y # Force away from obstacle
                dist_sq = dx**2 + dy**2
                
                # Add a small epsilon to avoid division by zero or very large forces when exactly on edge
                if dist_sq < 1: 
                    dist_sq = 1 

                mag = (dx**2 + dy**2)**0.5
                if mag == 0: 
                    continue # Avoid division by zero if points are identical

                norm_x = dx / mag
                norm_y = dy / mag

                # Apply inverse square law repulsion, potentially weaker for distant objects
                if obs_poly.intersects(rect_point_shapely):
                    force_mag = repulsion_strength / dist_sq # Strong repulsion for direct overlap
                else:
                    force_mag = repulsion_strength / (dist_sq * 2) # Weaker repulsion for nearby objects

                total_force_x += norm_x * force_mag
                total_force_y += norm_y * force_mag

        # Apply the total calculated force to the body at its center of gravity in the Y-down system
        body.apply_force_at_world_point((total_force_x, total_force_y), pos_current)


def apply_rectangle_attraction(rect_bodies, force_strength=1000):
    """
    Applies an attractive force between all pairs of dynamic rectangle bodies.
    The force is inversely proportional to the square of the distance between them.
    This helps keep the rectangles clumped together. Coordinates are Y-down.
    """
    for i, body_a in enumerate(rect_bodies):
        if body_a.body_type != pymunk.Body.DYNAMIC:
            continue

        for j, body_b in enumerate(rect_bodies):
            if i == j or body_b.body_type != pymunk.Body.DYNAMIC:
                continue

            pos_a = body_a.position
            pos_b = body_b.position

            dx = pos_b[0] - pos_a[0]
            dy = pos_b[1] - pos_a[1]

            dist_sq = dx**2 + dy**2 # Squared distance between the bodies

            if dist_sq == 0:
                continue # Avoid division by zero

            # Calculate the attractive force components (Y-down consistent).
            fx = (dx / dist_sq) * force_strength
            fy = (dy / dist_sq) * force_strength

            # Apply force to body_a towards body_b.
            body_a.apply_force_at_world_point((fx, fy), pos_a)
            # Apply an equal and opposite force to body_b towards body_a (Newton's Third Law).
            body_b.apply_force_at_world_point((-fx, -fy), pos_b)


def add_rectangle(space, pos):
    """
    Adds a dynamic rectangle body to the Pymunk space at the specified position.
    pos is expected to be in a Y-down coordinate system, matching Pygame.
    """
    width, height_rect = RECT_SIZE
    mass = 1
    moment = pymunk.moment_for_box(mass, (width, height_rect))
    
    body = pymunk.Body(mass, moment)
    body.position = pos[0], pos[1] # Use position directly (Y-down)
    body.velocity_limit = 500 # Limit maximum linear velocity
    body.angular_velocity_limit = 10 # Limit maximum angular velocity
    
    shape = pymunk.Poly.create_box(body, size=(width, height_rect))
    shape.friction = 0.4
    shape.elasticity = 0.0
    shape.color = pygame.Color("lightblue")
    
    space.add(body, shape)
    dynamic_bodies.append(body)


def generate_initial_rectangles(space, aoi_polygons, obstacle_polygons, rect_size, density_factor=20):
    """
    Generates initial rectangle bodies within AOI polygons, avoiding obstacle polygons.
    Rectangles are placed on a grid. All coordinates are Y-down.
    """
    # Create a union of all AOI polygons to treat them as one large area
    if not aoi_polygons:
        print("No 'aoi' polygons found for initial placement.")
        return

    aoi_union = aoi_polygons[0]
    for i in range(1, len(aoi_polygons)):
        aoi_union = aoi_union.union(aoi_polygons[i])
    
    if aoi_union.is_empty:
        print("AOI union is empty, no space to place rectangles.")
        return

    # Determine a grid covering the bounding box of the AOI union
    minx, miny, maxx, maxy = aoi_union.bounds

    # Step size for the grid, influenced by rectangle size and density factor
    step_x = rect_size[0] + density_factor
    step_y = rect_size[1] + density_factor

    num_placed = 0
    # Iterate through a grid in the Y-down coordinate system
    for y in range(int(miny), int(maxy), step_y):
        for x in range(int(minx), int(maxx), step_x):
            
            # Create a Shapely Polygon representing the potential rectangle (in Y-down coords)
            rect_shapely = Polygon([
                (x, y),
                (x + rect_size[0], y),
                (x + rect_size[0], y + rect_size[1]),
                (x, y + rect_size[1])
            ])

            # Check if this potential rectangle is within the AOI and does not intersect any obstacles
            # Using intersects and contains centroid for a more robust check
            if aoi_union.intersects(rect_shapely) and aoi_union.contains(rect_shapely.centroid):
                is_obstacle_overlap = False
                for obs_poly in obstacle_polygons:
                    if obs_poly.intersects(rect_shapely):
                        is_obstacle_overlap = True
                        break
                
                if not is_obstacle_overlap:
                    # Calculate the center of the rectangle (already in Y-down)
                    center_x = x + rect_size[0] / 2
                    center_y = y + rect_size[1] / 2
                    
                    add_rectangle(space, (center_x, center_y))
                    num_placed += 1
    print(f"Placed {num_placed} initial rectangles.")


def load_labelme_json_to_geodataframe(json_file_path):
    """
    Loads polygon data from a Labelme JSON file and converts it into a GeoDataFrame.
    Each polygon defined in the JSON becomes a Shapely Polygon geometry.
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return None
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None

    polygons = []
    labels = []

    for shape in data.get('shapes', []):
        if shape.get('shape_type') == 'polygon' and 'points' in shape:
            points = shape['points']
            if len(points) >= 3:
                try:
                    polygon = Polygon(points)
                    polygons.append(polygon)
                    labels.append(shape.get('label', 'N/A'))
                except Exception as e:
                    print(f"Invalid polygon data encountered for label '{shape.get('label', 'N/A')}': {e}")
                    continue

    if not polygons:
        print("No valid polygons found in JSON file.")
        return gpd.GeoDataFrame()

    return gpd.GeoDataFrame({'label': labels, 'geometry': polygons}, crs=None)


def main(json_path, initial_density_factor=20): # Added initial_density_factor parameter
    """
    Main function to run the Pygame and Pymunk simulation.
    It initializes Pygame, Pymunk space, loads polygons from a JSON,
    and handles game loop events and physics updates.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pymunk Polygon Simulation with Automated Rectangle Filling")
    clock = pygame.time.Clock()
    
    # Initialize DrawOptions without explicit screen_height mapping.
    # Pymunk and Pygame are now operating in the same Y-down system.
    draw_options = DrawOptions(screen) 

    space = pymunk.Space()
    space.gravity = (0, 900) # Gravity now points downwards in the Y-down system

    gdf = load_labelme_json_to_geodataframe(json_path)

    if gdf is None or gdf.empty:
        print("No valid polygons to display. Exiting simulation.")
        pygame.quit()
        return

    # Separate AOI and obstacle polygons based on their labels (case-insensitive)
    aoi_polygons = [row.geometry for idx, row in gdf.iterrows() if row['label'].lower() == 'aoi']
    obstacle_polygons = [row.geometry for idx, row in gdf.iterrows() if row['label'].lower() in ['road', 'pipeline', 'pond']]

    # Convert ALL polygons (AOI and obstacles) to static Pymunk segments.
    # Pymunk now uses the same Y-down coordinates as Shapely.
    shapely_polygons_to_static_segment_edges(gdf, space, scale_factor=1.0)

    # Automatically generate and place rectangles within the AOI, avoiding obstacles
    generate_initial_rectangles(space, aoi_polygons, obstacle_polygons, RECT_SIZE, density_factor=initial_density_factor) # Use the new parameter
    
    mouse_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    mouse_joint = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Left-click to add a new rectangle at mouse position
                mouse_pos = pygame.mouse.get_pos() # Already Y-down
                add_rectangle(space, mouse_pos)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                # Right-click to grab and drag a dynamic rectangle
                mouse_pos = pygame.mouse.get_pos() # Already Y-down
                mouse_p = pymunk.Vec2d(mouse_pos[0], mouse_pos[1]) # No flip
                hit = space.point_query_nearest(mouse_p, 0, pymunk.ShapeFilter())
                if hit and hit.shape.body.body_type == pymunk.Body.DYNAMIC:
                    mouse_body.position = mouse_p
                    world_anchor_on_body = hit.shape.body.world_to_local(mouse_p)
                    mouse_joint = pymunk.PivotJoint(mouse_body, hit.shape.body, (0, 0), world_anchor_on_body)
                    mouse_joint.max_force = 1e6
                    space.add(mouse_joint)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                # Release the grabbed rectangle
                if mouse_joint:
                    space.remove(mouse_joint)
                    mouse_joint = None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:  # Press 'd' key to delete a rectangle
                    mouse_pos = pygame.mouse.get_pos() # Already Y-down
                    mouse_p = pymunk.Vec2d(mouse_pos[0], mouse_pos[1]) # No flip
                    hit = space.point_query_nearest(mouse_p, 0, pymunk.ShapeFilter())
                    if hit and hit.shape.body.body_type == pymunk.Body.DYNAMIC:
                        body_to_remove = hit.shape.body
                        space.remove(body_to_remove, hit.shape)
                        if body_to_remove in dynamic_bodies:
                            dynamic_bodies.remove(body_to_remove)

        screen.fill((240, 240, 240)) # Fill background

        # Draw Pymunk objects using debug_draw.
        # Now Pymunk coordinates are aligned with Pygame, so debug_draw should work directly.
        space.debug_draw(draw_options)

        border_color = pygame.Color("white") 
        division_color = pygame.Color("gray") # Color for internal division lines
        
        # Manual drawing for dynamic bodies to add internal divisions
        for body in dynamic_bodies:
            if body.body_type == pymunk.Body.DYNAMIC:
                for shape in body.shapes:
                    if isinstance(shape, pymunk.Poly): # Check if it's a polygon shape (our rectangles are)
                        local_verts = shape.get_vertices()
                        world_verts = [body.local_to_world(lv) for lv in local_verts]
                        
                        pygame_verts = []
                        for v in world_verts:
                            # Now Pymunk world coordinates are directly usable for Pygame drawing (Y-down)
                            pygame_x = int(v.x)
                            pygame_y = int(v.y) 
                            pygame_verts.append((pygame_x, pygame_y))
                        
                        # Draw the main polygon outline for the rectangle
                        pygame.draw.polygon(screen, border_color, pygame_verts, BORDER_WIDTH)

                        # --- Draw internal divisions ---
                        rect_width, rect_height = RECT_SIZE
                        div_width, div_height = DIVISION_SIZE

                        # Calculate number of horizontal and vertical divisions
                        num_h_divisions = int(rect_width / div_width)
                        num_v_divisions = int(rect_height / div_height)
                        
                        # Horizontal lines
                        for i in range(1, num_v_divisions):
                            local_y = -rect_height / 2 + i * div_height
                            # Define local endpoints of the line relative to the body's center
                            p1_local = pymunk.Vec2d(-rect_width / 2, local_y)
                            p2_local = pymunk.Vec2d(rect_width / 2, local_y)
                            
                            # Transform local points to world points (Pymunk Y-down)
                            p1_world = body.local_to_world(p1_local)
                            p2_world = body.local_to_world(p2_local)
                            
                            # Transform world points to Pygame screen coordinates (Y-down, direct use)
                            p1_pygame = (int(p1_world.x), int(p1_world.y))
                            p2_pygame = (int(p2_world.x), int(p2_world.y))
                            
                            pygame.draw.line(screen, division_color, p1_pygame, p2_pygame, 1)

                        # Vertical lines
                        for i in range(1, num_h_divisions):
                            local_x = -rect_width / 2 + i * div_width
                            # Define local endpoints of the line relative to the body's center
                            p1_local = pymunk.Vec2d(local_x, -rect_height / 2)
                            p2_local = pymunk.Vec2d(local_x, rect_height / 2)

                            # Transform local points to world points (Pymunk Y-down)
                            p1_world = body.local_to_world(p1_local)
                            p2_world = body.local_to_world(p2_local)
                            
                            # Transform world points to Pygame screen coordinates (Y-down, direct use)
                            p1_pygame = (int(p1_world.x), int(p1_world.y))
                            p2_pygame = (int(p2_world.x), int(p2_world.y))

                            pygame.draw.line(screen, division_color, p1_pygame, p2_pygame, 1)
                        # --- End of internal divisions drawing ---


        pygame.display.flip() # Update the full display Surface to the screen

        # Apply forces to dynamic bodies: attraction to AOI, repulsion from obstacles
        # No height_screen needed as Y-axis is consistent.
        apply_polygon_forces(dynamic_bodies, aoi_polygons, obstacle_polygons, attraction_strength=80000, repulsion_strength=150000)
        # Apply attraction between rectangles to keep them somewhat together
        apply_rectangle_attraction(dynamic_bodies, force_strength=10000)

        # Update mouse body position for dragging (Pymunk needs Y-down from Pygame)
        mouse_pos = pygame.mouse.get_pos()
        mouse_body.position = pymunk.Vec2d(*mouse_pos)

        space.step(1 / 60.0) # Advance the physics simulation
        clock.tick(60) # Limit the frame rate to 60 FPS

    pygame.quit() # Uninitialize Pygame modules

if __name__ == "__main__":
    # Ensure this path is correct for your 'map1.json' file
    json_file_path = "map3.json" 
    # Calling main with a reduced density_factor to fill more rectangles
    main(json_file_path, initial_density_factor=1) 
