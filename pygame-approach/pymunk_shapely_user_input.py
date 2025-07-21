import json
import pymunk
import pymunk.pygame_util
from pymunk.pygame_util import DrawOptions
import pygame
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from shapely.affinity import scale
from shapely.ops import nearest_points
import math

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


def apply_edge_attraction(rect_bodies, polygons, force_strength=50000):
    """
    Applies an attractive force to dynamic rectangle bodies, pulling them towards
    the closest edge of the provided Shapely polygons. The force is inversely
    proportional to the square of the distance.
    """
    for body in rect_bodies:
        # Only apply force to dynamic bodies
        if body.body_type != pymunk.Body.DYNAMIC:
            continue

        pos = body.position # Current position of the rectangle's body
        rect_point = Point(pos[0], pos[1]) # Convert Pymunk position to Shapely Point

        min_dist = float('inf') # Initialize minimum distance to infinity
        closest_target = None   # Initialize closest target point

        for poly in polygons:
            if not poly.is_valid or poly.is_empty:
                continue

            # Create a LineString from the polygon's exterior coordinates.
            edge = LineString(poly.exterior.coords)
            
            # Find the closest point on the polygon's edge to the rectangle's center.
            nearest_geom = nearest_points(rect_point, edge)[1]

            # Calculate the distance from the rectangle's center to the closest point on the edge.
            dist = rect_point.distance(nearest_geom)
            if dist < min_dist:
                min_dist = dist
                closest_target = nearest_geom

        if closest_target is not None:
            # Calculate the vector from the rectangle's position to the closest target point.
            dx = closest_target.x - pos[0]
            dy = closest_target.y - pos[1]
            dist_sq = dx**2 + dy**2 # Distance squared

            if dist_sq == 0:
                continue # Avoid division by zero

            # Calculate the force components.
            fx = (dx / dist_sq) * force_strength
            fy = (dy / dist_sq) * force_strength
            
            # Apply the force to the body at its center of gravity.
            body.apply_force_at_world_point((fx, fy), pos)


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
    width, height = RECT_SIZE
    mass = 1
    moment = pymunk.moment_for_box(mass, (width, height))
    
    body = pymunk.Body(mass, moment)
    body.position = pos[0], pos[1] # Use position directly (Y-down)
    body.velocity_limit = 500 # Limit maximum linear velocity
    body.angular_velocity_limit = 10 # Limit maximum angular velocity
    
    shape = pymunk.Poly.create_box(body, size=(width, height))
    shape.friction = 0.4
    shape.elasticity = 0.0
    shape.color = pygame.Color("lightblue")
    
    space.add(body, shape)
    dynamic_bodies.append(body)


def find_point_on_line_at_distance(p1, p2, distance):
    """
    Finds a point on the line segment connecting p1 and p2,
    that is 'distance' units away from p1.

    Args:
        p1 (tuple): The starting coordinate (x1, y1).
        p2 (tuple): The ending coordinate (x2, y2).
        distance (float): The desired distance from p1.

    Returns:
        tuple: The coordinate (x, y) of the point, or None if the
               distance is greater than the length of the line segment.
    """
    x1, y1 = p1
    x2, y2 = p2

    # 1. Calculate the vector from p1 to p2
    vx = x2 - x1
    vy = y2 - y1

    # 2. Calculate the length (magnitude) of the vector
    line_length = math.sqrt(vx**2 + vy**2)

    # Check if the desired distance is within the line segment length
    if distance > line_length:
        print(f"Warning: Desired distance ({distance}) is greater than the line length ({line_length}).")
        return None

    # 3. Normalize the vector (get a unit vector)
    if line_length == 0: # Handle case where p1 and p2 are the same point
        return p1 if distance == 0 else None
    
    uvx = vx / line_length
    uvy = vy / line_length

    # 4. Scale the unit vector by the desired distance
    point_x = x1 + uvx * distance
    point_y = y1 + uvy * distance

    return (point_x, point_y)




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


    num_placed = 0
    

    probable_points = []
    shapely_probable_points = []

    for obstacle_polygon in obstacle_polygons:
        
        
        x,y = obstacle_polygon.boundary.xy
        x = [int(value) for value in x ]
        y = [int(value) for value in y ]

        cent_x = int(obstacle_polygon.centroid.x)
        cent_y = int(obstacle_polygon.centroid.y)


        for x_point,y_point in zip(x,y):
            p1 = (x_point,y_point)
            p2 = (cent_x,cent_y)

            shapely_probable_points.append(Point(x_point,y_point))

            distance = RECT_SIZE[0]//2
            point = find_point_on_line_at_distance(p1, p2, distance)
            if point:
                dont_add = False
                for shapely_probable_point in shapely_probable_points[:-1]:
                    if shapely_probable_point.distance(Point(x_point,y_point)) <= RECT_SIZE[1]:
                        dont_add = True
                        # break
                if not dont_add:
                    probable_points.append(point)

            distance = -RECT_SIZE[0]//2
            point = find_point_on_line_at_distance(p1, p2, distance)
            if point:
                dont_add = False
                for shapely_probable_point in shapely_probable_points[:-1]:
                    if shapely_probable_point.distance(Point(x_point,y_point)) <= RECT_SIZE[1]:
                        dont_add = True
                        # break
                if not dont_add:
                    probable_points.append(point)



    for point in probable_points:
        shapely_point = Point(point[0],point[1])
        if aoi_union.contains(shapely_point) and (not obstacle_polygon.intersects(shapely_point)):
            add_rectangle(space, point)
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


def print_statement_colored():
    """
    Prints the statement "1 unit = 10 sqft and make sure that block
    is properly divisible by plot and all the values are even"
    with color and symbols in a visually appealing way.
    """
    statement = "1 unit = 10 sqft and make sure that block is properly divisible by plot and all the values are even"
    
    # ANSI escape codes for colors and reset
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    
    # Symbols
    STAR = "⭐"
    CHECK = "✅"
    INFO = "ℹ️"

    # Calculate line length based on the statement and symbols
    # Adding extra space for symbols and borders
    line_length = len(statement) + len(STAR) + len(CHECK) + 6 

    # Top border
    print(f"{BLUE}{'=' * line_length}{RESET}")
    
    # Statement line with symbols and color
    print(f"{BLUE}| {STAR} {GREEN}{statement}{RESET} {CHECK} |{RESET}")
    
    # Bottom border
    print(f"{BLUE}{'=' * line_length}{RESET}")
    
    # Optional: Add a little info message
    print(f"\n{INFO} {YELLOW}Ensuring precise measurements and even divisions.{RESET}")



def get_colored_inputs():
    """
    Gets block, plot, and road dimensions with colorful and structured prompts.
    """
    # ANSI escape codes for colors
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    print(f"\n{BLUE}--- {BOLD}Enter Dimensions{RESET} ---\n")

    # Block Dimensions
    print(f"{YELLOW}--- Block Details ---{RESET}")
    block_width = int(input(f"{GREEN}📏 Enter block width: {RESET}"))
    print(f"  {BLUE}•{RESET} Block width set to: {BOLD}{block_width}{RESET}")

    block_height = int(input(f"{GREEN}📐 Enter block height: {RESET}"))
    print(f"  {BLUE}•{RESET} Block height set to: {BOLD}{block_height}{RESET}")

    print(f"\n{YELLOW}--- Plot Details ---{RESET}")
    # Plot Dimensions
    plot_width = int(input(f"{GREEN}📏 Enter plot width: {RESET}"))
    print(f"  {BLUE}•{RESET} Plot width set to: {BOLD}{plot_width}{RESET}")

    plot_height = int(input(f"{GREEN}📐 Enter plot height: {RESET}"))
    print(f"  {BLUE}•{RESET} Plot height set to: {BOLD}{plot_height}{RESET}")

    print(f"\n{YELLOW}--- Road Details ---{RESET}")
    # Road Dimension
    road_width = int(input(f"{GREEN}🛣️ Enter road width: {RESET}"))
    print(f"  {BLUE}•{RESET} Road width set to: {BOLD}{road_width}{RESET}")

    print(f"\n{BLUE}--- {BOLD}Input Collection Complete!{RESET} ---\n")

    return block_width, block_height, plot_width, plot_height, road_width


def print_total_area_beautiful(area,decimal=2):
    """
    Prints the total area statement with a clean, highlighted look.
    The 'area' argument should be an integer or float.
    """
    # Use f-string formatting for commas directly.
    # The ':,' format specifier adds commas as thousands separators.

    area = round(area, 2)
    area_value_formatted = f"{area:,}"

    statement = f"Total area for plotting = {area_value_formatted} sqft"

    # ANSI escape codes for colors
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    # Calculate line length with proper padding
    line_length = len(statement) + 6  # Padding and borders

    print(f"{CYAN}{'=' * line_length}{RESET}")
    print(f"{CYAN}| {BOLD}{statement}{RESET} |")
    print(f"{CYAN}{'=' * line_length}{RESET}")

def main(json_path, initial_density_factor=20): # Added initial_density_factor parameter
    """
    Main function to run the Pygame and Pymunk simulation.
    It initializes Pygame, Pymunk space, loads polygons from a JSON,
    and handles game loop events and physics updates.
    """

    global WIDTH, HEIGHT
    global BORDER_WIDTH
    global DIVISION_SIZE
    global RECT_SIZE

    # Define screen dimensions
    WIDTH, HEIGHT = 1041, 887
    # Define the size of the main rectangle (width, height) as requested: 40x80

    print_statement_colored()
    print("\n")
    block_width, block_height, plot_width, plot_height, road_width = get_colored_inputs()

    BORDER_WIDTH = road_width
    DIVISION_SIZE = (plot_width, plot_height)

    RECT_SIZE = (block_width + (road_width//2) , block_height + (road_width//2))

    
    # Border width for the main rectangles


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

    print_total_area_beautiful(aoi_polygons[0].area * 100)

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

        border_color = pygame.Color("black") 
        division_color = pygame.Color("purple") # Color for internal division lines
        
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
                        
                        rect_width  = rect_width - (BORDER_WIDTH//2)
                        rect_height = rect_height - (BORDER_WIDTH//2)
                        
                        div_width, div_height = DIVISION_SIZE

                        # Calculate number of horizontal and vertical divisions
                        num_h_divisions = int(rect_width / div_width)
                        num_v_divisions = int(rect_height / div_height)

                        # Horizontal lines
                        for i in range(1, num_v_divisions):
                            local_y = -rect_height / 2 + i * div_height # - (BORDER_WIDTH//2)
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
                            local_x = -rect_width / 2 + i * div_width # - (BORDER_WIDTH//2)
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
        # apply_polygon_forces(dynamic_bodies, aoi_polygons, obstacle_polygons, attraction_strength=80000, repulsion_strength=150000)

        apply_edge_attraction(dynamic_bodies, list(gdf.geometry), force_strength=80000)

        # Apply attraction between rectangles to keep them somewhat together
        # apply_rectangle_attraction(dynamic_bodies, force_strength=10000)

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
