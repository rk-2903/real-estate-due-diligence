import json
import pymunk
import pymunk.pygame_util
from pymunk.pygame_util import DrawOptions
import pygame
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import scale
from pymunk.pygame_util import DrawOptions
from shapely.ops import triangulate
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

WIDTH, HEIGHT = 1041, 887
RECT_SIZE = (85, 45)  # width, height
BORDER_WIDTH = 10
dynamic_bodies = []

def is_ccw(polygon):
    """Check if polygon coordinates are counter-clockwise"""
    return polygon.exterior.is_ccw

def shapely_to_pymunk_coords(polygon, height):
    """Convert shapely coords to pymunk (flip y-axis)"""
    return [(x, height - y) for x, y in polygon.exterior.coords[:-1]]  # exclude duplicate last point

def shapely_polygons_to_static_segment_edges(gdf, space, scale_factor=1.0):
    static_bodies = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if not geom.is_valid or geom.is_empty:
            continue

        polygons = [geom] if isinstance(geom, Polygon) else list(geom.geoms)

        for poly in polygons:
            scaled_poly = scale(poly, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
            coords = list(scaled_poly.exterior.coords)
            print(f"coords = {coords }")
            # coords = [(x, HEIGHT - y) for x, y in coords]  # flip y

            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            space.add(body)
            static_bodies.append(body)

            for i in range(len(coords) - 1):
                seg = pymunk.Segment(body, coords[i], coords[i+1], radius=1)
                seg.elasticity = 0.0
                seg.friction = 0.5
                space.add(seg)

    return static_bodies


def apply_edge_attraction(rect_bodies, polygons, force_strength=50000):
    for body in rect_bodies:
        if body.body_type != pymunk.Body.DYNAMIC:
            continue

        pos = body.position
        rect_point = Point(pos[0], pos[1])

        min_dist = float('inf')
        closest_target = None

        for poly in polygons:
            if not poly.is_valid or poly.is_empty:
                continue

            # Create LineString from polygon's exterior
            edge = LineString(poly.exterior.coords)
            nearest_geom = nearest_points(rect_point, edge)[1]

            dist = rect_point.distance(nearest_geom)
            if dist < min_dist:
                min_dist = dist
                closest_target = nearest_geom

        if closest_target is not None:
            dx = closest_target.x - pos[0]
            dy = closest_target.y - pos[1]
            dist_sq = dx**2 + dy**2

            if dist_sq == 0:
                continue

            fx = (dx / dist_sq) * force_strength
            fy = (dy / dist_sq) * force_strength
            body.apply_force_at_world_point((fx, fy), pos)


def add_rectangle(space, pos):
    width, height = RECT_SIZE
    mass = 1
    moment = pymunk.moment_for_box(mass, (width, height))
    body = pymunk.Body(mass, moment)

    # Flip Y coordinate from Pygame to Pymunk
    x, y = pos
    # y = HEIGHT - y  # Flip only here
    body.position = x, y

    shape = pymunk.Poly.create_box(body, size=(width, height))
    shape.friction = 0.4
    shape.elasticity = 0.0
    # Set the fill color for the rectangle
    shape.color = pygame.Color("lightblue")  # You can choose any color, e.g., (173, 216, 230)
    space.add(body, shape)
    dynamic_bodies.append(body)



def load_labelme_json_to_geodataframe(json_file_path):
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
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
                    print(f"Invalid polygon: {e}")
                    continue

    if not polygons:
        return gpd.GeoDataFrame()

    return gpd.GeoDataFrame({'label': labels, 'geometry': polygons}, crs=None)



def main(json_path):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    draw_options = DrawOptions(screen)

    space = pymunk.Space()
    space.gravity = (0, -900)

    gdf = load_labelme_json_to_geodataframe(json_path)

    if gdf is None or gdf.empty:
        print("No valid polygons to display.")
        return



    shapely_polygons_to_static_segment_edges(gdf, space, scale_factor=1.0)
    mouse_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    mouse_joint = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                add_rectangle(space, mouse_pos)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                mouse_pos = pygame.mouse.get_pos()
                mouse_p = pymunk.Vec2d(mouse_pos[0], mouse_pos[1])
                hit = space.point_query_nearest(mouse_p, 0, pymunk.ShapeFilter())
                if hit and hit.shape.body.body_type == pymunk.Body.DYNAMIC:
                    mouse_body.position = mouse_p
                    # Anchor points for joint also need to be in Pymunk world coordinates
                    world_anchor_on_body = hit.shape.body.world_to_local(mouse_p)
                    mouse_joint = pymunk.PivotJoint(mouse_body, hit.shape.body, (0, 0), hit.shape.body.world_to_local(mouse_p))
                    mouse_joint.max_force = 1e6
                    space.add(mouse_joint)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                if mouse_joint:
                    space.remove(mouse_joint)
                    mouse_joint = None

        screen.fill((240, 240, 240))
        space.debug_draw(draw_options)


        # --- Add this block for drawing white borders ---
        border_color = pygame.Color("white")  # Or use (255, 255, 255)
        

        for body in dynamic_bodies:
            if body.body_type == pymunk.Body.DYNAMIC:
                for shape in body.shapes:
                    if isinstance(shape, pymunk.Poly): # Check if it's a polygon shape
                        # Get vertices in local Pymunk coordinates
                        local_verts = shape.get_vertices()
                        # Transform vertices to Pymunk world coordinates
                        world_verts = [body.local_to_world(lv) for lv in local_verts]
                        
                        # Transform Pymunk world coordinates to Pygame screen coordinates
                        # This involves flipping the y-coordinate as Pymunk's y-axis
                        # typically points upwards, while Pygame's points downwards.
                        # HEIGHT must be the height of your Pygame screen.
                        pygame_verts = []
                        for v in world_verts:
                            # This is the standard conversion used by pymunk.pygame_util.DrawOptions
                            pygame_x = int(v.x)
                            pygame_y = int(v.y) 
                            pygame_verts.append((pygame_x, pygame_y))
                        
                        # Draw the polygon outline on the screen
                        pygame.draw.polygon(screen, border_color, pygame_verts, BORDER_WIDTH)
        # --- End of border drawing block ---


        pygame.display.flip()

        apply_edge_attraction(dynamic_bodies, list(gdf.geometry), force_strength=80000)

        # Update mouse body position
        mouse_pos = pygame.mouse.get_pos()
        mouse_body.position = pymunk.Vec2d(*mouse_pos)

        space.step(1 / 60.0)
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    json_file_path = "map1.json"  # Change this to your JSON file path
    main(json_file_path)
