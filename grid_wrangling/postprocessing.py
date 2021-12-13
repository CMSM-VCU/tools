import pyvista as pv
import numpy as np
import pandas as pd
import os



filename = [x for x in os.listdir('.') if '.h5' in x][0]
folder = filename[:filename.rfind("/")+1]

test_image = False
exaggeration = 1.0


image_size = (600, 1000)

# dataset = "u2"
# dataset_display_name = "Veritcal Displacement"

dataset = "dmg"
dataset_display_name = "Damage"

plotter = pv.Plotter(off_screen=True, window_size=image_size)


# Front view
camera_position = np.array([0,0,1])
camera_look_at = np.array([0,0,0])
camera_up = np.array([0,1,0])
camera_scale = 2



def render_image(hf, timestep):
    print(f"Plotting timestep {timestep}")
    # Get point positions
    position = np.array(hf.loc[timestep, ("x1", "x2", "x3")])
    displacement = np.array(hf.loc[timestep, ("u1", "u2", "u3")])
    displaced_position = position + (displacement * exaggeration)

    # Create pyvista object
    polydata = pv.PolyData(displaced_position)
    polydata[dataset_display_name] = hf.loc[timestep, dataset]

    # Pyvista plotting options
    plotter.add_mesh(polydata,scalar_bar_args={'title': dataset_display_name}, cmap='plasma')
    plotter.show_axes()

    # Camera view stuffs
    plotter.enable_parallel_projection()
    plotter.set_position(camera_position)
    plotter.set_focus(camera_look_at)
    plotter.set_viewup(camera_up)
    plotter.camera.parallel_scale = camera_scale
    plotter.enable_anti_aliasing()

    # Timestep in upper right corner
    plotter.add_text(f"{timestep}", position='upper_left', color="white")
    
    # plotter.show()
    plotter.screenshot(f"{folder}{dataset_display_name}/{timestep}.png",
                        transparent_background=True,
                        window_size=image_size)
    plotter.clear()




if __name__ == "__main__":
    print("Reading in data...")
    hf = pd.read_hdf(filename, key="data")
    print("Data has been read")
    timesteps = list(hf.index.levels[0])
    data = list(hf.columns)
    print(f"Datasets: {data}")

    try:
        os.mkdir(f"{folder}{dataset_display_name}")
    except FileExistsError:
        pass

    if test_image:
        for timestep in timesteps[-1:]:
            render_image(hf, timestep)
    else:
        for timestep in timesteps:
            render_image(hf, timestep)