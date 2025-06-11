import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FormatStrFormatter


def plot_3d_motion(save_path, kinematic_tree, mp_joints, title, figsize=(10, 10), fps=120, radius=6, mode='interaction'):
    """
    Function to plot an interaction between two agents in 3D in matplotlib
        :param save_path: path to save the animation
        :param kinematic_tree: kinematic tree of the motion
        :param mp_joints: list of motion data for each agent
        :param title: title of the plot
        :param figsize: size of the figure
        :param fps: frames per second of the animation
        :param radius: radius of the plot
        :param mode: mode of the plot
    """
    matplotlib.use('Agg')

    # Define initial limits of the plot
    def init():
        ax.set_xlim3d([-radius / 4, radius / 4])
        ax.set_ylim3d([0, radius / 2])
        ax.set_zlim3d([0, radius / 2])
        ax.grid(b=False)

    # Funtion to plot a floor in the animation
    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)


    # Create the figure and axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    init()

    # Offsets and colors
    mp_offset = list(range(-len(mp_joints)//2, len(mp_joints)//2, 1))
    colors = ['red', 'green', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    mp_colors = [[colors[i]] * 15 for i in range(len(mp_offset))]

    # Store the data for each agent
    mp_data = []
    for i,joints in enumerate(mp_joints):

        data = joints.copy().reshape(len(joints), -1, 3)

        MINS = data.min(axis=0).min(axis=0)
        MAXS = data.max(axis=0).max(axis=0)

        height_offset = MINS[1]
        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]

        mp_data.append({"joints":data,
                        "MINS":MINS,
                        "MAXS":MAXS,
                        "trajec":trajec, })
        
    def update(index):
        """
        Update function for the matplotlib animation
            :param index: index of the frame
        """
        # Update the progress bar
        bar.update(1)

        # Clear the axis and setting initial parameters
        ax.clear()
        plt.axis('off')

        # Calculate midpoint between two agents for current frame
        if len(mp_joints) > 1:
            mid_x = (mp_data[0]["joints"][index, 0, 0] + mp_data[1]["joints"][index, 0, 0]) / 2
            mid_y = (mp_data[0]["joints"][index, 0, 1] + mp_data[1]["joints"][index, 0, 1]) / 2
            mid_z = (mp_data[0]["joints"][index, 0, 2] + mp_data[1]["joints"][index, 0, 2]) / 2
        else:  
            mid_x = (mp_data[0]["joints"][index, 0, 0] + mp_data[0]["joints"][index, 0, 0]) / 2
            mid_y = (mp_data[0]["joints"][index, 0, 1] + mp_data[0]["joints"][index, 0, 1]) / 2
            mid_z = (mp_data[0]["joints"][index, 0, 2] + mp_data[0]["joints"][index, 0, 2]) / 2

        # Set camera view to focus on midpoint
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        
        # Adjust limits to center around midpoint
        ax.set_xlim3d([mid_x - radius / 4, mid_x + radius / 4])
        ax.set_ylim3d([mid_y - radius / 4, mid_y + radius / 4])
        ax.set_zlim3d([mid_z - radius / 4, mid_z + radius / 4])
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # Plot the floor
        plot_xzPlane(-3, 3, 0, -3, 3)

        # Plot each of the persons in the motion
        for pid,data in enumerate(mp_data):
            for i, (chain, color) in enumerate(zip(kinematic_tree, mp_colors[pid])):
                linewidth = 3.0
                ax.plot3D(data["joints"][index, chain, 0], 
                          data["joints"][index, chain, 1], 
                          data["joints"][index, chain, 2], 
                          linewidth=linewidth,
                          color=color,
                          alpha=1)

    # Generate animation
    frame_number = min([data.shape[0] for data in mp_joints])
    bar = tqdm(total=frame_number+1)
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close()


def plot_influence_1(history1, history2, save_path):
    # Define the custom colormap
    cmap_custom = LinearSegmentedColormap.from_list("custom_cmap", ['#8A033E', '#C44E30', '#FCC00B'])
    
    with PdfPages(f"{save_path}_influence1.pdf") as pdf:
        plot_width = 6  # Width for more space
        plot_height = 5
        font_size = 15  # Font size
        label_padding = 10  # Label padding
        max_val_1 = history1[:, 0, :, 0].max()

        # Create the plot
        fig1 = plt.figure(figsize=(plot_width, plot_height))
        ax1 = fig1.add_subplot(111)
        
        # Generate a range of counter values from 1000 to 0
        x_values = np.linspace(0, 1000, 1000)
        x_values_2 = np.linspace(50, 0, 50)

        # Normalize based on the range of values across both histories for consistent color mapping
        norm = Normalize(vmin=history1.min(), vmax=history1.max())

        # Main plot with gradient color
        for i in range(history1.shape[0] - 1):
            val = history1[i, 0, 0, 0]
            color = cmap_custom(norm(val))  # Color based on normalized value
            ax1.plot([i, i+1], history1[i:i+2, 0, 0, 0], color=color, linewidth=3)
        
        # Set axis labels, ticks, and formatting
        ax1.yaxis.set_ticks(np.linspace(0, max_val_1, 3))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.xaxis.set_ticks(np.linspace(0, history1.shape[0], 4).astype(int))
        ax1.set_xlabel('Denoising Step', fontsize=font_size, labelpad=label_padding)
        ax1.tick_params(axis='both', which='major', labelsize=font_size)

        # Adjust layout and save the figure
        fig1.tight_layout()
        pdf.savefig(fig1)
        plt.close(fig1)

def plot_influence_2(history1, history2, save_path):

    # Create a custom colormap from the start and end colors (hex codes supported)
    cmap_custom = LinearSegmentedColormap.from_list("custom_cmap", ['#8A033E', '#C44E30', '#FCC00B'])

    with PdfPages(f"{save_path}_influence2.pdf") as pdf:
        x = np.arange(history1.shape[0])
        y = np.arange(history1.shape[2])
        X, Y = np.meshgrid(x, y, indexing='ij')

        max_val_1 = history1[:, 0, :, 0].max()

        plot_width = 18  # Increased width for more space
        plot_height = 6
        plot_box_aspect = [1.5, 1.5, 0.9]  # Adjusted aspect for a wider plot
        font_size = 14  # Set desired font size here
        label_padding = 10  # Set label padding here

        # Plot 1: Denoising - Timestep (Person 1)
        fig1 = plt.figure(figsize=(plot_width, plot_height))
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.set_box_aspect(plot_box_aspect)
        surf1 = ax1.plot_surface(X, Y, history1[:, 0, :, 0], cmap=cmap_custom,edgecolor='none')
        ax1.set_xlabel('Denoising Step', fontsize=font_size, labelpad=label_padding)
        ax1.set_ylabel('Timestep', fontsize=font_size, labelpad=label_padding)
        ax1.set_zlim(0, max_val_1)
        ax1.zaxis.set_label_position('lower')  # Move z-axis label to the left
        ax1.zaxis.set_ticks_position('lower')  # Move z-axis label to the left
        # Set z-axis ticks with 2 decimal places
        ax1.zaxis.set_ticks(np.linspace(0, max_val_1, 3))  # Set tick positions
        ax1.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Limit ticks to 2 decimal places
        ax1.xaxis.set_ticks(np.linspace(0, 50, 4).astype(int))  # Set z-axis ticks (5 ticks from 0 to 1)
        ax1.yaxis.set_ticks(np.linspace(0, 300, 5).astype(int))  # Set z-axis ticks (5 ticks from 0 to 1)
        ax1.tick_params(axis='both', which='major', labelsize=font_size)  # Set tick label size
        #fig1.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        pdf.savefig(fig1)
        plt.close(fig1)

def plot_influence_3(history1, history2, save_path):

    # Concatenate to recover the original 23 values along the last axis
    history1 = np.concatenate([history1[..., :66:3], history1[..., 192][..., np.newaxis]], axis=-1)
    print(history1.shape)

   # Create a custom colormap from the start and end colors (hex codes supported)
    cmap_custom = LinearSegmentedColormap.from_list("custom_cmap", ['#084E8C','#A7BFBB','#B3CE75'])

    with PdfPages(f"{save_path}_influence3.pdf") as pdf:
        x = np.arange(history1.shape[0])
        y = np.arange(history1.shape[-1])
        X, Y = np.meshgrid(x, y, indexing='ij')

        max_val_1 = history1[:, 0, 0, :].max()

        plot_width = 18  # Increased width for more space
        plot_height = 6
        plot_box_aspect = [1.5, 1.5, 0.9]  # Adjusted aspect for a wider plot
        font_size = 14  # Set desired font size here
        label_padding = 10  # Set label padding here

         # Plot 2: Denoising - Joint (Person 1)
        fig2 = plt.figure(figsize=(plot_width, plot_height))
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.set_box_aspect(plot_box_aspect)
        surf2 = ax2.plot_surface(X, Y, history1[:, 0, 0, :], cmap=cmap_custom)
        ax2.set_xlabel('Denoising Step', fontsize=font_size, labelpad=label_padding)
        ax2.set_ylabel('Joint', fontsize=font_size, labelpad=label_padding)
        ax2.set_zlim(0, max_val_1)
        ax2.zaxis.set_label_position('lower')  # Move z-axis label to the left
        ax2.zaxis.set_ticks_position('lower')  # Move z-axis label to the left
        # Set z-axis ticks with 2 decimal places
        ax2.zaxis.set_ticks(np.linspace(0, max_val_1, 3))  # Set tick positions
        ax2.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Limit ticks to 2 decimal places
        ax2.xaxis.set_ticks(np.linspace(0, 50, 4).astype(int))  # Set z-axis ticks (5 ticks from 0 to 1)
        ax2.yaxis.set_ticks(np.linspace(0, 23, 5).astype(int))  # Set z-axis ticks (5 ticks from 0 to 1)
        ax2.tick_params(axis='both', which='major', labelsize=font_size)
        pdf.savefig(fig2)
        plt.close(fig2)

def plot_influence_4(history1, history2, save_path, plot_person1_only=True):

    # Concatenate to recover the original 23 values along the last axis
    history1 = np.concatenate([history1[..., :66:3], history1[..., 192][..., np.newaxis]], axis=-1)

   # Create a custom colormap from the start and end colors (hex codes supported)
    cmap_custom = LinearSegmentedColormap.from_list("custom_cmap", ['#8A033E', '#C44E30', '#FCC00B'])

    with PdfPages(f"{save_path}_influence4.pdf") as pdf:
        x1 = np.arange(history1.shape[0])
        y1 = np.arange(history2.shape[2])
        X1, Y1 = np.meshgrid(x1, y1, indexing='ij')

        x2 = np.arange(history1.shape[0])
        y2 = np.arange(history1.shape[-1])
        X2, Y2 = np.meshgrid(x2, y2, indexing='ij')

        max_val_1 = history1[:, 0, :, :].mean(axis=-1).max()
        max_val_2 = history1[:, 0, :, :].mean(axis=-2).max()

        plot_width = 18  # Increased width for more space
        plot_height = 6
        plot_box_aspect = [1.5, 1.5, 0.9]  # Adjusted aspect for a wider plot
        font_size = 14  # Set desired font size here
        label_padding = 10  # Set label padding here

        # Plot 1: Denoising - Timestep (Person 1)
        fig1 = plt.figure(figsize=(plot_width, plot_height))
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.set_box_aspect(plot_box_aspect)
        surf1 = ax1.plot_surface(X1, Y1, history1[:, 0, :, :].mean(axis=-1), cmap=cmap_custom)
        ax1.set_xlabel('Denoising Step', fontsize=font_size, labelpad=label_padding)
        ax1.set_ylabel('Timestep', fontsize=font_size, labelpad=label_padding)
        ax1.set_zlim(0, max_val_1)
        ax1.zaxis.set_label_position('lower')  # Move z-axis label to the left
        ax1.zaxis.set_ticks_position('lower')  # Move z-axis label to the left
        # Set z-axis ticks with 2 decimal places
        ax1.zaxis.set_ticks(np.linspace(0, max_val_1, 3))  # Set tick positions
        ax1.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Limit ticks to 2 decimal places
        ax1.xaxis.set_ticks(np.linspace(0, 50, 4).astype(int))  # Set z-axis ticks (5 ticks from 0 to 1)
        ax1.yaxis.set_ticks(np.linspace(0, 300, 5).astype(int))  # Set z-axis ticks (5 ticks from 0 to 1)
        ax1.tick_params(axis='both', which='major', labelsize=font_size)  # Set tick label size
        #fig1.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        pdf.savefig(fig1)
        plt.close(fig1)

        # Plot 2: Denoising - Joint (Person 1)
        fig2 = plt.figure(figsize=(plot_width, plot_height))
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.set_box_aspect(plot_box_aspect)
        surf2 = ax2.plot_surface(X2, Y2, history1[:, 0, :, :].mean(axis=-2), cmap=cmap_custom)
        ax2.set_xlabel('Denoising Step', fontsize=font_size, labelpad=label_padding*2)
        ax2.set_ylabel('Joint', fontsize=font_size, labelpad=label_padding)
        ax2.set_zlim(0, max_val_2)
        ax2.zaxis.set_label_position('lower')  # Move z-axis label to the left
        ax2.zaxis.set_ticks_position('lower')  # Move z-axis label to the left
        # Set z-axis ticks with 2 decimal places
        ax2.zaxis.set_ticks(np.linspace(0, max_val_2, 3))  # Set tick positions
        ax2.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Limit ticks to 2 decimal places
        ax2.xaxis.set_ticks(np.linspace(0, 50, 4).astype(int))  # Set z-axis ticks (5 ticks from 0 to 1)
        ax2.yaxis.set_ticks(np.linspace(0, 23, 5).astype(int))  # Set z-axis ticks (5 ticks from 0 to 1)
        ax2.tick_params(axis='both', which='major', labelsize=font_size)
        #fig2.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        pdf.savefig(fig2)
        plt.close(fig2)

        # Only plot Person 2 if plot_person1_only is False
        if not plot_person1_only:
            # Plot 3: Denoising - Timestep (Person 2)
            fig3 = plt.figure(figsize=(plot_width, plot_height))
            ax3 = fig3.add_subplot(111, projection='3d')
            ax3.set_box_aspect(plot_box_aspect)
            surf3 = ax3.plot_surface(X1, Y1, history2[:, 0, :, :].mean(axis=-1), cmap='plasma')
            ax3.set_xlabel('Denoising Step', fontsize=font_size, labelpad=label_padding)
            ax3.set_ylabel('Timestep', fontsize=font_size, labelpad=label_padding)
            ax3.set_zlabel('Weight', fontsize=font_size, labelpad=label_padding)
            ax3.set_zlim(0, 1)
            ax3.tick_params(axis='both', which='major', labelsize=font_size)
            fig3.tight_layout()
            fig3.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            pdf.savefig(fig3)
            plt.close(fig3)

            # Plot 4: Denoising - Joint (Person 2)
            fig4 = plt.figure(figsize=(plot_width, plot_height))
            ax4 = fig4.add_subplot(111, projection='3d')
            ax4.set_box_aspect(plot_box_aspect)
            surf4 = ax4.plot_surface(X2, Y2, history2[:, 0, :, :].mean(axis=-2), cmap='plasma')
            ax4.set_xlabel('Denoising Step', fontsize=font_size, labelpad=label_padding)
            ax4.set_ylabel('Joint', fontsize=font_size, labelpad=label_padding)
            ax4.set_zlabel('Weight', fontsize=font_size, labelpad=label_padding)
            ax4.set_zlim(0, 1)
            ax4.tick_params(axis='both', which='major', labelsize=font_size)
            fig4.tight_layout()
            fig4.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            pdf.savefig(fig4)
            plt.close(fig4)


def plot_influence(history1, history2, mode, save_path):
    if mode == 1:
        plot_influence_1(history1, history2, save_path)
    elif mode == 2:
        plot_influence_2(history1, history2, save_path)
    elif mode == 3:
        plot_influence_3(history1, history2, save_path)
    elif mode == 4:
        plot_influence_4(history1, history2, save_path)
    else:
        raise ValueError("Mode not supported")

