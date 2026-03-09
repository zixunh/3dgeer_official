import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.patches as patches


def plot(im, ranges, tile_width, tile_height, isect_ids, flatten_ids, unsorted_isect_ids):
    assert ranges.shape[0] == tile_width * tile_height, f"{ranges.shape[0]} != {tile_width * tile_height}"
    ranges = ranges.cpu().numpy()
    isect_ids = isect_ids.cpu().numpy()
    unsorted_isect_ids = unsorted_isect_ids.cpu().numpy()
    # print("Sort: ", np.allclose(isect_ids, unsorted_isect_ids))


    num_tiles = tile_width * tile_height

    
    tile_ids = isect_ids // (2 ** 32)
    counts_python = np.bincount(tile_ids)
    assert counts_python.size <= num_tiles

    padding = np.zeros((num_tiles,))
    padding[:counts_python.size] = counts_python
    counts_python = padding

    # print("Num associations", isect_ids.shape)
    # print(counts_python.shape)
    # print(ranges.shape)
    # print(ranges)
    # print("Count", counts_python)
    # print("Ranges count", ranges[:, 1] - ranges[:, 0])
    # print(isect_ids // (2 ** 32))
    # print(count.shape)

    plt.hist(ranges[:, 1] - ranges[:, 0])
    plt.show()
    
    # compute tile counts
    counts_cuda = (ranges[:, 1] - ranges[:, 0]).reshape((tile_height, tile_width))

    # create 1 row, 2 columns of subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # subplot 1: the image
    axes[0].imshow(im)  # clip to [0,1]
    axes[0].set_title("Image")
    axes[0].axis("off")  # optional, hides axes
    
    # subplot 2: heatmap of counts
    im2 = axes[1].imshow(counts_cuda, origin="lower", cmap="viridis")
    axes[1].set_title("Tile counts (CUDA)")
    fig.colorbar(im2, ax=axes[1])  # add colorbar for heatmap

    # subplot 3: heatmap of counts (python)
    im3 = axes[2].imshow(counts_python.reshape((tile_height, tile_width)), origin="lower", cmap="viridis")
    axes[2].set_title("Tile counts (Python)")
    fig.colorbar(im3, ax=axes[2])  # add colorbar for heatmap
    
    plt.tight_layout()
    plt.show()

def plot_hist(isect_ids: torch.Tensor, flatten_ids: torch.Tensor):
    x = (isect_ids // (2**32)).cpu().numpy()
    y = flatten_ids.cpu().numpy()
    plt.hist2d(x, y, cmap='plasma')  # bins control resolution
    plt.colorbar(label='Counts')  # shows density
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Heatmap of X-Y Pairs')
    plt.show()

def plot_projection(means3d_viewspace, aabb, means2d, Ks, image_width, image_height, omni_tan_theta, omni_tan_phi):
    print("---------------------------------------------------------")
    means3d_viewspace = means3d_viewspace.cpu().numpy()
    aabb = aabb.cpu().numpy()
    means2d = means2d.cpu().numpy().squeeze()
    Ks = Ks.cpu().numpy().squeeze()
    omni_tan_theta, omni_tan_phi = omni_tan_theta.cpu().numpy(), omni_tan_phi.cpu().numpy()

    # print(omni_tan_theta)

    geer = means3d_viewspace.size != 0

    print("Means 3D Shape:", means3d_viewspace.shape)
    print("Means 2D:", means2d.shape)
    print("Ks shape:", Ks.shape)
    # print(means3d_viewspace[:20])

    depths = np.linalg.norm(means3d_viewspace, axis=1)
    non_zero_3d = means3d_viewspace[depths != 0]
    print("Non-zero depth shape:", non_zero_3d.shape)

    K = Ks
    points2d = K @ non_zero_3d.T
    points2d = (points2d / (points2d[2] + 1e-16)).T[:, :2]

    onscreen_3d = (0 <= points2d[:, 0]) & (points2d[:, 0] < image_width) & (0 <= points2d[:, 1]) & (points2d[:, 1] < image_height)
    means3d2d_onscreen = points2d[onscreen_3d]

    onscreen_2d = (0 <= means2d[:, 0]) & (means2d[:, 0] < image_width) & (0 <= means2d[:, 1]) & (means2d[:, 1] < image_height)
    means2d_onscreen = means2d[onscreen_2d]

    # print("AABB", aabb)

    theta_grid = omni_tan_theta * K[0, 0] + K[0, 2]
    phi_grid = omni_tan_phi * K[1, 1] + K[1, 2]

    print(theta_grid)

    fig, ax = plt.subplots()

    # ax.vlines(theta_grid, 0, image_height, colors='blue', linewidth=1)
    # ax.hlines(phi_grid, 0, image_width, colors='blue', linewidth=1)

    edge_colors = [
        'red',        # classic red
        'blue',       # classic blue
        'green',      # classic green
        'orange',     # bright orange
        'purple',     # deep purple
        'cyan',       # bright cyan
        'magenta',    # pink/magenta
        'yellow',     # bright yellow (may need darker facecolor to see)
        'brown',      # earthy brown
        'black'       # classic black
    ]
    counter = 0
    for mean2d, box in zip(means2d, aabb):
        if not (0 <= mean2d[0] < image_width and 0 <= mean2d[1] < image_height):
            continue
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        if width * height == 0:
            continue
        if not (xmin <= mean2d[0] < xmax and ymin <= mean2d[1] < ymax):
            rect = patches.Rectangle(
                (xmin, ymin),  # bottom-left corner
                width,
                height,
                linewidth=1,
                edgecolor=edge_colors[counter],
                facecolor='none'  # transparent fill
            )
            ax.add_patch(rect)
            ax.scatter([mean2d[0]], [mean2d[1]], c=edge_colors[counter])
            counter += 1
            # print(f"Something's wrong: {counter}")
            if counter >= len(edge_colors):
                break
        # counter += 1
        # if counter >= 1:
        #     break

    ax.set_aspect('equal')  # keeps the rectangles proportional
    plt.xlim(aabb[:, [0,2]].min() - 1, aabb[:, [0,2]].max() + 1)
    plt.ylim(aabb[:, [1,3]].min() - 1, aabb[:, [1,3]].max() + 1)

    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(means2d_onscreen[:, 0], means2d_onscreen[:, 1])
    axes[0].set_title("2D")

    axes[1].scatter(means3d2d_onscreen[:, 0], means3d2d_onscreen[:, 1])
    axes[1].set_title("3D")
    plt.show()
    
    print("Max depth:", np.max(depths))
    print("Num less than 5:", f"{np.count_nonzero(depths < 5)}")
    plt.hist(depths, range=(0, 5))
    plt.show()
    # print(means3d_viewspace[0])