import torch
import matplotlib.pyplot as plt

from trulens.visualizations import Visualizer, get_backend
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TRULENS_BACKEND'] = 'pytorch'


class HeatmapVisualizer(Visualizer):
    """
    Visualizes attributions by overlaying an attribution heatmap over the
    original image, similar to how GradCAM visualizes attributions.
    """

    def __init__(
        self,
        overlay_opacity=0.5,
        normalization_type=None,
        blur=10.,
        cmap='jet'
    ):
        super().__init__(
            combine_channels=True,
            normalization_type=normalization_type,
            blur=blur,
            cmap=cmap
        )

        self.default_overlay_opacity = overlay_opacity

    def __call__(
        self,
        attributions,
        x,
        output_file=None,
        imshow=True,
        fig=None,
        return_tiled=False,
        overlay_opacity=None,
        normalization_type=None,
        blur=None,
        cmap=None
    ) -> np.ndarray:
        _, normalization_type, blur, cmap = self._check_args(
            attributions, None, normalization_type, blur, cmap
        )

        # Combine the channels.
        attributions = attributions.mean(
            axis=get_backend().channel_axis, keepdims=True
        )

        # Blur the attributions so the explanation is smoother.
        if blur:
            attributions = self._blur(attributions, blur)

        # Normalize the attributions.
        attributions = self._normalize(attributions, normalization_type)

        tiled_attributions = self.tiler.tile(attributions)

        # Normalize the pixels to be in the range [0, 1].
        x = self._normalize(x, '01')
        tiled_x = self.tiler.tile(x)

        if cmap is None:
            cmap = self.default_cmap

        if overlay_opacity is None:
            overlay_opacity = self.default_overlay_opacity

        # Display the figure:
        

        

        if output_file:
            _fig = plt.figure() if fig is None else fig
            plt.axis('off')
            plt.imshow(tiled_x)
            plt.imshow(tiled_attributions, alpha=overlay_opacity, cmap=cmap)
            plt.savefig(output_file, bbox_inches=0)

        if imshow:
            _fig = plt.figure() if fig is None else fig
            plt.axis('off')
            plt.imshow(tiled_x)
            plt.imshow(tiled_attributions, alpha=overlay_opacity, cmap=cmap)
            plt.show()

        elif fig is None:
            _fig = plt.figure() if fig is None else fig
            plt.close(_fig)

        return tiled_x, tiled_attributions if return_tiled else attributions


G = [0, 255, 0]
R = [255, 0, 0]


def convert_to_gray_scale(attributions):
    return np.average(attributions, axis=2)


def linear_transform(attributions, clip_above_percentile=99.9, clip_below_percentile=70, low=0.2, plot_distribution=False):  # 99.9 70.0
    m = compute_threshold_by_top_percentage(
        attributions, percentage=100-clip_above_percentile, plot_distribution=plot_distribution)
    e = compute_threshold_by_top_percentage(
        attributions, percentage=100-clip_below_percentile, plot_distribution=plot_distribution)
    transformed = (1 - low) * (np.abs(attributions) - e) / (m - e) + low
    transformed *= np.sign(attributions)
    transformed *= (transformed >= low)
    transformed = np.clip(transformed, 0.0, 1.0)
    return transformed


def compute_threshold_by_top_percentage(attributions, percentage=60, plot_distribution=True):
    if percentage < 0 or percentage > 100:
        raise ValueError('percentage must be in [0, 100]')
    if percentage == 100:
        return np.min(attributions)
    flat_attributions = attributions.flatten()
    attribution_sum = np.sum(flat_attributions)
    sorted_attributions = np.sort(np.abs(flat_attributions))[::-1]
    cum_sum = 100.0 * np.cumsum(sorted_attributions) / attribution_sum
    threshold_idx = np.where(cum_sum >= percentage)[0][0]
    threshold = sorted_attributions[threshold_idx]
    if plot_distribution:
        raise NotImplementedError
    return threshold


def polarity_function(attributions, polarity):
    if polarity == 'positive':
        return np.clip(attributions, 0, 1)
    elif polarity == 'negative':
        return np.clip(attributions, -1, 0)
    else:
        raise NotImplementedError


def overlay_function(attributions, image):
    return np.clip(0.7 * image + 0.5 * attributions, 0, 255)


def visualize(attributions, image, positive_channel=G, negative_channel=R, polarity='positive',
              clip_above_percentile=99.9, clip_below_percentile=0, morphological_cleanup=False,
              structure=np.ones((3, 3)), outlines=False, outlines_component_percentage=90, overlay=True,
              mask_mode=False, plot_distribution=False):
    if polarity == 'both':
        raise NotImplementedError

    elif polarity == 'positive':
        attributions = polarity_function(attributions, polarity=polarity)
        channel = positive_channel

    # convert the attributions to the gray scale
    attributions = convert_to_gray_scale(attributions)
    attributions = linear_transform(attributions, clip_above_percentile,
                                    clip_below_percentile, 0.0, plot_distribution=plot_distribution)
    attributions = np.power(attributions, 0.75)
    attributions_mask = attributions.copy()
    if morphological_cleanup:
        raise NotImplementedError
    if outlines:
        raise NotImplementedError
    attributions = np.expand_dims(attributions, 2) * channel
    if overlay:
        if mask_mode == False:
            attributions = overlay_function(attributions, image)
        else:
            attributions = np.expand_dims(attributions_mask, 2)
            attributions = np.clip(attributions * image, 0, 255)
            attributions = attributions[:, :, (2, 1, 0)]
    return attributions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mask_viz = HeatmapVisualizer(blur=7, normalization_type="signed_max")


def return_hm(ex):
    percentile = 80
    upperbound = 99
    ex = np.mean(ex, axis=0)
    q = np.percentile(ex, percentile)
    u = np.percentile(ex, upperbound)
    # q=0
    ex[ex<q] = q
    ex[ex>u] = u
    ex = (ex-q)/(u-q + 1e-6)
    ex = ex.transpose(1,0)
    return ex

def plot_axes_hm(attribution,img,axes):
    # im_, mask = mask_viz(attribution, img.cpu().detach(
    #     ).numpy(), overlay_opacity=0.5, imshow=False, return_tiled=True)
    temp = attribution.squeeze()
    # axes.imshow(im_)
    axes.imshow(return_hm(temp),cmap='gray')
    # axes.imshow(mask, alpha=0.5, cmap='jet')
    return axes

def plot_axes(attribution,img,axes):
    im_, mask = mask_viz(attribution, img.cpu().detach(
        ).numpy(), overlay_opacity=0.5, imshow=False, return_tiled=True)
    axes.imshow(im_)
    axes.imshow(mask, alpha=0.5, cmap='jet')
    return axes


def return_mask(attribution,img):
    im_, mask = mask_viz(attribution, img.cpu().detach(
        ).numpy(), overlay_opacity=0.5, imshow=False, return_tiled=True)
    return mask

def plot_save(attribution,img,name):
    im_, mask = mask_viz(attribution, img.cpu().detach(
        ).numpy(), overlay_opacity=0.5, imshow=False, return_tiled=True)
    plt.figure()
    plt.imshow(im_)
    plt.imshow(mask, alpha=0.5, cmap='jet')
    plt.axis('off')
    plt.savefig(name,bbox_inches='tight',dpi=300)

def plot(attribution,img):
    im_, mask = mask_viz(attribution, img.cpu().detach(
        ).numpy(), overlay_opacity=0.5, imshow=False, return_tiled=True)
    plt.figure()
    plt.imshow(im_)
    print(mask.shape)
    plt.imshow(mask, alpha=0.5, cmap='jet')
    plt.axis('off')
    plt.show()