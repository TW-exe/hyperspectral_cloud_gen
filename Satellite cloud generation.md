python=3.9.18
pytorch=1.10.0

Plot all bands of landsat 9 with and without clouds 
Test each parameter with an example of its function and the control 

## Explain what the parameters of each function does
Args:

        input (Tensor) : input image in shape [B,C,H,W] # batch channel height width
Input of hyper spectral image is a tensor that includes batch, channel, height, and width. Regarding the images collected from landsat 9, such images are not an array of images within a single tif file but multiple images within there respective tif files. This means that the image input in our case is simply just an array of images. 


        max_lvl (float or tuple of floats): Indicates the maximum strength of the cloud (1.0 means that some pixels will be fully non-transparent)
This is the maximum intensity a cloud pixel may have. 1 being completely non transparent 


        min_lvl (float or tuple of floats): Indicates the minimum strength of the cloud (0.0 means that some pixels will have no cloud)
This is the minimum intensity a cloud pixel may have. 0 being completely transparent 


        channel_magnitude (Tensor) : cloud magnitudes in each channel, shape [B,C,1,1]
little information on documentation but it seems that this allows you to manipulate cloud intensities on different channel frequencies 


        clear_threshold (float): An optional threshold for cutting off some part of the initial generated cloud mask
This is a threshold in which the cloud mask needs a minimum amount of clear sky to surpass. If the clear_threshold is 0.2 20 percent of the image has to have no clouds at all.


        shadow_max_lvl (float): Indicates the maximum strength of the cloud (1.0 means that some pixels will be completely black)
This is just how intense the shadow of the cloud is. 


        noise_type (string: 'perlin', 'flex'): Method of noise generation (currently supported: 'perlin', 'flex')
This kwarg allows you to manipulate the type of noise used to generate the cloud coverage. 


        const_scale (bool): If True, the spatial frequencies of the cloud/shadow shape are scaled based on the image size (this makes the cloud preserve its appearance regardless of image resolution)
scales the cloud resolution to the image resolution


        decay_factor (float): decay factor that narrows the spectrum of the generated noise (higher values, such as 2.0 will reduce the amplitude of high spatial frequencies, yielding a 'blurry' cloud)
I do not understand 


        locality degree (int): more local clouds shapes can be achieved by multiplying several random cloud shapes with each other (value of 1 disables this effect, and higher integers correspond to the number of multiplied masks)
Creates more cloud shapes that are more similar to one another. In a sense locality. 


        channel_offset (int): optional offset that can randomly misalign spatially the individual cloud mask channels (by a value in range -channel_offset and +channel_offset)
allows you to misalign the cloud masks on different channels


        blur_scaling (float): Scaling factor for the variance of locally varying Gaussian blur (dependent on cloud thickness). Value of 0 will disable this feature.
Allows you to apply gaussian blur to synthetically manipulate cloud thickness


        cloud_color (bool): If True, it will adjust the color of the cloud based on the mean color of the clear sky image
allows you to manipulate the cloud colour based on the mean colour of the clear sky image 


        return_cloud (bool): If True, it will return a channel-wise cloud mask of shape [height, width, channels] along with the cloudy image



    Returns:
        Tensor: Tensor containing a generated cloudy image (and a cloud mask if return_cloud == True)



## Understanding the q_mag function

The relationship between the reflected cloud power depends on the level of reflected power in that channel. To generate a cloud with similar levels, we can extract the ratio of power between the cloud mask region and the clear-sky region.

This is possible using the functions from `src/band_magnitudes.py`. In the example below, we will extract the relationships based on quantile values, using the `q_mag` function.