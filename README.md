# Deep Dream with TensorFlow and InceptionV3

## Overview

This repository contains a Python script for implementing the Deep Dream algorithm using TensorFlow and the InceptionV3 model. Deep Dream enhances patterns and features in an input image based on the neural network's recognition. Additionally, a Streamlit app is provided to interactively generate Deep Dream images.

## Original and Dreamified Images

- **Original Image:**
  
    ![Original Image](https://images.pexels.com/photos/1933239/pexels-photo-1933239.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2)

- **Dreamified Image:**

    <img src="https://i.imgur.com/qpFypMq.png" alt="Dreamified Image" width="100%">

## Customization

- Adjust the `layers_contributions` list to include different layers for the deep dream algorithm.

- Experiment with the `epochs`, `steps_per_epoch`, and `weight` parameters in the `run_gradient_ascent` function for different results.

- Try different input images for diverse deep dream effects.

## Streamlit App

- The Streamlit app allows for interactive Deep Dream image generation. Upload an image, select layers, and adjust parameters using the sidebar sliders.
- The deployed app can be accessed by clicking over [here](https://huggingface.co/spaces/unnati026/DeepDream)
