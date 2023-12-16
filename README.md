<!DOCTYPE html>
<html>

<head>
    <title>Deep Dream with TensorFlow and InceptionV3</title>
    <style>
        img {
            width: 45%; /* Adjust the width as needed */
            margin: 5px;
        }

        .image-container {
            display: flex;
            justify-content: space-around;
        }
    </style>
</head>

<body>

<h1>Deep Dream with TensorFlow and InceptionV3</h1>

<h2>Overview</h2>

<p>This repository contains a Python script for implementing the Deep Dream algorithm using TensorFlow and the InceptionV3 model. Deep Dream enhances patterns and features in an input image based on the neural network's recognition. Additionally, a Streamlit app is provided to interactively generate Deep Dream images.</p>

<h3>Streamlit App</h3>

<h2>Original and Dreamified Images</h2>

<div class="image-container">
    <div>
        <p><strong>Original Image:</strong></p>
        <img src="https://images.pexels.com/photos/1933239/pexels-photo-1933239.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
            alt="Original Image">
    </div>

    <div>
        <p><strong>Dreamified Image:</strong></p>
        <img src="https://i.imgur.com/ncb9YuL.jpg" alt="Dreamified Image">
    </div>
</div>

<h2>Customization</h2>

<ul>
    <li>Adjust the <code>layers_contributions</code> list to include different layers for the deep dream algorithm.</li>

    <li>Experiment with the <code>epochs</code>, <code>steps_per_epoch</code>, and <code>weight</code> parameters in the
        <code>run_gradient_ascent</code> function for different results.</li>

    <li>Try different input images for diverse deep dream effects.</li>
</ul>

<h2>Streamlit App</h2>

<p>The Streamlit app allows for interactive Deep Dream image generation. Upload an image, select layers, and adjust parameters using the sidebar sliders.</p>

<p>The deployed web app can be accessed by clicking <a href='https://huggingface.co/spaces/unnati026/DeepDream'>here</a></p>

</body>

</html>
