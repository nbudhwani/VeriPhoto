// Log a message to verify that the ONNX Runtime Web is available
console.log("ONNX Runtime Web imported successfully!", ort);  // 'ort' is available globally now

document.addEventListener('DOMContentLoaded', function () {
  // Set up the button to execute the content script
  document.getElementById('refresh').addEventListener('click', function () {
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      chrome.scripting.executeScript({
        target: { tabId: tabs[0].id },
        files: ['content.js']
      });
    });
  });

  loadModelAndRunInference();
});

// Load the ONNX model and run inference on images
async function loadModelAndRunInference() {
  try {
    // Load the ONNX model
    const session = await ort.InferenceSession.create('model.onnx'); // Replace with your ONNX model file name
    console.log('ONNX model loaded successfully!');

    // Get images from the DOM and run inference
    document.querySelectorAll('img').forEach(async (img) => {
      const imageData = getImageData(img);
      console.log('It got to before the imageData break!');
      if (imageData) {
        console.log("Image data prepared for inference.");
        // Run inference on the image data
        const isDeepfake = await runInference(session, imageData);
        console.log("Inference result:", isDeepfake);

        // Highlight the image if it is a deepfake
        if (isDeepfake) {
          img.style.border = "5px solid red";  // Highlight detected deepfakes
          console.log("Deepfake detected for this image.");
        } else {
          console.log("This image is not a deepfake.");
        }
      }
    });
  } catch (error) {
    console.error("Failed to load ONNX model or run inference:", error);
  }
}

// Function to run inference using the ONNX model
async function runInference(session, imageData) {
  try {
    // Convert image data to a format suitable for ONNX model input
    const inputTensor = preprocessImageData(imageData);
    const feeds = { input: inputTensor }; // Adjust input name based on your model

    // Run inference
    const output = await session.run(feeds);
    console.log("Inference executed successfully!");

    // Example: Log the output data
    const probability = output['output'].data[0]; // Assuming 'output' is the correct output name
    console.log("Model output probability of being a deepfake:", probability);

    return probability > 0.5;  // Example threshold to decide deepfake
  } catch (error) {
    console.error("Error during inference:", error);
    return null;
  }
}

// Function to preprocess image data for ONNX model
function preprocessImageData(imageData) {
  const width = imageData.width;
  const height = imageData.height;
  const data = imageData.data;
  
  // Resize and normalize image data
  const resizedData = resizeImageData(data, width, height, 224, 224); // Resize to 224x224
  const floatData = normalizeImageData(resizedData);  // Normalize with mean and std

  return new ort.Tensor('float32', floatData, [1, 3, 224, 224]); // Adjust shape based on model requirements
}

// Function to resize image data
function resizeImageData(data, originalWidth, originalHeight, targetWidth, targetHeight) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = targetWidth;
  canvas.height = targetHeight;

  const imageData = new ImageData(new Uint8ClampedArray(data), originalWidth, originalHeight);
  ctx.putImageData(imageData, 0, 0);

  ctx.drawImage(canvas, 0, 0, originalWidth, originalHeight, 0, 0, targetWidth, targetHeight);
  return ctx.getImageData(0, 0, targetWidth, targetHeight).data;
}

// Function to normalize image data
function normalizeImageData(data) {
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  const normalizedData = new Float32Array(data.length / 4 * 3);  // 3 channels, discard alpha

  for (let i = 0; i < data.length / 4; i++) {
    normalizedData[i * 3 + 0] = (data[i * 4 + 0] / 255.0 - mean[0]) / std[0];  // Red
    normalizedData[i * 3 + 1] = (data[i * 4 + 1] / 255.0 - mean[1]) / std[1];  // Green
    normalizedData[i * 3 + 2] = (data[i * 4 + 2] / 255.0 - mean[2]) / std[2];  // Blue
  }

  return normalizedData;
}

// Function to get image data from the DOM
function getImageData(imgElement) {
  const canvas = document.createElement('canvas');
  canvas.width = 224;  // Adjust size as needed
  canvas.height = 224;
  const ctx = canvas.getContext('2d');

  // Handle cross-origin images
  imgElement.crossOrigin = 'Anonymous';  // Set crossOrigin to 'Anonymous'

  // Draw the image on the canvas
  ctx.drawImage(imgElement, 0, 0, 224, 224);  // Resize and draw

  // Try to get image data safely
  try {
    return ctx.getImageData(0, 0, 224, 224);
  } catch (e) {
    console.error('Error accessing canvas data:', e);
    return null;
  }
}